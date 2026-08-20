"""Ablate the 749 external MPCD broken-conductor training images.

The experiment keeps validation/test data and every hyperparameter fixed.  It
only changes whether the training list contains the 749 current MPCD images.
No dataset file is moved, deleted, or rewritten.

By default, the script runs YOLOv8-L with seeds 42, 3407 and 2026.  Compatible
"with MPCD" weights from the existing three-seed experiment are reused, so a
normal run only needs to train the three "without MPCD" models.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any

import yaml

from run_detection_baselines import load_model, serialize_metrics


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CONDITIONS = ("without_mpcd", "with_mpcd")
CONDITION_LABELS = {
    "without_mpcd": "Without 749 external images",
    "with_mpcd": "With 749 external images",
}
OVERALL_KEYS = {
    "overall_precision": "metrics/precision(B)",
    "overall_recall": "metrics/recall(B)",
    "overall_mAP50": "metrics/mAP50(B)",
    "overall_mAP50-95": "metrics/mAP50-95(B)",
}
TARGET_KEYS = {
    "broken_precision": "precision",
    "broken_recall": "recall",
    "broken_mAP50": "mAP50",
    "broken_mAP50-95": "mAP50-95",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Three-seed ablation of the 749 external MPCD broken-conductor "
            "training images using YOLOv8-L."
        )
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "external_data" / "mpcd" / "import_manifest.jsonl",
        help="Auditable MPCD import manifest.",
    )
    parser.add_argument("--expected-external-count", type=int, default=749)
    parser.add_argument(
        "--external-prefix",
        default="mpcd_train_",
        help=(
            "Safe filename-prefix fallback used only when the import manifest "
            "is unavailable."
        ),
    )
    parser.add_argument("--target-class-id", type=int, default=12)
    parser.add_argument(
        "--seeds",
        nargs=3,
        type=int,
        default=[42, 3407, 2026],
        metavar=("SEED1", "SEED2", "SEED3"),
    )
    parser.add_argument("--model", default="yolov8l.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use AMP; disable with --no-amp if numerical errors occur.",
    )
    parser.add_argument(
        "--project", type=Path, default=ROOT / "runs" / "mpcd_749_ablation"
    )
    parser.add_argument(
        "--reuse-existing-with",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse protocol-compatible YOLOv8-L weights already trained with "
            "the complete train/images directory."
        ),
    )
    parser.add_argument(
        "--resume-interrupted",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
        help="Run both conditions by default; useful for resuming one side only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and audit the split files, but do not load or train a model.",
    )
    return parser.parse_args()


def resolve_dataset_root(data_yaml: Path) -> tuple[Path, dict[str, Any]]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    source = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    for key in ("val", "test", "nc", "names"):
        if key not in source:
            raise KeyError(f"Missing {key!r} in {data_yaml}")
    declared = Path(source.get("path", data_yaml.parent))
    if not declared.is_absolute():
        declared = data_yaml.parent / declared
    return declared.resolve(), source


def resolve_source(value: Any, dataset_root: Path) -> str:
    text = str(value)
    if "://" in text:
        return text
    path = Path(text)
    if not path.is_absolute():
        path = dataset_root / path
    return str(path.resolve())


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def read_external_set(
    manifest: Path,
    root: Path,
    train_images: Path,
    train_labels: Path,
    expected_count: int,
    target_class_id: int,
    external_prefix: str,
) -> tuple[set[Path], list[dict[str, Any]], int, int, str]:
    rows: list[dict[str, Any]] = []
    if manifest.is_file():
        selection_method = "import_manifest"
        for line_number, raw in enumerate(
            manifest.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid manifest JSON at line {line_number}: {exc}"
                ) from exc
    else:
        # The import tool gives every MPCD destination this dedicated prefix.
        # This fallback remains strict: an exact expected count, matching label
        # files and target-class-only annotations are all checked below.
        selection_method = "strict_filename_prefix_fallback"
        fallback_images = sorted(
            path.resolve()
            for path in train_images.iterdir()
            if path.is_file()
            and path.name.startswith(external_prefix)
            and path.suffix.lower() in IMAGE_SUFFIXES
        )
        print(f"[warning] MPCD manifest is unavailable: {manifest}")
        print(
            f"[fallback] Selecting only train/images/{external_prefix}* "
            f"and applying strict count/label checks."
        )
        for image in fallback_images:
            label = (train_labels / f"{image.stem}.txt").resolve()
            rows.append(
                {
                    "destination_image": image.relative_to(root).as_posix(),
                    "destination_label": label.relative_to(root).as_posix(),
                    "source_dataset": "MPCD",
                    "selection_method": selection_method,
                }
            )

    selected: list[dict[str, Any]] = []
    selected_images: set[Path] = set()
    total_boxes = 0
    for row in rows:
        image = (root / Path(str(row["destination_image"]))).resolve()
        label = (root / Path(str(row["destination_label"]))).resolve()

        # Six near-duplicates were moved to _quarantine after import.  Only
        # manifest rows whose current destination remains in train/ are part of
        # this 749-image ablation.
        if not is_relative_to(image, train_images):
            continue
        if not is_relative_to(label, train_labels):
            raise ValueError(f"External label is outside train/labels: {label}")
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing external image/label pair: {image}, {label}")
        if image in selected_images:
            raise ValueError(f"Duplicate destination in MPCD manifest: {image}")

        nonempty = [line for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not nonempty:
            raise ValueError(f"Empty external label: {label}")
        for line_number, line in enumerate(nonempty, start=1):
            parts = line.split()
            if len(parts) != 5 or int(parts[0]) != target_class_id:
                raise ValueError(
                    f"Unexpected label at {label}:{line_number}; expected only "
                    f"class {target_class_id}."
                )
        total_boxes += len(nonempty)
        selected_images.add(image)
        selected.append({**row, "resolved_image": str(image), "resolved_label": str(label)})

    if len(selected_images) != expected_count:
        raise ValueError(
            f"Expected {expected_count} current MPCD training images, found "
            f"{len(selected_images)}. Do not run an incomparable ablation."
        )
    return selected_images, selected, total_boxes, len(rows), selection_method


def list_training_images(train_images: Path) -> list[Path]:
    if not train_images.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {train_images}")
    images = sorted(
        path.resolve()
        for path in train_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No training images found in {train_images}")
    return images


def write_lines(paths: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(path.as_posix() for path in paths) + "\n", encoding="utf-8"
    )


def prepare_conditions(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    data_yaml = args.data.resolve()
    dataset_root, source = resolve_dataset_root(data_yaml)
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    all_images = list_training_images(train_images)

    external, audit_rows, external_boxes, source_row_count, selection_method = read_external_set(
        args.manifest.resolve(),
        dataset_root,
        train_images.resolve(),
        train_labels.resolve(),
        args.expected_external_count,
        args.target_class_id,
        args.external_prefix,
    )
    all_set = set(all_images)
    if not external.issubset(all_set):
        missing = sorted(str(path) for path in external - all_set)
        raise ValueError(f"Manifest images are absent from train/images: {missing[:5]}")

    groups = {
        "without_mpcd": [path for path in all_images if path not in external],
        "with_mpcd": all_images,
    }
    if set(groups["with_mpcd"]) - set(groups["without_mpcd"]) != external:
        raise AssertionError("The two training conditions differ by more than MPCD images.")

    config_dir = args.project / "_configs"
    list_paths: dict[str, Path] = {}
    yaml_paths: dict[str, Path] = {}
    for condition, images in groups.items():
        list_path = config_dir / f"train_{condition}.txt"
        write_lines(images, list_path)
        list_paths[condition] = list_path.resolve()

        config = {
            "path": str(dataset_root),
            "train": str(list_path.resolve()),
            "val": resolve_source(source["val"], dataset_root),
            "test": resolve_source(source["test"], dataset_root),
            "nc": int(source["nc"]),
            "names": source["names"],
        }
        yaml_path = config_dir / f"data_{condition}.yaml"
        yaml_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        yaml_paths[condition] = yaml_path.resolve()

    for split in ("val", "test"):
        resolved = resolve_source(source[split], dataset_root)
        if "://" not in resolved and not Path(resolved).exists():
            raise FileNotFoundError(f"{split} source not found: {resolved}")

    audit = {
        "source_data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "manifest": str(args.manifest.resolve()),
        "selection_method": selection_method,
        "source_rows": source_row_count,
        "current_external_images": len(external),
        "current_external_boxes": external_boxes,
        "target_class_id": args.target_class_id,
        "target_class_name": str(source["names"][args.target_class_id]),
        "with_mpcd_train_images": len(groups["with_mpcd"]),
        "without_mpcd_train_images": len(groups["without_mpcd"]),
        "difference_images": len(external),
        "condition_lists": {key: str(value) for key, value in list_paths.items()},
        "external_samples": audit_rows,
    }
    audit_path = config_dir / "mpcd_749_split_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return dataset_root, yaml_paths, audit


def same_number(actual: Any, expected: float, tolerance: float = 1e-12) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def protocol_mismatches(
    args_yaml: Path,
    args: argparse.Namespace,
    seed: int,
    expected_train_suffix: str,
) -> list[str]:
    if not args_yaml.is_file():
        return ["missing args.yaml"]
    try:
        saved = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot read args.yaml: {exc}"]

    exact = {
        "seed": seed,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "optimizer": "AdamW",
        "amp": args.amp,
        "deterministic": True,
        "mosaic": 0.0,
        "mixup": 0.0,
        "close_mosaic": 0,
    }
    mismatches = [
        f"{key}: saved={saved.get(key)!r}, requested={value!r}"
        for key, value in exact.items()
        if saved.get(key) != value
    ]
    numeric = {
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
    }
    mismatches.extend(
        f"{key}: saved={saved.get(key)!r}, requested={value!r}"
        for key, value in numeric.items()
        if not same_number(saved.get(key), value)
    )

    # Verify which training source produced a reused weight.  Copied AutoDL
    # runs may retain /root/... paths, so also inspect the matching local
    # _configs file and compare the portable path suffix.
    saved_data = Path(str(saved.get("data", "")))
    candidates = [saved_data]
    if saved_data.name:
        candidates.append(args_yaml.parent.parent / "_configs" / saved_data.name)
    saved_train: str | None = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            config = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if "train" in config:
            saved_train = str(config["train"]).replace("\\", "/").rstrip("/")
            break
    expected = expected_train_suffix.replace("\\", "/").rstrip("/")
    if saved_train is None:
        mismatches.append("cannot verify the saved training source")
    elif not saved_train.endswith(expected):
        mismatches.append(
            f"training source: saved={saved_train!r}, expected suffix={expected!r}"
        )
    return mismatches


def existing_with_run(seed: int, args: argparse.Namespace) -> Path:
    if seed == 42:
        return ROOT / "runs" / "baselines" / "baseline_yolov8_l"
    return ROOT / "runs" / "three_seed_detection" / f"baseline_yolov8_l_seed{seed}"


def run_name(condition: str, seed: int) -> str:
    return f"yolov8_l_{condition}_seed{seed}"


def find_reusable(
    condition: str, seed: int, args: argparse.Namespace
) -> tuple[Path | None, str | None]:
    own_dir = args.project / run_name(condition, seed)
    own_best = own_dir / "weights" / "best.pt"
    if own_best.is_file():
        mismatches = protocol_mismatches(
            own_dir / "args.yaml", args, seed, f"train_{condition}.txt"
        )
        if mismatches:
            raise RuntimeError(
                f"Incompatible existing run: {own_dir}\n  - " + "\n  - ".join(mismatches)
            )
        return own_best, "existing ablation run"

    if condition == "with_mpcd" and args.reuse_existing_with:
        legacy = existing_with_run(seed, args)
        best = legacy / "weights" / "best.pt"
        if best.is_file():
            mismatches = protocol_mismatches(
                legacy / "args.yaml", args, seed, "train/images"
            )
            if not mismatches:
                return best, "verified existing full-training-set run"
            print(f"[not reused] {legacy}")
            for mismatch in mismatches:
                print(f"  - {mismatch}")
    return None, None


def train_or_resume(
    condition: str,
    seed: int,
    data_yaml: Path,
    args: argparse.Namespace,
) -> tuple[Path, str]:
    reusable, source = find_reusable(condition, seed, args)
    if reusable:
        print(f"[reuse] {CONDITION_LABELS[condition]}, seed={seed}: {reusable}")
        return reusable, str(source)

    destination = args.project / run_name(condition, seed)
    last = destination / "weights" / "last.pt"
    if args.dry_run:
        action = "resume" if last.is_file() and args.resume_interrupted else "train"
        print(f"[{action}] {CONDITION_LABELS[condition]}, seed={seed} -> {destination}")
        return destination / "weights" / "best.pt", action

    if last.is_file() and args.resume_interrupted:
        print(f"[resume] {CONDITION_LABELS[condition]}, seed={seed}: {last}")
        model = load_model("yolo", str(last))
        model.train(resume=True)
        source = "resumed interrupted ablation run"
    else:
        if destination.exists():
            raise RuntimeError(
                f"Incomplete run directory exists: {destination}\n"
                "Move it aside, or retain last.pt and use --resume-interrupted."
            )
        print(f"[train] {CONDITION_LABELS[condition]}, seed={seed}")
        model = load_model("yolo", args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            project=str(args.project),
            name=run_name(condition, seed),
            exist_ok=False,
            pretrained=True,
            optimizer="AdamW",
            cos_lr=True,
            amp=args.amp,
            plots=True,
            val=True,
            deterministic=True,
            seed=seed,
            lr0=args.lr0,
            lrf=0.01,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=5.0,
            translate=0.08,
            scale=0.35,
            fliplr=0.5,
            mosaic=0.0,
            mixup=0.0,
            close_mosaic=0,
        )
        source = "new ablation training run"

    best = destination / "weights" / "best.pt"
    trainer_best = Path(getattr(model.trainer, "best", best))
    if trainer_best.is_file():
        best = trainer_best
    if not best.is_file():
        raise FileNotFoundError(f"Best weight not found after training: {best}")
    return best, str(source)


def evaluate(
    condition: str,
    seed: int,
    weight: Path,
    source: str,
    data_yaml: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = load_model("yolo", str(weight))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        plots=False,
    )
    serialized = serialize_metrics(metrics, model, "test")
    target = next(
        (
            item
            for item in serialized["per_class"]
            if int(item["class_id"]) == args.target_class_id
        ),
        None,
    )
    if target is None:
        raise RuntimeError(
            f"Class {args.target_class_id} is absent from test metrics. "
            "The fixed test split cannot evaluate the target-class ablation."
        )
    return {
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "seed": seed,
        "source": source,
        "best_weight": str(weight.resolve()),
        "target_class": target,
        "test": serialized,
    }


def sample_stats(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        selected = [report for report in reports if report["condition"] == condition]
        if not selected:
            continue
        row: dict[str, Any] = {
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "n": len(selected),
            "seeds": [report["seed"] for report in selected],
        }
        for label, key in OVERALL_KEYS.items():
            values = [float(report["test"]["overall"][key]) for report in selected]
            mean, std = sample_stats(values)
            row[f"{label}_mean"] = mean
            row[f"{label}_std"] = std
        for label, key in TARGET_KEYS.items():
            values = [float(report["target_class"][key]) for report in selected]
            mean, std = sample_stats(values)
            row[f"{label}_mean"] = mean
            row[f"{label}_std"] = std
        output.append(row)
    return output


def comparison(summary: list[dict[str, Any]]) -> dict[str, Any] | None:
    indexed = {row["condition"]: row for row in summary}
    if not all(condition in indexed for condition in CONDITIONS):
        return None
    output: dict[str, Any] = {
        "definition": "with_mpcd minus without_mpcd",
        "paired_seeds": sorted(
            set(indexed["with_mpcd"]["seeds"])
            & set(indexed["without_mpcd"]["seeds"])
        ),
    }
    for label in (*OVERALL_KEYS, *TARGET_KEYS):
        output[f"{label}_absolute_gain"] = (
            indexed["with_mpcd"][f"{label}_mean"]
            - indexed["without_mpcd"][f"{label}_mean"]
        )
        output[f"{label}_percentage_point_gain"] = (
            100.0 * output[f"{label}_absolute_gain"]
        )
    return output


def paired_seed_deltas(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return with-minus-without changes for each matched random seed."""
    indexed = {
        (str(report["condition"]), int(report["seed"])): report for report in reports
    }
    rows: list[dict[str, Any]] = []
    seeds = sorted(
        seed
        for condition, seed in indexed
        if condition == "with_mpcd" and ("without_mpcd", seed) in indexed
    )
    for seed in seeds:
        with_row = indexed[("with_mpcd", seed)]
        without_row = indexed[("without_mpcd", seed)]
        row: dict[str, Any] = {"seed": seed}
        for label, key in OVERALL_KEYS.items():
            row[f"{label}_gain"] = float(with_row["test"]["overall"][key]) - float(
                without_row["test"]["overall"][key]
            )
        for label, key in TARGET_KEYS.items():
            row[f"{label}_gain"] = float(with_row["target_class"][key]) - float(
                without_row["target_class"][key]
            )
        rows.append(row)
    return rows


def write_csv(reports: list[dict[str, Any]], summary: list[dict[str, Any]], project: Path) -> None:
    run_fields = [
        "condition",
        "seed",
        *OVERALL_KEYS,
        *TARGET_KEYS,
        "best_weight",
    ]
    with (project / "mpcd_749_ablation_runs.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "condition": report["condition"],
                    "seed": report["seed"],
                    **{
                        label: report["test"]["overall"][key]
                        for label, key in OVERALL_KEYS.items()
                    },
                    **{
                        label: report["target_class"][key]
                        for label, key in TARGET_KEYS.items()
                    },
                    "best_weight": report["best_weight"],
                }
            )

    if summary:
        with (project / "mpcd_749_ablation_summary.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)

    paired = paired_seed_deltas(reports)
    if paired:
        with (project / "mpcd_749_ablation_paired_deltas.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)


def write_latex(summary: list[dict[str, Any]], project: Path) -> None:
    lines = [
        "% Mean $\\pm$ sample standard deviation over three random seeds.",
        "% Setting & Broken-conductor R & Broken-conductor AP50 & "
        "Broken-conductor AP50:95 & Overall AP50 & Overall AP50:95 \\\\",
    ]
    metrics = (
        "broken_recall",
        "broken_mAP50",
        "broken_mAP50-95",
        "overall_mAP50",
        "overall_mAP50-95",
    )
    for row in summary:
        cells = [row["condition_label"]]
        for metric in metrics:
            cells.append(
                f"{100.0 * row[f'{metric}_mean']:.2f} $\\pm$ "
                f"{100.0 * row[f'{metric}_std']:.2f}"
            )
        lines.append(" & ".join(cells) + r" \\")
    (project / "mpcd_749_ablation_latex_rows.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def save_outputs(
    args: argparse.Namespace,
    audit: dict[str, Any],
    reports: list[dict[str, Any]],
) -> None:
    summary = aggregate(reports)
    delta = comparison(summary)
    payload = {
        "experiment": "749-image MPCD broken-conductor training-data ablation",
        "protocol": {
            "model": "YOLOv8-L",
            "initial_weight": args.model,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "optimizer": "AdamW",
            "lr0": args.lr0,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "amp": args.amp,
            "deterministic": True,
            "augmentation": {
                "hsv_h": 0.015,
                "hsv_s": 0.5,
                "hsv_v": 0.3,
                "degrees": 5.0,
                "translate": 0.08,
                "scale": 0.35,
                "fliplr": 0.5,
                "mosaic": 0.0,
                "mixup": 0.0,
            },
            "test_conf": args.conf,
            "test_iou": args.iou,
            "standard_deviation": "sample (n-1)",
        },
        "data_audit": {key: value for key, value in audit.items() if key != "external_samples"},
        "runs": reports,
        "summary": summary,
        "paired_seed_deltas": paired_seed_deltas(reports),
        "with_minus_without": delta,
    }
    (args.project / "mpcd_749_ablation_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(reports, summary, args.project)
    write_latex(summary, args.project)


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != 3:
        raise ValueError(f"--seeds must contain three distinct values: {args.seeds}")
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError(f"Duplicate --conditions values: {args.conditions}")

    args.project = args.project.resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    config_dir = ROOT / "runs" / ".ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    _, condition_yamls, audit = prepare_conditions(args)
    print("MPCD ablation audit:")
    print(f"  current external images: {audit['current_external_images']}")
    print(f"  external broken-conductor boxes: {audit['current_external_boxes']}")
    print(f"  with MPCD training images: {audit['with_mpcd_train_images']}")
    print(f"  without MPCD training images: {audit['without_mpcd_train_images']}")
    print(f"  fixed seeds: {args.seeds}")

    reports: list[dict[str, Any]] = []
    existing_report = args.project / "mpcd_749_ablation_metrics.json"
    if existing_report.is_file():
        try:
            saved = json.loads(existing_report.read_text(encoding="utf-8"))
            reports = [
                row
                for row in saved.get("runs", [])
                if row.get("condition") in CONDITIONS
                and int(row.get("seed", -1)) in args.seeds
            ]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            reports = []

    for condition in args.conditions:
        for seed in args.seeds:
            already = next(
                (
                    row
                    for row in reports
                    if row["condition"] == condition and int(row["seed"]) == seed
                ),
                None,
            )
            if already is not None and Path(already["best_weight"]).is_file():
                print(f"[report reuse] {CONDITION_LABELS[condition]}, seed={seed}")
                continue

            weight, source = train_or_resume(
                condition, seed, condition_yamls[condition], args
            )
            if args.dry_run:
                continue
            report = evaluate(
                condition,
                seed,
                weight,
                source,
                condition_yamls[condition],
                args,
            )
            reports = [
                row
                for row in reports
                if not (row["condition"] == condition and int(row["seed"]) == seed)
            ]
            reports.append(report)
            reports.sort(key=lambda row: (CONDITIONS.index(row["condition"]), row["seed"]))
            save_outputs(args, audit, reports)

    if args.dry_run:
        print("\nDry run complete: no model was loaded or trained.")
        print(f"Split audit: {args.project / '_configs' / 'mpcd_749_split_audit.json'}")
        return

    save_outputs(args, audit, reports)
    print("\nAblation complete.")
    print(f"JSON: {args.project / 'mpcd_749_ablation_metrics.json'}")
    print(f"CSV: {args.project / 'mpcd_749_ablation_summary.csv'}")
    print(f"LaTeX: {args.project / 'mpcd_749_ablation_latex_rows.txt'}")


if __name__ == "__main__":
    main()
