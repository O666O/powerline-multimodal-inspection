"""Run three-seed detector experiments and create paper-ready summaries.

The script compares RT-DETR-L, YOLOv8-L and YOLO26-L under one fixed
training/evaluation protocol.  It can reuse the already completed seed-42
baseline after checking its ``args.yaml`` and only trains the two missing
seeds by default.
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

from run_detection_baselines import (
    EXPERIMENTS,
    load_model,
    make_baseline_yaml,
    serialize_metrics,
)


ROOT = Path(__file__).resolve().parent
DISPLAY_NAMES = {
    "rtdetr": "RT-DETR-L",
    "yolov8": "YOLOv8-L",
    "yolo26": "YOLO26-L",
}
METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate object detectors with exactly three seeds."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENTS),
        default=["rtdetr", "yolov8", "yolo26"],
    )
    parser.add_argument(
        "--seeds",
        nargs=3,
        type=int,
        default=[42, 3407, 2026],
        metavar=("SEED1", "SEED2", "SEED3"),
        help="Exactly three distinct seeds.",
    )
    parser.add_argument(
        "--baseline-train",
        default="train/images",
        help="Fixed training source relative to the dataset root.",
    )
    parser.add_argument("--rtdetr-weight", default="rtdetr-l.pt")
    parser.add_argument("--yolo-weight", default="yolov8l.pt")
    parser.add_argument("--yolo26-weight", default="yolo26l.pt")
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
        help="Use AMP; use --no-amp if RT-DETR produces NaN.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=ROOT / "runs" / "three_seed_detection",
    )
    parser.add_argument(
        "--existing-baseline-project",
        type=Path,
        default=ROOT / "runs" / "baselines",
        help="Location of the completed seed-42 baseline runs.",
    )
    parser.add_argument(
        "--reuse-existing-seed42",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse compatible legacy seed-42 weights after protocol checks.",
    )
    parser.add_argument(
        "--resume-interrupted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume a per-seed run when last.pt exists but best.pt does not.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the run plan without loading a model.",
    )
    return parser.parse_args()


def experiment_spec(key: str, args: argparse.Namespace) -> dict[str, str]:
    spec = dict(EXPERIMENTS[key])
    if key == "rtdetr":
        spec["weight"] = args.rtdetr_weight
    elif key == "yolov8":
        spec["weight"] = args.yolo_weight
    else:
        spec["weight"] = args.yolo26_weight
    return spec


def same_number(actual: Any, expected: float, tolerance: float = 1e-12) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def protocol_mismatches(
    args_yaml: Path, args: argparse.Namespace, seed: int
) -> list[str]:
    """Return reasons why a completed run is incompatible with this protocol."""
    if not args_yaml.is_file():
        return [f"missing {args_yaml.name}"]
    try:
        saved = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot read args.yaml: {exc}"]

    exact = {
        "seed": seed,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
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
    return mismatches


def run_name(key: str, seed: int) -> str:
    return f"{EXPERIMENTS[key]['name']}_seed{seed}"


def compatible_completed_run(
    run_dir: Path, args: argparse.Namespace, seed: int
) -> tuple[Path | None, list[str]]:
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        return None, ["best.pt does not exist"]
    mismatches = protocol_mismatches(run_dir / "args.yaml", args, seed)
    return (best if not mismatches else None), mismatches


def find_reusable_weight(
    key: str, seed: int, args: argparse.Namespace
) -> tuple[Path | None, str | None]:
    """Prefer a matching seed-specific run, then the checked legacy seed-42 run."""
    specific_dir = args.project.resolve() / run_name(key, seed)
    best, mismatches = compatible_completed_run(specific_dir, args, seed)
    if best:
        return best, "existing seed-specific run"
    if (specific_dir / "weights" / "best.pt").is_file() and mismatches:
        raise RuntimeError(
            f"Existing run is incompatible: {specific_dir}\n  - "
            + "\n  - ".join(mismatches)
            + "\nMove that directory aside or use a different --project."
        )

    if seed == 42 and args.reuse_existing_seed42:
        legacy_dir = args.existing_baseline_project.resolve() / EXPERIMENTS[key]["name"]
        best, mismatches = compatible_completed_run(legacy_dir, args, seed)
        if best:
            return best, "verified legacy seed-42 baseline"
        print(f"[not reused] {legacy_dir}")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
    return None, None


def train_or_resume(
    key: str,
    seed: int,
    args: argparse.Namespace,
    fixed_yaml: Path,
) -> tuple[Path, str]:
    reusable, source = find_reusable_weight(key, seed, args)
    if reusable:
        print(f"[reuse] {DISPLAY_NAMES[key]} seed={seed}: {reusable}")
        return reusable, str(source)

    spec = experiment_spec(key, args)
    destination = args.project.resolve() / run_name(key, seed)
    last_weight = destination / "weights" / "last.pt"

    if args.dry_run:
        action = "resume" if last_weight.is_file() and args.resume_interrupted else "train"
        print(
            f"[{action}] {DISPLAY_NAMES[key]} seed={seed} -> {destination} "
            f"from {last_weight if action == 'resume' else spec['weight']}"
        )
        return destination / "weights" / "best.pt", action

    if last_weight.is_file() and args.resume_interrupted:
        print(f"[resume] {DISPLAY_NAMES[key]} seed={seed}: {last_weight}")
        model = load_model(spec["family"], str(last_weight))
        model.train(resume=True)
        action = "resumed interrupted run"
    else:
        if destination.exists():
            raise RuntimeError(
                f"Incomplete run directory already exists: {destination}\n"
                "Use --resume-interrupted when it contains weights/last.pt, or move "
                "the incomplete directory aside before starting again."
            )
        print(f"[train] {DISPLAY_NAMES[key]} seed={seed} from {spec['weight']}")
        model = load_model(spec["family"], spec["weight"])
        model.train(
            data=str(fixed_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            project=str(args.project.resolve()),
            name=run_name(key, seed),
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
        action = "new training run"

    best = destination / "weights" / "best.pt"
    trainer_best = Path(getattr(model.trainer, "best", best))
    if trainer_best.is_file():
        best = trainer_best
    if not best.is_file():
        raise FileNotFoundError(f"Best weight not found after training: {best}")
    return best, action


def evaluate_test(
    key: str,
    seed: int,
    best_weight: Path,
    source: str,
    args: argparse.Namespace,
    fixed_yaml: Path,
) -> dict[str, Any]:
    spec = experiment_spec(key, args)
    model = load_model(spec["family"], str(best_weight))
    metrics = model.val(
        data=str(fixed_yaml),
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
    return {
        "experiment": key,
        "display_name": DISPLAY_NAMES[key],
        "seed": seed,
        "source": source,
        "best_weight": str(best_weight.resolve()),
        "test": serialize_metrics(metrics, model, "test"),
    }


def sample_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_runs(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in EXPERIMENTS:
        selected = [report for report in reports if report["experiment"] == key]
        if not selected:
            continue
        row: dict[str, Any] = {
            "experiment": key,
            "display_name": DISPLAY_NAMES[key],
            "n": len(selected),
            "seeds": [int(report["seed"]) for report in selected],
        }
        for label, metric_key in METRICS.items():
            values = [float(report["test"]["overall"][metric_key]) for report in selected]
            mean, std = sample_stats(values)
            row[f"{label}_mean"] = mean
            row[f"{label}_std"] = std
            row[f"{label}_mean_percent"] = 100.0 * mean
            row[f"{label}_std_percent"] = 100.0 * std
        rows.append(row)
    return rows


def aggregate_per_class(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for report in reports:
        for item in report["test"]["per_class"]:
            key = (
                str(report["experiment"]),
                int(item["class_id"]),
                str(item["class_name"]),
            )
            grouped.setdefault(key, []).append(item)

    rows = []
    for (experiment, class_id, class_name), items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "experiment": experiment,
            "display_name": DISPLAY_NAMES[experiment],
            "class_id": class_id,
            "class_name": class_name,
            "n": len(items),
        }
        for metric in ("precision", "recall", "mAP50", "mAP50-95"):
            mean, std = sample_stats([float(item[metric]) for item in items])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    return rows


def write_run_csv(reports: list[dict[str, Any]], output: Path) -> None:
    fields = ["experiment", "display_name", "seed", *METRICS, "best_weight"]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            overall = report["test"]["overall"]
            writer.writerow(
                {
                    "experiment": report["experiment"],
                    "display_name": report["display_name"],
                    "seed": report["seed"],
                    **{label: overall[key] for label, key in METRICS.items()},
                    "best_weight": report["best_weight"],
                }
            )


def write_dict_csv(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_rows(summary: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "% Three independent runs; values are mean $\\pm$ sample standard deviation.",
        "% Model & Precision (\\%) & Recall (\\%) & mAP@50 (\\%) & "
        "mAP@50:95 (\\%) \\\\",
    ]
    for row in summary:
        cells = [row["display_name"]]
        for metric in ("precision", "recall", "mAP50", "mAP50-95"):
            cells.append(
                f"{row[f'{metric}_mean_percent']:.2f} $\\pm$ "
                f"{row[f'{metric}_std_percent']:.2f}"
            )
        lines.append(" & ".join(cells) + r" \\")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    args: argparse.Namespace,
    fixed_yaml: Path,
    reports: list[dict[str, Any]],
) -> None:
    summary = aggregate_runs(reports)
    per_class = aggregate_per_class(reports)
    protocol = {
        "data_yaml": str(fixed_yaml.resolve()),
        "training_source": args.baseline_train,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "amp": args.amp,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "test_conf": args.conf,
        "test_iou": args.iou,
        "standard_deviation": "sample (n-1)",
    }
    payload = {"protocol": protocol, "runs": reports, "summary": summary}
    report_path = args.project / "three_seed_metrics.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_run_csv(reports, args.project / "three_seed_runs.csv")
    write_dict_csv(summary, args.project / "three_seed_summary.csv")
    write_dict_csv(per_class, args.project / "three_seed_per_class_summary.csv")
    write_latex_rows(summary, args.project / "three_seed_latex_rows.txt")

    print("\nThree-seed summary (held-out test split):")
    for row in summary:
        values = ", ".join(
            f"{metric}={row[f'{metric}_mean_percent']:.2f}+-"
            f"{row[f'{metric}_std_percent']:.2f}%"
            for metric in ("precision", "recall", "mAP50", "mAP50-95")
        )
        print(f"  {row['display_name']} (n={row['n']}): {values}")
    print(f"\nJSON: {report_path}")
    print(f"Paper table: {args.project / 'three_seed_latex_rows.txt'}")


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != 3:
        raise ValueError(f"--seeds must contain three distinct integers: {args.seeds}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    args.project = args.project.resolve()
    args.existing_baseline_project = args.existing_baseline_project.resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    config_dir = ROOT / "runs" / ".ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    fixed_yaml = args.project / "_configs" / "three_seed_data.yaml"
    resolved = make_baseline_yaml(args.data, args.baseline_train, fixed_yaml)
    for split in ("train", "val", "test"):
        value = str(resolved[split])
        if "://" not in value and not Path(value).exists():
            raise FileNotFoundError(f"{split} source not found: {value}")

    print(f"Fixed data YAML: {fixed_yaml}")
    print(f"Seeds: {args.seeds}")
    print(f"Models: {', '.join(DISPLAY_NAMES[key] for key in args.experiments)}")
    print(f"Training source: {resolved['train']}")
    print(f"Test source: {resolved['test']}")

    reports: list[dict[str, Any]] = []
    for key in args.experiments:
        for seed in args.seeds:
            best_weight, source = train_or_resume(key, seed, args, fixed_yaml.resolve())
            if args.dry_run:
                continue
            report = evaluate_test(
                key, seed, best_weight, source, args, fixed_yaml.resolve()
            )
            reports.append(report)
            save_outputs(args, fixed_yaml, reports)

    if args.dry_run:
        print("\nDry run complete; no model was loaded or trained.")
        return
    save_outputs(args, fixed_yaml, reports)


if __name__ == "__main__":
    main()
