"""Train and evaluate fair object-detection baselines on one fixed test split.

The script intentionally uses ``train/images`` once per image for baseline
training.  The proposed model may continue to use ``data.yaml`` with
``train_quality_balanced.txt``.  Validation and test paths are copied from the
same source YAML, so every method is evaluated on exactly the same images.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "rtdetr": {"family": "rtdetr", "weight": "rtdetr-l.pt", "name": "baseline_rtdetr_l"},
    "yolov8": {"family": "yolo", "weight": "yolov8l.pt", "name": "baseline_yolov8_l"},
    "yolo26": {"family": "yolo", "weight": "yolo26l.pt", "name": "baseline_yolo26_l"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train RT-DETR-L, YOLOv8-L and YOLO26-L with one fixed protocol, "
            "then evaluate them on the same validation and test splits."
        )
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENTS),
        default=["rtdetr", "yolov8", "yolo26"],
        help="Baselines to run sequentially.",
    )
    parser.add_argument(
        "--baseline-train",
        default="train/images",
        help="Unbalanced baseline training source, relative to the dataset root.",
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
    parser.add_argument("--seed", type=int, default=42)
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
        help="Use automatic mixed precision. Disable with --no-amp if NaN occurs.",
    )
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "baselines")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--merge-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When running only YOLO26-L, retain compatible RT-DETR-L/YOLOv8-L "
            "rows already stored in baseline_metrics.json."
        ),
    )
    return parser.parse_args()


def resolve_dataset_path(value: str, dataset_root: Path) -> str:
    """Return an absolute local path while leaving URLs untouched."""
    if "://" in value:
        return value
    path = Path(value)
    if not path.is_absolute():
        path = dataset_root / path
    return str(path.resolve())


def make_baseline_yaml(source_yaml: Path, train_source: str, output: Path) -> dict[str, Any]:
    source_yaml = source_yaml.resolve()
    source = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    declared_root = Path(source.get("path", source_yaml.parent))
    if not declared_root.is_absolute():
        declared_root = source_yaml.parent / declared_root
    dataset_root = declared_root.resolve()

    config: dict[str, Any] = {
        "path": str(dataset_root),
        "train": resolve_dataset_path(train_source, dataset_root),
        "val": resolve_dataset_path(str(source["val"]), dataset_root),
        "test": resolve_dataset_path(str(source["test"]), dataset_root),
        "nc": int(source["nc"]),
        "names": source["names"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return config


def load_model(family: str, weight: str):
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as exc:
        raise SystemExit("Please install dependencies: pip install -U ultralytics pyyaml") from exc
    try:
        return RTDETR(weight) if family == "rtdetr" else YOLO(weight)
    except Exception as exc:
        if Path(weight).name.lower().startswith("yolo26"):
            raise RuntimeError(
                "Unable to load YOLO26-L. Run `pip install -U ultralytics` "
                "and verify that yolo26l.pt is a complete checkpoint."
            ) from exc
        raise


def to_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    return [float(item) for item in value]


def to_int_list(value: Any) -> list[int]:
    return [int(item) for item in to_float_list(value)]


def serialize_metrics(metrics: Any, model: Any, split: str) -> dict[str, Any]:
    overall = {str(key): float(value) for key, value in metrics.results_dict.items()}
    names = metrics.names
    if isinstance(names, list):
        names = dict(enumerate(names))

    precision = to_float_list(getattr(metrics.box, "p", None))
    recall = to_float_list(getattr(metrics.box, "r", None))
    ap50 = to_float_list(getattr(metrics.box, "ap50", None))
    ap5095 = to_float_list(getattr(metrics.box, "maps", None))
    class_ids = to_int_list(getattr(metrics.box, "ap_class_index", None))
    count = min(len(precision), len(recall), len(ap50), len(ap5095))
    if len(class_ids) != count:
        class_ids = list(range(count))
    per_class = []
    for index in range(count):
        class_id = class_ids[index]
        per_class.append(
            {
                "class_id": class_id,
                "class_name": str(names.get(class_id, class_id)),
                "precision": precision[index],
                "recall": recall[index],
                "mAP50": ap50[index],
                "mAP50-95": ap5095[index],
            }
        )

    speed = {str(key): float(value) for key, value in getattr(metrics, "speed", {}).items()}
    inference_ms = speed.get("inference", 0.0)
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    return {
        "split": split,
        "overall": overall,
        "per_class": per_class,
        "speed_ms_per_image": speed,
        "inference_fps": 1000.0 / inference_ms if inference_ms > 0 else None,
        "parameters": int(parameters),
    }


def evaluate(model: Any, args: argparse.Namespace, data_yaml: Path, split: str) -> dict[str, Any]:
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        plots=True,
    )
    return serialize_metrics(metrics, model, split)


def train_one(key: str, args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    spec = dict(EXPERIMENTS[key])
    if key == "rtdetr":
        spec["weight"] = args.rtdetr_weight
    elif key == "yolov8":
        spec["weight"] = args.yolo_weight
    elif key == "yolo26":
        spec["weight"] = args.yolo26_weight

    run_dir = args.project.resolve() / spec["name"]
    best_weight = run_dir / "weights" / "best.pt"
    if best_weight.is_file() and not args.overwrite:
        print(f"[reuse] Existing best weight: {best_weight}")
    else:
        model = load_model(spec["family"], spec["weight"])
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            project=str(args.project.resolve()),
            name=spec["name"],
            exist_ok=args.overwrite,
            pretrained=True,
            optimizer="AdamW",
            cos_lr=True,
            amp=args.amp,
            plots=True,
            val=True,
            deterministic=True,
            seed=args.seed,
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
        trainer_best = Path(model.trainer.best)
        if trainer_best.is_file():
            best_weight = trainer_best

    if not best_weight.is_file():
        raise FileNotFoundError(f"Best weight not found: {best_weight}")

    best_model = load_model(spec["family"], str(best_weight))
    return {
        "experiment": key,
        "family": spec["family"],
        "initial_weight": spec["weight"],
        "best_weight": str(best_weight.resolve()),
        "validation": evaluate(best_model, args, data_yaml, "val"),
        "test": evaluate(best_model, args, data_yaml, "test"),
    }


def write_summary_csv(reports: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "experiment",
        "precision",
        "recall",
        "mAP50",
        "mAP50-95",
        "parameters",
        "inference_fps",
        "best_weight",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            test = report["test"]
            overall = test["overall"]
            writer.writerow(
                {
                    "experiment": report["experiment"],
                    "precision": overall.get("metrics/precision(B)"),
                    "recall": overall.get("metrics/recall(B)"),
                    "mAP50": overall.get("metrics/mAP50(B)"),
                    "mAP50-95": overall.get("metrics/mAP50-95(B)"),
                    "parameters": test["parameters"],
                    "inference_fps": test["inference_fps"],
                    "best_weight": report["best_weight"],
                }
            )


def write_per_class_csv(reports: list[dict[str, Any]], output: Path) -> None:
    """Write class-wise held-out test metrics for paper/error analysis."""
    fields = [
        "experiment",
        "class_id",
        "class_name",
        "precision",
        "recall",
        "mAP50",
        "mAP50-95",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for row in report["test"]["per_class"]:
                writer.writerow({"experiment": report["experiment"], **row})


def merge_compatible_existing_reports(
    report_path: Path,
    new_reports: list[dict[str, Any]],
    protocol: dict[str, Any],
    enabled: bool,
) -> list[dict[str, Any]]:
    """Preserve earlier comparison rows only when the core protocol matches."""
    if not enabled or not report_path.is_file():
        return new_reports
    try:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warning] Existing report was not merged: {exc}")
        return new_reports

    previous_protocol = previous.get("protocol", {})
    compatibility_keys = ("epochs", "imgsz", "batch", "seed")
    mismatches = [
        key
        for key in compatibility_keys
        if previous_protocol.get(key) != protocol.get(key)
    ]
    if mismatches:
        print(
            "[warning] Existing reports were not merged because these protocol "
            f"fields differ: {', '.join(mismatches)}"
        )
        return new_reports

    merged = {
        str(report["experiment"]): report
        for report in previous.get("experiments", [])
        if isinstance(report, dict) and "experiment" in report
    }
    merged.update({str(report["experiment"]): report for report in new_reports})
    preferred_order = list(EXPERIMENTS)
    return [merged[key] for key in preferred_order if key in merged] + [
        report for key, report in merged.items() if key not in preferred_order
    ]


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    args.project.mkdir(parents=True, exist_ok=True)
    config_dir = ROOT / "runs" / ".ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    baseline_yaml = args.project / "_configs" / "baseline_data.yaml"
    resolved_data = make_baseline_yaml(args.data, args.baseline_train, baseline_yaml)
    train_path = Path(resolved_data["train"])
    if not train_path.exists():
        raise FileNotFoundError(f"Baseline training source not found: {train_path}")

    print(f"Fixed baseline YAML: {baseline_yaml}")
    print(f"Training source: {resolved_data['train']}")
    print(f"Validation source: {resolved_data['val']}")
    print(f"Test source: {resolved_data['test']}")

    report_path = args.project / "baseline_metrics.json"
    protocol = {
        "data_yaml": str(baseline_yaml.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "amp": args.amp,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
    }
    new_reports = [train_one(key, args, baseline_yaml.resolve()) for key in args.experiments]
    reports = merge_compatible_existing_reports(
        report_path, new_reports, protocol, args.merge_existing
    )
    report_path.write_text(
        json.dumps(
            {
                "protocol": protocol,
                "experiments": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = args.project / "baseline_summary.csv"
    write_summary_csv(reports, csv_path)
    per_class_csv_path = args.project / "baseline_per_class_test.csv"
    write_per_class_csv(reports, per_class_csv_path)
    print(f"\nJSON report: {report_path}")
    print(f"CSV summary: {csv_path}")
    print(f"Per-class test CSV: {per_class_csv_path}")


if __name__ == "__main__":
    main()
