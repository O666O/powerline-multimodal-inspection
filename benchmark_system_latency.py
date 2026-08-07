"""Benchmark inference latency for the power-line inspection system.

The benchmark excludes model/index loading and warm-up. Images are preloaded so
that the reported neural-network latency is not dominated by disk I/O.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import use_model as app


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(function):
    sync_device()
    start = time.perf_counter()
    output = function()
    sync_device()
    return (time.perf_counter() - start) * 1000.0, output


def summarize(name: str, values: list[float]) -> dict:
    ordered = np.asarray(values, dtype=np.float64)
    mean_ms = float(ordered.mean())
    return {
        "module": name,
        "runs": len(values),
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(float(ordered.std(ddof=1)) if len(values) > 1 else 0.0, 3),
        "median_ms": round(float(np.median(ordered)), 3),
        "p95_ms": round(float(np.percentile(ordered, 95)), 3),
        "fps": round(1000.0 / mean_ms, 3) if mean_ms > 0 else None,
    }


def repeat_benchmark(name, function, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        function()
    sync_device()
    values = []
    for _ in range(repeats):
        elapsed, _ = timed_call(function)
        values.append(elapsed)
    return summarize(name, values)


def find_detection_images(root: Path, limit: int) -> list[np.ndarray]:
    image_dir = root / "test" / "images"
    paths = sorted(
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No test images found under {image_dir}")
    selected = paths[: max(1, min(limit, len(paths)))]
    return [np.asarray(Image.open(path).convert("RGB")) for path in selected]


def write_outputs(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "system_latency.json"
    csv_path = output_dir / "system_latency.csv"
    md_path = output_dir / "system_latency_table.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["module", "runs", "mean_ms", "std_ms", "median_ms", "p95_ms", "fps"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report["results"])

    lines = [
        "| Module | Runs | Mean (ms) | Std (ms) | Median (ms) | P95 (ms) | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['module']} | {row['runs']} | {row['mean_ms']:.3f} | "
            f"{row['std_ms']:.3f} | {row['median_ms']:.3f} | "
            f"{row['p95_ms']:.3f} | {row['fps']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"Table: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Chinese-CLIP and YOLOv8 latency")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--image-pool", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--output", default="runs/system_latency")
    args = parser.parse_args()

    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("--warmup and --repeats must be positive")

    root = Path(app.PROJECT_ROOT)
    print(f"Device: {app.DEVICE}")
    print("Loading Chinese-CLIP and FAISS indexes ...")
    app.load_global_resources(use_finetuned=True)
    print("Loading YOLOv8 detector ...")
    detector = app.load_defect_detector()
    if detector is None:
        raise RuntimeError(f"Detector could not be loaded from {app.DEFECT_MODEL_PATH}")

    query = app.text_list[0] if app.text_list else "输电线路绝缘子"
    first_crop_path = Path(app.img_id2path[app.img_id_list[0]])
    clip_image = Image.open(first_crop_path).convert("RGB")
    detection_images = find_detection_images(root, args.image_pool)

    def encode_text():
        with torch.inference_mode():
            return app.encode_text_features([query])

    text_feature = encode_text().detach().cpu().numpy().astype("float32")

    def faiss_text_to_image():
        return app.img_index.search(text_feature, 5)

    def text_to_image_retrieval():
        with torch.inference_mode():
            feature = app.encode_text_features([query])
        feature_np = feature.detach().cpu().numpy().astype("float32")
        return app.img_index.search(feature_np, 5)

    def encode_image():
        with torch.inference_mode():
            return app.encode_image_features([clip_image])

    image_feature = encode_image().detach().cpu().numpy().astype("float32")

    def faiss_image_to_text():
        return app.text_index.search(image_feature, min(5, len(app.text_list)))

    def image_to_text_retrieval():
        with torch.inference_mode():
            feature = app.encode_image_features([clip_image])
        feature_np = feature.detach().cpu().numpy().astype("float32")
        return app.text_index.search(feature_np, min(5, len(app.text_list)))

    detector_counter = {"value": 0}

    def yolo_detection():
        index = detector_counter["value"] % len(detection_images)
        detector_counter["value"] += 1
        return detector.predict(
            source=detection_images[index],
            imgsz=args.imgsz,
            conf=app.DEFECT_CONF_THRES,
            device=app.DEVICE,
            verbose=False,
        )

    tests = [
        ("Chinese-CLIP text encoding", encode_text),
        ("FAISS image search", faiss_text_to_image),
        ("Text-to-image retrieval", text_to_image_retrieval),
        ("Chinese-CLIP image encoding", encode_image),
        ("FAISS text search", faiss_image_to_text),
        ("Image-to-text retrieval", image_to_text_retrieval),
        (f"YOLOv8 detection ({args.imgsz}px)", yolo_detection),
    ]

    results = []
    for name, function in tests:
        print(f"Benchmarking {name} ...")
        results.append(repeat_benchmark(name, function, args.warmup, args.repeats))

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    report = {
        "measurement_protocol": {
            "batch_size": 1,
            "warmup_runs": args.warmup,
            "measured_runs": args.repeats,
            "disk_io_excluded": True,
            "model_loading_excluded": True,
            "remote_llm_api_excluded": True,
            "synchronization": "torch.cuda.synchronize before and after each timed GPU call",
        },
        "environment": {
            "device": app.DEVICE,
            "gpu": gpu_name,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "detector_weight": str(app.DEFECT_MODEL_PATH),
            "chinese_clip_model": str(app.FINETUNED_MODEL_PATH),
            "detector_imgsz": args.imgsz,
        },
        "results": results,
    }
    write_outputs(root / args.output, report)


if __name__ == "__main__":
    main()
