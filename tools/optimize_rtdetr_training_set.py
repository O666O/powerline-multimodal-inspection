"""Audit and improve the RT-DETR training set without changing val/test."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--target-images", type=int, default=1800)
    parser.add_argument("--max-repeat", type=int, default=3)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def inspect_pair(pair):
    image_path, label_path, num_classes = pair
    result = {
        "image": image_path,
        "label": label_path,
        "classes": set(),
        "clean_lines": [],
        "duplicate_boxes": 0,
        "invalid": [],
        "image_error": None,
    }
    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as error:  # Pillow exposes several decoder-specific errors.
        result["image_error"] = repr(error)

    seen = set()
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except Exception as error:
        result["invalid"].append(f"label_read_error:{error!r}")
        return result

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            class_id = int(parts[0])
            values = tuple(float(value) for value in parts[1:])
        except (ValueError, IndexError) as error:
            result["invalid"].append(f"line_{line_number}:parse:{error!r}")
            continue
        if len(values) != 4:
            result["invalid"].append(f"line_{line_number}:not_box")
            continue
        cx, cy, width, height = values
        if not (
            0 <= class_id < num_classes
            and all(math.isfinite(value) for value in values)
            and 0 <= cx <= 1
            and 0 <= cy <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            result["invalid"].append(f"line_{line_number}:range")
            continue
        canonical = (
            class_id,
            round(cx, 8),
            round(cy, 8),
            round(width, 8),
            round(height, 8),
        )
        if canonical in seen:
            result["duplicate_boxes"] += 1
            continue
        seen.add(canonical)
        result["classes"].add(class_id)
        result["clean_lines"].append(
            f"{class_id} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}"
        )
    return result


def move_to_quarantine(root: Path, image_path: Path, label_path: Path):
    quarantine = root / "quarantine" / "rtdetr_training_quality"
    image_target = quarantine / "images" / image_path.name
    label_target = quarantine / "labels" / label_path.name
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    if image_target.exists() or label_target.exists():
        raise FileExistsError(f"Quarantine collision for {image_path.name}")
    shutil.move(str(image_path), str(image_target))
    shutil.move(str(label_path), str(label_target))


def main():
    args = parse_args()
    root = args.root.resolve()
    data_path = root / "data.yaml"
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    class_names = config["names"]
    num_classes = len(class_names)

    image_dir = root / "train" / "images"
    label_dir = root / "train" / "labels"
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    label_by_stem = {path.stem: path for path in label_dir.glob("*.txt")}
    missing_labels = [path for path in image_paths if path.stem not in label_by_stem]
    image_by_stem = {path.stem: path for path in image_paths}
    missing_images = [
        path for stem, path in label_by_stem.items() if stem not in image_by_stem
    ]
    if missing_labels or missing_images:
        raise RuntimeError(
            f"Unpaired files: {len(missing_labels)} images, "
            f"{len(missing_images)} labels"
        )

    pairs = [
        (image_path, label_by_stem[image_path.stem], num_classes)
        for image_path in image_paths
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspected = list(executor.map(inspect_pair, pairs))

    corrupt = [
        item for item in inspected if item["image_error"] or item["invalid"]
    ]
    duplicate_boxes = sum(item["duplicate_boxes"] for item in inspected)
    files_with_duplicate_boxes = sum(
        bool(item["duplicate_boxes"]) for item in inspected
    )

    usable = [item for item in inspected if item not in corrupt]
    class_images = Counter()
    class_instances = Counter()
    for item in usable:
        for class_id in item["classes"]:
            class_images[class_id] += 1
        for line in item["clean_lines"]:
            class_instances[int(line.split()[0])] += 1

    repeat_by_class = {}
    for class_id in range(num_classes):
        count = class_images[class_id]
        if count == 0:
            repeat = args.max_repeat
        else:
            repeat = math.ceil(args.target_images / count)
        repeat_by_class[class_id] = max(1, min(args.max_repeat, repeat))

    balanced_lines = []
    repeat_histogram = Counter()
    effective_class_images = Counter()
    for item in usable:
        if not item["classes"]:
            repeat = 1
        else:
            repeat = max(repeat_by_class[class_id] for class_id in item["classes"])
        repeat_histogram[repeat] += 1
        for class_id in item["classes"]:
            effective_class_images[class_id] += repeat
        relative = item["image"].relative_to(root).as_posix()
        balanced_lines.extend([f"./{relative}"] * repeat)

    report = {
        "source_images": len(image_paths),
        "source_instances": sum(class_instances.values()) + duplicate_boxes,
        "usable_images": len(usable),
        "corrupt_or_invalid_pairs": len(corrupt),
        "duplicate_boxes_removed": duplicate_boxes,
        "files_with_duplicate_boxes": files_with_duplicate_boxes,
        "target_effective_images_per_class": args.target_images,
        "max_repeat": args.max_repeat,
        "balanced_epoch_images": len(balanced_lines),
        "repeat_histogram": {
            str(repeat): count for repeat, count in sorted(repeat_histogram.items())
        },
        "classes": {
            str(class_id): {
                "name": class_names[class_id],
                "source_images": class_images[class_id],
                "source_instances": class_instances[class_id],
                "repeat_factor": repeat_by_class[class_id],
                "effective_epoch_images": effective_class_images[class_id],
            }
            for class_id in range(num_classes)
        },
        "invalid_examples": [
            {
                "image": str(item["image"].relative_to(root)),
                "label": str(item["label"].relative_to(root)),
                "image_error": item["image_error"],
                "invalid": item["invalid"],
            }
            for item in corrupt[:100]
        ],
    }

    report_path = root / "rtdetr_training_quality_report.json"
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("Dry run complete. Re-run with --apply to write cleaned labels/list.")
        return

    for item in corrupt:
        move_to_quarantine(root, item["image"], item["label"])
    for item in usable:
        if item["duplicate_boxes"]:
            temporary = item["label"].with_suffix(".txt.quality_tmp")
            temporary.write_text(
                "\n".join(item["clean_lines"]) + "\n", encoding="utf-8"
            )
            os.replace(temporary, item["label"])

    list_path = root / "train_quality_balanced.txt"
    list_path.write_text("\n".join(balanced_lines) + "\n", encoding="utf-8")
    config["train"] = list_path.name
    data_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    report["balanced_list"] = list_path.name
    report["data_yaml_train"] = config["train"]
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
