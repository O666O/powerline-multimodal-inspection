"""Build the fine-grained RT-DETR dataset from audited external sources.

The operation is transactional at the dataset level:
1. external samples are prepared and validated in a staging directory;
2. current labels and data.yaml are backed up;
3. old class IDs are remapped;
4. staged external images/labels are moved into the dataset splits;
5. data.yaml and an import manifest are updated.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from PIL import Image


OLD_NAMES = [
    "Damper - Spiral",
    "Damper - Stockbridge",
    "Glass Insulator",
    "Lightning Rod Suspension",
    "Polymer Insulator",
    "Polymer Insulator Shackle",
    "Vari-Grip",
    "Yoke",
    "Foreign Objects on Power Lines",
    "broken power lines",
    "ice on transmission lines",
]

NEW_NAMES = [
    "Damper - Spiral",
    "Damper - Stockbridge",
    "Glass Insulator",
    "Lightning Rod Suspension",
    "Polymer Insulator",
    "Polymer Insulator Shackle",
    "Vari-Grip",
    "Yoke",
    "Bird Nest",
    "Plastic Bag",
    "Fluttering Object",
    "Balloon",
    "Other Foreign Object",
    "Broken Power Lines",
    "Ice on Transmission Lines",
    "Broken Insulator",
]

OLD_TO_NEW = {**{index: index for index in range(8)}, 8: 12, 9: 13, 10: 14}
RAILFOD_TO_NEW = {1: 8, 2: 9, 3: 10, 4: 11}
HF_TO_NEW = {0: 13, 1: 15}
SPLITS = ("train", "valid", "test")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--railfod-zip",
        type=Path,
        default=Path("external_data/railfod23/RailFOD23_v2.zip"),
    )
    parser.add_argument(
        "--hf-dir",
        type=Path,
        default=Path("external_data/powerline_components_faults_hf/data"),
    )
    parser.add_argument("--seed", default="powerline-rtdetr-fine-v1")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def normalized_railfod_group(file_name: str) -> str:
    return re.sub(r"_1(?=\.[^.]+$)", "", file_name.lower())


def normalized_hf_group(file_name: str) -> str:
    return file_name.lower().split(".rf.")[0]


def assigned_split(group: str, seed: str) -> str:
    value = int.from_bytes(
        hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()[:8], "big"
    )
    bucket = value % 10_000
    if bucket < 7_000:
        return "train"
    if bucket < 8_500:
        return "valid"
    return "test"


def yolo_line(class_id: int, bbox_xywh, width: int, height: int) -> str:
    x, y, box_width, box_height = bbox_xywh
    center_x = (x + box_width / 2) / width
    center_y = (y + box_height / 2) / height
    return (
        f"{class_id} {center_x:.8f} {center_y:.8f} "
        f"{box_width / width:.8f} {box_height / height:.8f}"
    )


def prepare_staging(root: Path, railfod_zip: Path, hf_dir: Path, seed: str):
    staging = root / "external_data" / "rtdetr_import_staging"
    if staging.exists():
        resolved = staging.resolve()
        expected_parent = (root / "external_data").resolve()
        if resolved.parent != expected_parent or resolved.name != "rtdetr_import_staging":
            raise RuntimeError(f"Unsafe staging path: {resolved}")
        shutil.rmtree(staging)
    for split in SPLITS:
        (staging / split / "images").mkdir(parents=True, exist_ok=True)
        (staging / split / "labels").mkdir(parents=True, exist_ok=True)

    source_counts = defaultdict(Counter)
    source_images = Counter()
    source_groups = defaultdict(lambda: defaultdict(set))

    with zipfile.ZipFile(railfod_zip) as archive:
        annotation_path = "coco/New_an/train_val.json"
        coco = json.loads(archive.read(annotation_path))
        image_by_id = {image["id"]: image for image in coco["images"]}
        annotations = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations[annotation["image_id"]].append(annotation)

        for image_id, image_info in image_by_id.items():
            file_name = image_info["file_name"]
            group = normalized_railfod_group(file_name)
            split = assigned_split(f"railfod:{group}", seed)
            source_groups["railfod"][split].add(group)
            output_name = f"railfod23_{file_name}"
            image_output = staging / split / "images" / output_name
            label_output = staging / split / "labels" / f"{Path(output_name).stem}.txt"

            image_bytes = archive.read(f"coco/Images/{file_name}")
            image_output.write_bytes(image_bytes)
            lines = []
            present_classes = set()
            for annotation in annotations[image_id]:
                new_class = RAILFOD_TO_NEW[annotation["category_id"]]
                lines.append(
                    yolo_line(
                        new_class,
                        annotation["bbox"],
                        image_info["width"],
                        image_info["height"],
                    )
                )
                source_counts[("railfod", split)][new_class] += 1
                present_classes.add(new_class)
            label_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
            source_images[("railfod", split)] += 1
            for class_id in present_classes:
                source_counts[("railfod_images", split)][class_id] += 1

    parquet_files = sorted(hf_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {hf_dir}")

    crop_hashes = set()
    hf_skipped = Counter()
    for parquet_path in parquet_files:
        rows = pq.read_table(parquet_path).to_pylist()
        for row_index, row in enumerate(rows):
            file_name = row["image"]["path"]
            group = normalized_hf_group(file_name)
            split = assigned_split(f"hf:{group}", seed)
            source_groups["hf"][split].add(group)
            source = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            width, height = source.size

            for object_index, (bbox, source_class) in enumerate(
                zip(row["bboxes"], row["labels"])
            ):
                if source_class not in HF_TO_NEW:
                    continue
                if len(bbox) != 4:
                    hf_skipped["invalid_length"] += 1
                    continue
                x1, y1, x2, y2 = map(float, bbox)
                x1 = max(0.0, min(float(width), x1))
                y1 = max(0.0, min(float(height), y1))
                x2 = max(0.0, min(float(width), x2))
                y2 = max(0.0, min(float(height), y2))
                if x2 - x1 < 8 or y2 - y1 < 8:
                    hf_skipped["invalid_or_tiny_box"] += 1
                    continue

                padding_x = (x2 - x1) * 0.20
                padding_y = (y2 - y1) * 0.20
                left = max(0, int(x1 - padding_x))
                top = max(0, int(y1 - padding_y))
                right = min(width, int(x2 + padding_x + 0.999999))
                bottom = min(height, int(y2 + padding_y + 0.999999))
                crop = source.crop((left, top, right, bottom))

                buffer = io.BytesIO()
                crop.save(buffer, format="JPEG", quality=95)
                crop_bytes = buffer.getvalue()
                crop_hash = hashlib.sha256(crop_bytes).hexdigest()
                if crop_hash in crop_hashes:
                    hf_skipped["exact_duplicate_crop"] += 1
                    continue
                crop_hashes.add(crop_hash)

                token = hashlib.sha1(
                    f"{file_name}:{row_index}:{object_index}".encode("utf-8")
                ).hexdigest()[:16]
                output_name = f"hfpclf_{token}.jpg"
                image_output = staging / split / "images" / output_name
                label_output = staging / split / "labels" / f"hfpclf_{token}.txt"
                image_output.write_bytes(crop_bytes)

                crop_width, crop_height = crop.size
                relative_box = (
                    x1 - left,
                    y1 - top,
                    x2 - x1,
                    y2 - y1,
                )
                new_class = HF_TO_NEW[source_class]
                label_output.write_text(
                    yolo_line(new_class, relative_box, crop_width, crop_height) + "\n",
                    encoding="utf-8",
                )
                source_counts[("hf", split)][new_class] += 1
                source_counts[("hf_images", split)][new_class] += 1
                source_images[("hf", split)] += 1

    # Validate staging pairs and class IDs.
    for split in SPLITS:
        images = {
            path.stem for path in (staging / split / "images").iterdir() if path.is_file()
        }
        labels = {path.stem for path in (staging / split / "labels").glob("*.txt")}
        if images != labels:
            raise RuntimeError(
                f"Staging image/label mismatch in {split}: "
                f"{len(images - labels)} images without labels, "
                f"{len(labels - images)} labels without images"
            )
        for label_path in (staging / split / "labels").glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                class_id = int(line.split()[0])
                if not 0 <= class_id < len(NEW_NAMES):
                    raise RuntimeError(f"Invalid staged class {class_id}: {label_path}")

    summary = {
        "railfod": {
            split: {
                "images": source_images[("railfod", split)],
                "unique_groups": len(source_groups["railfod"][split]),
                "class_instances": {
                    str(class_id): source_counts[("railfod", split)][class_id]
                    for class_id in RAILFOD_TO_NEW.values()
                },
                "class_images": {
                    str(class_id): source_counts[("railfod_images", split)][class_id]
                    for class_id in RAILFOD_TO_NEW.values()
                },
            }
            for split in SPLITS
        },
        "hf_powerline_faults": {
            split: {
                "images": source_images[("hf", split)],
                "unique_groups": len(source_groups["hf"][split]),
                "class_instances": {
                    str(class_id): source_counts[("hf", split)][class_id]
                    for class_id in HF_TO_NEW.values()
                },
            }
            for split in SPLITS
        },
        "hf_skipped": dict(hf_skipped),
    }
    return staging, summary


def backup_current_dataset(root: Path) -> Path:
    backup = root / "external_data" / "rtdetr_11class_labels_backup.zip"
    if backup.exists():
        return backup
    with zipfile.ZipFile(backup, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(root / "data.yaml", "data.yaml")
        for split in SPLITS:
            for label_path in sorted((root / split / "labels").glob("*.txt")):
                archive.write(label_path, f"{split}/labels/{label_path.name}")
    return backup


def remap_current_labels(root: Path):
    counts = Counter()
    for split in SPLITS:
        for label_path in sorted((root / split / "labels").glob("*.txt")):
            if label_path.name.startswith(("railfod23_", "hfpclf_")):
                continue
            output_lines = []
            for line in label_path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                old_class = int(parts[0])
                if old_class not in OLD_TO_NEW:
                    raise RuntimeError(f"Unexpected old class {old_class}: {label_path}")
                new_class = OLD_TO_NEW[old_class]
                parts[0] = str(new_class)
                output_lines.append(" ".join(parts))
                counts[(split, new_class)] += 1
            temporary = label_path.with_suffix(".txt.remap_tmp")
            temporary.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
            os.replace(temporary, label_path)
    return counts


def install_staging(root: Path, staging: Path):
    installed = Counter()
    for split in SPLITS:
        for kind in ("images", "labels"):
            destination = root / split / kind
            for source in (staging / split / kind).iterdir():
                target = destination / source.name
                if target.exists():
                    raise FileExistsError(f"Refusing to overwrite: {target}")
                shutil.move(str(source), str(target))
                installed[(split, kind)] += 1
    return installed


def update_data_yaml(root: Path):
    path = root / "data.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["nc"] = len(NEW_NAMES)
    config["names"] = NEW_NAMES
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def scan_final(root: Path):
    result = {}
    for split in SPLITS:
        images = {
            path.stem for path in (root / split / "images").iterdir() if path.is_file()
        }
        labels = {path.stem for path in (root / split / "labels").glob("*.txt")}
        class_instances = Counter()
        class_images = Counter()
        invalid = []
        for label_path in (root / split / "labels").glob("*.txt"):
            present = set()
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                parts = line.split()
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
                if not 0 <= class_id < len(NEW_NAMES):
                    invalid.append(f"{label_path}:{line_number}:class")
                if any(not 0.0 <= value <= 1.0 for value in values):
                    invalid.append(f"{label_path}:{line_number}:coordinate")
                class_instances[class_id] += 1
                present.add(class_id)
            for class_id in present:
                class_images[class_id] += 1
        result[split] = {
            "images": len(images),
            "labels": len(labels),
            "images_without_labels": len(images - labels),
            "labels_without_images": len(labels - images),
            "invalid_annotations": len(invalid),
            "class_images": {
                str(index): class_images[index] for index in range(len(NEW_NAMES))
            },
            "class_instances": {
                str(index): class_instances[index] for index in range(len(NEW_NAMES))
            },
        }
    return result


def main():
    args = parse_args()
    root = args.root.resolve()
    railfod_zip = (root / args.railfod_zip).resolve()
    hf_dir = (root / args.hf_dir).resolve()

    if not railfod_zip.is_file():
        raise FileNotFoundError(railfod_zip)
    current = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    if current.get("names") != OLD_NAMES:
        raise RuntimeError(
            "data.yaml is not the expected 11-class source dataset; "
            "refusing to remap labels."
        )

    staging, external_summary = prepare_staging(
        root, railfod_zip, hf_dir, args.seed
    )
    preview = {
        "new_names": NEW_NAMES,
        "external_summary": external_summary,
        "staging": str(staging),
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if not args.apply:
        print("Dry run complete. Re-run with --apply to install the staged dataset.")
        return

    backup = backup_current_dataset(root)
    remapped = remap_current_labels(root)
    installed = install_staging(root, staging)
    update_data_yaml(root)
    final_summary = scan_final(root)

    manifest = {
        "format": "fine-grained RT-DETR dataset import",
        "new_names": NEW_NAMES,
        "old_to_new": {str(key): value for key, value in OLD_TO_NEW.items()},
        "sources": {
            "railfod23": {
                "url": "https://doi.org/10.6084/m9.figshare.24180738.v3",
                "license": "CC BY 4.0",
                "archive_md5": "888c0a2ac1a222a086dbe72d60eca670",
            },
            "powerline_components_and_faults": {
                "url": (
                    "https://huggingface.co/datasets/"
                    "docmhvr/powerline-components-and-faults"
                ),
                "license": "MIT (as stated by the dataset card)",
            },
        },
        "split_method": (
            "SHA-256 deterministic 70/15/15 split grouped by normalized source ID"
        ),
        "external_summary": external_summary,
        "backup": str(backup.relative_to(root)),
        "remapped_original_instances": {
            f"{split}:{class_id}": count
            for (split, class_id), count in sorted(remapped.items())
        },
        "installed_files": {
            f"{split}:{kind}": count
            for (split, kind), count in sorted(installed.items())
        },
        "final_summary": final_summary,
    }
    manifest_path = root / "external_data" / "rtdetr_fine_import_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(staging)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
