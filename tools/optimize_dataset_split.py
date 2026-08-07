"""按拍摄序列重新划分 YOLO 数据集，降低相邻帧跨集合泄漏。"""

import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "valid", "test")
RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}


def source_and_frame(stem):
    original = stem.split("_jpg.rf.")[0]
    match = re.match(r"^(.*?)(\d+)$", original)
    if not match:
        return original, None
    source = match.group(1).rstrip("_-")
    return source, int(match.group(2))


def read_records(root):
    records = []
    for split in SPLITS:
        for image_path in sorted((root / split / "images").glob("*.jpg")):
            label_path = root / split / "labels" / f"{image_path.stem}.txt"
            classes = set()
            for line in label_path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    classes.add(int(line.split()[0]))
            source, frame = source_and_frame(image_path.stem)
            records.append(
                {
                    "stem": image_path.stem,
                    "old_split": split,
                    "source": source,
                    "frame": frame,
                    "classes": sorted(classes),
                }
            )
    return records


def count_group(records):
    class_counts = Counter()
    for record in records:
        for class_id in record["classes"]:
            class_counts[class_id] += 1
    return {"images": len(records), "classes": class_counts}


def plan_split(records, long_sequence_threshold, boundary_gap):
    by_source = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)

    assignments = {}
    fixed_counts = {
        split: {"images": 0, "classes": Counter()} for split in SPLITS
    }
    flexible_groups = []
    quarantine = []

    for source, items in by_source.items():
        items.sort(
            key=lambda item: (
                item["frame"] is None,
                item["frame"] if item["frame"] is not None else item["stem"],
            )
        )
        numeric_frames = all(item["frame"] is not None for item in items)
        if len(items) >= long_sequence_threshold and numeric_frames:
            boundaries = list(
                range(long_sequence_threshold, len(items), long_sequence_threshold)
            )
            kept_items = []
            for index, item in enumerate(items):
                if any(abs(index - boundary) <= boundary_gap for boundary in boundaries):
                    assignments[item["stem"]] = "quarantine"
                    quarantine.append(item)
                else:
                    kept_items.append((index, item))

            chunks = defaultdict(list)
            for index, item in kept_items:
                block_id = index // long_sequence_threshold
                chunks[block_id].append(item)
            for block_id, chunk_items in chunks.items():
                group_name = f"{source}#block{block_id:03d}"
                flexible_groups.append(
                    (group_name, chunk_items, count_group(chunk_items))
                )
        else:
            flexible_groups.append((source, items, count_group(items)))

    usable_count = len(records) - len(quarantine)
    total_classes = Counter()
    for record in records:
        if assignments.get(record["stem"]) != "quarantine":
            total_classes.update(record["classes"])
    image_targets = {
        split: usable_count * RATIOS[split] for split in SPLITS
    }
    class_targets = {
        split: {
            class_id: total * RATIOS[split]
            for class_id, total in total_classes.items()
        }
        for split in SPLITS
    }

    flexible_groups.sort(
        key=lambda item: (
            -sum(
                count / max(total_classes[class_id], 1)
                for class_id, count in item[2]["classes"].items()
            ),
            -item[2]["images"],
            item[0],
        )
    )

    for source, items, stats in flexible_groups:
        best_split = None
        best_score = math.inf
        for split in SPLITS:
            image_before = fixed_counts[split]["images"]
            image_after = fixed_counts[split]["images"] + stats["images"]
            image_target = max(image_targets[split], 1)
            score = 5.0 * (
                ((image_after - image_target) / image_target) ** 2
                - ((image_before - image_target) / image_target) ** 2
            )
            for class_id in total_classes:
                added = stats["classes"][class_id]
                class_before = fixed_counts[split]["classes"][class_id]
                class_after = fixed_counts[split]["classes"][class_id] + added
                target = max(class_targets[split][class_id], 1)
                score += (
                    ((class_after - target) / target) ** 2
                    - ((class_before - target) / target) ** 2
                )
            if score < best_score:
                best_score = score
                best_split = split

        for item in items:
            assignments[item["stem"]] = best_split
        fixed_counts[best_split]["images"] += stats["images"]
        fixed_counts[best_split]["classes"].update(stats["classes"])

    return assignments, quarantine


def summarize(records, assignments):
    summary = {
        split: {"images": 0, "class_images": Counter()} for split in SPLITS
    }
    summary["quarantine"] = {"images": 0, "class_images": Counter()}
    moved = 0
    for record in records:
        destination = assignments[record["stem"]]
        summary[destination]["images"] += 1
        summary[destination]["class_images"].update(record["classes"])
        if destination != record["old_split"]:
            moved += 1
    return summary, moved


def apply_plan(root, records, assignments, manifest_path):
    staging = root / ".split_staging"
    staging.mkdir(exist_ok=True)
    for split in (*SPLITS, "quarantine"):
        (staging / split / "images").mkdir(parents=True, exist_ok=True)
        (staging / split / "labels").mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for record in records:
            destination = assignments[record["stem"]]
            old_split = record["old_split"]
            image_name = f"{record['stem']}.jpg"
            label_name = f"{record['stem']}.txt"
            old_image = root / old_split / "images" / image_name
            old_label = root / old_split / "labels" / label_name
            new_image = staging / destination / "images" / image_name
            new_label = staging / destination / "labels" / label_name
            shutil.move(old_image, new_image)
            shutil.move(old_label, new_label)
            manifest.write(
                json.dumps(
                    {
                        **record,
                        "new_split": destination,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    for split in SPLITS:
        for kind in ("images", "labels"):
            current = root / split / kind
            if any(current.iterdir()):
                raise RuntimeError(f"Staging failed; directory is not empty: {current}")
            current.rmdir()
            shutil.move(staging / split / kind, current)

    quarantine_root = root / "quarantine"
    quarantine_root.mkdir(exist_ok=True)
    for kind in ("images", "labels"):
        target = quarantine_root / kind
        if target.exists():
            raise FileExistsError(f"Quarantine already exists: {target}")
        shutil.move(staging / "quarantine" / kind, target)

    for split in (*SPLITS, "quarantine"):
        leftover = staging / split
        if leftover.exists():
            leftover.rmdir()
    staging.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--long-sequence-threshold", type=int, default=200)
    parser.add_argument("--boundary-gap", type=int, default=3)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    records = read_records(root)
    assignments, quarantine = plan_split(
        records,
        long_sequence_threshold=args.long_sequence_threshold,
        boundary_gap=args.boundary_gap,
    )
    summary, moved = summarize(records, assignments)
    output = {
        "total_source_images": len(records),
        "moved_or_quarantined": moved,
        "quarantine_images": len(quarantine),
        "splits": {
            split: {
                "images": values["images"],
                "class_images": dict(sorted(values["class_images"].items())),
            }
            for split, values in summary.items()
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.apply:
        manifest_path = root / "split_optimization_manifest.jsonl"
        apply_plan(root, records, assignments, manifest_path)
        (root / "split_optimization_summary.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Applied. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
