"""Import deduplicated MPCD broken-cable samples into the RT-DETR train split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--class-id", type=int, default=12)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    root = args.root.resolve()
    external_root = root / "external_data" / "mpcd"
    dataset_root = external_root / "extracted" / "MulticableData"
    candidate_path = external_root / "new_candidates.txt"
    manifest_path = external_root / "import_manifest.jsonl"
    image_destination = root / "train" / "images"
    label_destination = root / "train" / "labels"

    if manifest_path.exists():
        raise FileExistsError(f"Import manifest already exists: {manifest_path}")

    candidates = []
    for raw_line in candidate_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        source_text, distance_text = raw_line.rsplit("\t", 1)
        source_image = root / Path(source_text)
        if source_image.parent.name != "train2017":
            continue
        source_label = (
            dataset_root
            / "detection_brokencable"
            / "labels"
            / "train2017"
            / f"{source_image.stem}.txt"
        )
        destination_stem = f"mpcd_train_{source_image.stem}"
        destination_image = image_destination / f"{destination_stem}.jpg"
        destination_label = label_destination / f"{destination_stem}.txt"
        candidates.append(
            {
                "source_image": source_image,
                "source_label": source_label,
                "distance": int(distance_text),
                "destination_image": destination_image,
                "destination_label": destination_label,
            }
        )

    prepared = []
    for item in candidates:
        if not item["source_image"].is_file() or not item["source_label"].is_file():
            raise FileNotFoundError(f"Missing source pair: {item}")
        if item["destination_image"].exists() or item["destination_label"].exists():
            raise FileExistsError(f"Destination collision: {item}")

        with Image.open(item["source_image"]) as image:
            image.verify()

        converted_lines = []
        for line_number, line in enumerate(
            item["source_label"].read_text(encoding="utf-8-sig").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5 or int(parts[0]) != 0:
                raise ValueError(
                    f"Unexpected MPCD label at {item['source_label']}:{line_number}"
                )
            values = [float(value) for value in parts[1:]]
            cx, cy, width, height = values
            if not (
                0 <= cx <= 1
                and 0 <= cy <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                raise ValueError(
                    f"Out-of-range MPCD box at {item['source_label']}:{line_number}"
                )
            converted_lines.append(
                f"{args.class_id} " + " ".join(f"{value:.8f}" for value in values)
            )
        if not converted_lines:
            raise ValueError(f"No boxes in MPCD label: {item['source_label']}")

        item["converted_lines"] = converted_lines
        item["sha256"] = file_sha256(item["source_image"])
        prepared.append(item)

    print(f"Validated MPCD training samples: {len(prepared)}")
    print(f"Broken-cable boxes: {sum(len(x['converted_lines']) for x in prepared)}")
    if not args.apply:
        print("Dry run complete. Re-run with --apply to import files.")
        return

    manifest_rows = []
    for item in prepared:
        shutil.copy2(item["source_image"], item["destination_image"])
        temporary = item["destination_label"].with_suffix(".txt.import_tmp")
        temporary.write_text(
            "\n".join(item["converted_lines"]) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, item["destination_label"])
        manifest_rows.append(
            {
                "source": item["source_image"].relative_to(root).as_posix(),
                "destination_image": item["destination_image"].relative_to(root).as_posix(),
                "destination_label": item["destination_label"].relative_to(root).as_posix(),
                "source_sha256": item["sha256"],
                "nearest_existing_dhash_distance": item["distance"],
                "source_dataset": "MPCD",
                "source_url": "https://github.com/phd-benel/PowerLine-MTYOLO",
            }
        )

    manifest_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"Imported samples: {len(manifest_rows)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
