"""将 YOLO 多边形标注转换为外接矩形，并保存原始行以便恢复。"""

import argparse
import json
from pathlib import Path


def polygon_to_box(parts):
    class_id = parts[0]
    values = [float(value) for value in parts[1:]]
    xs = values[0::2]
    ys = values[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return (
        f"{class_id} {cx:.6f} {cy:.6f} "
        f"{width:.6f} {height:.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--backup", type=Path, default=Path("polygon_annotations_backup.jsonl")
    )
    args = parser.parse_args()

    root = args.root.resolve()
    backup = args.backup
    if not backup.is_absolute():
        backup = root / backup
    if backup.exists():
        raise FileExistsError(f"备份文件已存在，停止以避免覆盖：{backup}")

    converted = 0
    changed_files = 0
    with backup.open("w", encoding="utf-8", newline="\n") as backup_file:
        for split in ("train", "valid", "test", "quarantine"):
            label_dir = root / split / "labels"
            if not label_dir.is_dir():
                continue
            for label_path in sorted(label_dir.glob("*.txt")):
                original_lines = label_path.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                output_lines = []
                changed = False
                for line_index, line in enumerate(original_lines):
                    stripped = line.strip()
                    if not stripped:
                        output_lines.append(line)
                        continue
                    parts = stripped.split()
                    if len(parts) > 5:
                        backup_file.write(
                            json.dumps(
                                {
                                    "split": split,
                                    "label": label_path.name,
                                    "line_index": line_index,
                                    "original": stripped,
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        output_lines.append(polygon_to_box(parts))
                        converted += 1
                        changed = True
                    else:
                        output_lines.append(stripped)

                if changed:
                    label_path.write_text(
                        "\n".join(output_lines) + "\n", encoding="utf-8"
                    )
                    changed_files += 1

    print(f"converted_polygons={converted}")
    print(f"changed_label_files={changed_files}")
    print(f"backup={backup}")


if __name__ == "__main__":
    main()
