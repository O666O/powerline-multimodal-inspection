import argparse
import base64
import io
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


SOURCE_CLASS_NAMES = [
    "螺旋式防振器",
    "斯托克布里奇防振锤",
    "玻璃绝缘子",
    "避雷线悬垂装置",
    "复合绝缘子",
    "复合绝缘子挂环",
    "预绞式耐张线夹",
    "联板",
    "输电线路异物",
    "断裂的输电线路",
    "输电线路覆冰",
]

# Chinese-CLIP 只负责设备类别识别；异常/缺陷类别交给 RT-DETR。
DEVICE_CLASS_IDS = tuple(range(8))
CLASS_NAMES = [SOURCE_CLASS_NAMES[class_id] for class_id in DEVICE_CLASS_IDS]
SOURCE_TO_CLIP_CLASS = {
    source_class_id: clip_class_id
    for clip_class_id, source_class_id in enumerate(DEVICE_CLASS_IDS)
}

CAPTION_TEMPLATES = [
    "输电线路上的{label}",
    "无人机巡检图像中的{label}",
    "高压输电线路设备：{label}",
    "电力巡检发现的{label}",
    "航拍画面中的{label}",
    "输电线路局部区域的{label}",
]


def parse_box(parts):
    values = [float(value) for value in parts[1:]]
    if len(parts) == 5:
        cx, cy, width, height = values
        return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2

    xs = values[0::2]
    ys = values[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def padded_pixel_box(box, width, height, padding):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    x1 -= box_width * padding
    x2 += box_width * padding
    y1 -= box_height * padding
    y2 += box_height * padding

    left = max(0, min(width - 1, int(x1 * width)))
    top = max(0, min(height - 1, int(y1 * height)))
    right = max(left + 1, min(width, int(x2 * width + 0.999999)))
    bottom = max(top + 1, min(height, int(y2 * height + 0.999999)))

    min_crop_size = 4
    if right - left < min_crop_size:
        center = (left + right) // 2
        left = max(0, center - min_crop_size // 2)
        right = min(width, left + min_crop_size)
        left = max(0, right - min_crop_size)
    if bottom - top < min_crop_size:
        center = (top + bottom) // 2
        top = max(0, center - min_crop_size // 2)
        bottom = min(height, top + min_crop_size)
        top = max(0, bottom - min_crop_size)
    return left, top, right, bottom


def build_split(root, output, split, padding, jpeg_quality):
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    crop_dir = output / "images" / split
    crop_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = output / f"{split}_pairs.jsonl"
    images_tsv_path = output / f"{split}_imgs.tsv"
    texts_path = output / f"{split}_texts.jsonl"
    class_counts = Counter()
    excluded_annotations = Counter()
    skipped = []
    image_id = 0
    text_id = 0

    with (
        pairs_path.open("w", encoding="utf-8", newline="\n") as pairs_file,
        images_tsv_path.open("w", encoding="utf-8", newline="\n") as images_file,
        texts_path.open("w", encoding="utf-8", newline="\n") as texts_file,
    ):
        for label_path in sorted(label_dir.glob("*.txt")):
            source_image = image_dir / f"{label_path.stem}.jpg"
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            if not lines:
                continue

            with Image.open(source_image) as source:
                source = source.convert("RGB")
                width, height = source.size

                for object_index, line in enumerate(lines):
                    parts = line.split()
                    source_class_id = int(parts[0])
                    if not 0 <= source_class_id < len(SOURCE_CLASS_NAMES):
                        raise ValueError(
                            f"Invalid class {source_class_id}: {label_path}"
                        )
                    if source_class_id not in SOURCE_TO_CLIP_CLASS:
                        excluded_annotations[source_class_id] += 1
                        continue
                    class_id = SOURCE_TO_CLIP_CLASS[source_class_id]

                    pixel_box = padded_pixel_box(
                        parse_box(parts), width, height, padding
                    )
                    if pixel_box[2] - pixel_box[0] < 2 or pixel_box[3] - pixel_box[1] < 2:
                        skipped.append(
                            {
                                "source": str(source_image.relative_to(root)),
                                "object_index": object_index,
                                "reason": "crop_too_small",
                            }
                        )
                        continue

                    crop = source.crop(pixel_box)
                    buffer = io.BytesIO()
                    crop.save(buffer, format="JPEG", quality=jpeg_quality)
                    crop_bytes = buffer.getvalue()
                    crop_name = f"{image_id:08d}.jpg"
                    crop_path = crop_dir / crop_name
                    crop_path.write_bytes(crop_bytes)

                    label = CLASS_NAMES[class_id]
                    template_index = (image_id + class_id) % len(CAPTION_TEMPLATES)
                    caption = CAPTION_TEMPLATES[template_index].format(label=label)
                    relative_crop = crop_path.relative_to(output).as_posix()
                    relative_source = source_image.relative_to(root).as_posix()

                    pair = {
                        "image_id": image_id,
                        "text_id": text_id,
                        "image": relative_crop,
                        "text": caption,
                        "class_id": class_id,
                        "class_name": label,
                        "source_image": relative_source,
                        "source_object_index": object_index,
                        "crop_xyxy": list(pixel_box),
                    }
                    pairs_file.write(
                        json.dumps(pair, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    images_file.write(
                        f"{image_id}\t{base64.b64encode(crop_bytes).decode('ascii')}\n"
                    )
                    texts_file.write(
                        json.dumps(
                            {
                                "text_id": text_id,
                                "text": caption,
                                "image_ids": [image_id],
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    class_counts[class_id] += 1
                    image_id += 1
                    text_id += 1

    return {
        "split": split,
        "pairs": image_id,
        "class_annotations": {
            str(class_id): class_counts[class_id]
            for class_id in range(len(CLASS_NAMES))
        },
        "excluded_non_device_annotations": {
            str(class_id): excluded_annotations[class_id]
            for class_id in range(len(SOURCE_CLASS_NAMES))
            if class_id not in SOURCE_TO_CLIP_CLASS
        },
        "skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("chinese_clip_dataset")
    )
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    if output == root or root in output.parents and output.name in {
        "train",
        "valid",
        "test",
    }:
        raise ValueError(f"拒绝清理不安全的输出目录：{output}")
    output.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        crop_dir = output / "images" / split
        if crop_dir.exists():
            shutil.rmtree(crop_dir)
        for suffix in ("pairs.jsonl", "imgs.tsv", "texts.jsonl"):
            generated_file = output / f"{split}_{suffix}"
            if generated_file.exists():
                generated_file.unlink()

    summaries = [
        build_split(
            root,
            output,
            split,
            padding=args.padding,
            jpeg_quality=args.jpeg_quality,
        )
        for split in ("train", "valid", "test")
    ]

    (output / "label_cn.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n", encoding="utf-8"
    )
    metadata = {
        "format": "Chinese-CLIP image-text pairs",
        "source_format": "YOLO detection/segmentation",
        "crop_padding": args.padding,
        "jpeg_quality": args.jpeg_quality,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "source_device_class_ids": list(DEVICE_CLASS_IDS),
        "caption_templates": CAPTION_TEMPLATES,
        "splits": summaries,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
