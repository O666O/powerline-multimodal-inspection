"""Create a reproducible 2x4 qualitative detection figure from a YOLO test run.

The script compares held-out YOLO ground-truth labels with prediction labels
created by ``yolo detect predict ... save_txt=True save_conf=True``. It selects
representative success and failure cases deterministically, draws ground-truth
and predicted boxes together, and records the selected images in JSON.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml
    from matplotlib.patches import Rectangle
    from PIL import Image
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}. Install plotting dependencies with: "
        "pip install matplotlib pillow pyyaml numpy"
    ) from exc


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--images", default="test/images")
    parser.add_argument("--gt-labels", default="test/labels")
    parser.add_argument(
        "--pred-labels",
        default="runs/qualitative/test_predictions/labels",
    )
    parser.add_argument(
        "--output",
        default="runs/qualitative/qualitative_test_cases.png",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--small-area", type=float, default=0.01)
    return parser.parse_args()


def load_names(data_path: Path) -> list[str]:
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names, key=lambda x: int(x))]
    return [str(name) for name in names]


def load_boxes(path: Path, width: int, height: int, prediction: bool) -> list[Box]:
    if not path.exists():
        return []
    boxes: list[Box] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) < 5:
            raise ValueError(f"Invalid YOLO row at {path}:{line_number}: {line}")
        cls = int(float(values[0]))
        xc, yc, bw, bh = map(float, values[1:5])
        conf = float(values[5]) if prediction and len(values) >= 6 else 1.0
        x1 = max(0.0, (xc - bw / 2.0) * width)
        y1 = max(0.0, (yc - bh / 2.0) * height)
        x2 = min(float(width), (xc + bw / 2.0) * width)
        y2 = min(float(height), (yc + bh / 2.0) * height)
        boxes.append(Box(cls, x1, y1, x2, y2, conf))
    return boxes


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt: list[Box], pred: list[Box], threshold: float):
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g.cls == p.cls:
                overlap = iou(g, p)
                if overlap >= threshold:
                    candidates.append((overlap, gi, pi))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for overlap, gi, pi in sorted(candidates, reverse=True):
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            pairs.append((gi, pi, overlap))
    return matched_gt, matched_pred, pairs


def analyze_images(args: argparse.Namespace, names: list[str]) -> list[dict]:
    image_dir = Path(args.images)
    gt_dir = Path(args.gt_labels)
    pred_dir = Path(args.pred_labels)
    broken_id = next(
        (i for i, name in enumerate(names) if name.lower() == "broken power lines"),
        12,
    )
    records = []
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        with Image.open(image_path) as image:
            width, height = image.size
        gt = load_boxes(gt_dir / f"{image_path.stem}.txt", width, height, False)
        pred = load_boxes(pred_dir / f"{image_path.stem}.txt", width, height, True)
        matched_gt, matched_pred, pairs = match_boxes(gt, pred, args.iou)
        unmatched_gt = [i for i in range(len(gt)) if i not in matched_gt]
        unmatched_pred = [i for i in range(len(pred)) if i not in matched_pred]
        normalized_areas = [box.area / (width * height) for box in gt]
        records.append(
            {
                "image": image_path,
                "width": width,
                "height": height,
                "gt": gt,
                "pred": pred,
                "pairs": pairs,
                "matched_gt": matched_gt,
                "matched_pred": matched_pred,
                "unmatched_gt": unmatched_gt,
                "unmatched_pred": unmatched_pred,
                "broken_id": broken_id,
                "areas": normalized_areas,
            }
        )
    return records


def select_cases(records: list[dict], small_area: float) -> list[tuple[str, dict]]:
    def is_aigc(r):
        return "aigc" in r["image"].name.lower()

    def no_errors(r):
        return not r["unmatched_gt"] and not r["unmatched_pred"] and bool(r["gt"])

    def abnormal_count(r):
        return sum(box.cls >= 8 for box in r["gt"])

    def field_abnormal_count(r):
        # Prefer defects that are meaningful in actual line-inspection scenes.
        return sum(box.cls in {8, 9, 10, 12, 13} for box in r["gt"])

    def broken_tp_count(r):
        return sum(r["gt"][gi].cls == r["broken_id"] for gi, _, _ in r["pairs"])

    def broken_fn_count(r):
        return sum(r["gt"][i].cls == r["broken_id"] for i in r["unmatched_gt"])

    def broken_fp_count(r):
        return sum(r["pred"][i].cls == r["broken_id"] for i in r["unmatched_pred"])

    specifications = [
        (
            "Equipment success",
            lambda r: no_errors(r)
            and not is_aigc(r)
            and 1 <= len(r["gt"]) <= 6
            and any(box.cls < 8 for box in r["gt"]),
            lambda r: (len(r["gt"]), sum(p.conf for p in r["pred"])),
        ),
        (
            "Multi-object success",
            lambda r: no_errors(r) and not is_aigc(r) and 3 <= len(r["gt"]) <= 8,
            lambda r: (len(r["gt"]), sum(p.conf for p in r["pred"])),
        ),
        (
            "Abnormal-object success",
            lambda r: no_errors(r)
            and not is_aigc(r)
            and 1 <= len(r["gt"]) <= 6
            and field_abnormal_count(r) > 0,
            lambda r: (field_abnormal_count(r), sum(p.conf for p in r["pred"])),
        ),
        (
            "Small-target success",
            lambda r: no_errors(r)
            and not is_aigc(r)
            and len(r["gt"]) <= 8
            and any(a <= small_area for a in r["areas"]),
            lambda r: (-min(r["areas"]), len(r["gt"])),
        ),
        (
            "Broken conductor: TP",
            lambda r: broken_tp_count(r) > 0,
            lambda r: (broken_tp_count(r), max((p.conf for p in r["pred"]), default=0.0)),
        ),
        (
            "Broken conductor: FN",
            lambda r: broken_fn_count(r) > 0,
            lambda r: (broken_fn_count(r), len(r["gt"])),
        ),
        (
            "Broken conductor: hard FP",
            lambda r: not any(g.cls == r["broken_id"] for g in r["gt"]) and broken_fp_count(r) > 0,
            lambda r: (
                broken_fp_count(r),
                max((r["pred"][i].conf for i in r["unmatched_pred"] if r["pred"][i].cls == r["broken_id"]), default=0.0),
            ),
        ),
        (
            "Complex-scene failure",
            lambda r: not is_aigc(r)
            and 2 <= len(r["gt"]) <= 8
            and len(r["pred"]) <= 10
            and bool(r["unmatched_gt"] or r["unmatched_pred"]),
            lambda r: (len(r["unmatched_gt"]) + len(r["unmatched_pred"]), len(r["gt"])),
        ),
    ]

    selected: list[tuple[str, dict]] = []
    used: set[Path] = set()
    for title, predicate, score in specifications:
        candidates = [r for r in records if r["image"] not in used and predicate(r)]
        if candidates:
            chosen = max(candidates, key=lambda r: (score(r), r["image"].name))
            selected.append((title, chosen))
            used.add(chosen["image"])

    fallbacks = sorted(
        (r for r in records if r["image"] not in used and r["gt"]),
        key=lambda r: (
            bool(r["unmatched_gt"] or r["unmatched_pred"]),
            len(r["gt"]),
            r["image"].name,
        ),
        reverse=True,
    )
    while len(selected) < 8 and fallbacks:
        chosen = fallbacks.pop(0)
        selected.append(("Additional representative case", chosen))
        used.add(chosen["image"])
    if len(selected) < 8:
        raise RuntimeError(f"Only {len(selected)} usable cases were found; eight are required.")
    return selected[:8]


def draw_box(ax, box: Box, name: str, ground_truth: bool) -> None:
    color = "#20c55a" if ground_truth else "#ef4444"
    linestyle = "-" if ground_truth else "--"
    prefix = "GT" if ground_truth else "P"
    label = f"{prefix}: {name}" if ground_truth else f"{prefix}: {name} {box.conf:.2f}"
    ax.add_patch(
        Rectangle(
            (box.x1, box.y1),
            box.x2 - box.x1,
            box.y2 - box.y1,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            linestyle=linestyle,
        )
    )
    ax.text(
        box.x1,
        max(2.0, box.y1 - 3.0),
        label,
        color="white",
        fontsize=7,
        va="bottom",
        bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def create_figure(selected: list[tuple[str, dict]], names: list[str], output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.5), constrained_layout=True)
    panel_letters = "abcdefgh"
    audit = []
    for index, (title, record) in enumerate(selected):
        ax = axes.flat[index]
        with Image.open(record["image"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        ax.imshow(rgb)
        for box in record["gt"]:
            draw_box(ax, box, names[box.cls] if box.cls < len(names) else str(box.cls), True)
        for box in record["pred"]:
            draw_box(ax, box, names[box.cls] if box.cls < len(names) else str(box.cls), False)
        tp = len(record["pairs"])
        fp = len(record["unmatched_pred"])
        fn = len(record["unmatched_gt"])
        ax.set_title(f"({panel_letters[index]}) {title}\nTP={tp}, FP={fp}, FN={fn}", fontsize=10)
        ax.axis("off")
        audit.append(
            {
                "panel": panel_letters[index],
                "case": title,
                "image": str(record["image"]),
                "ground_truth_boxes": len(record["gt"]),
                "prediction_boxes": len(record["pred"]),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            }
        )
    fig.suptitle(
        "YOLOv8-L qualitative results on the held-out test split\n"
        "Ground truth: green solid boxes; predictions: red dashed boxes",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output.with_suffix(".json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_labels)
    if not pred_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction-label directory not found: {pred_dir}. "
            "Run YOLO prediction with save_txt=True and pass the resulting labels directory "
            "through --pred-labels."
        )
    prediction_files = list(pred_dir.glob("*.txt"))
    if not prediction_files:
        raise RuntimeError(
            f"No prediction label files were found in: {pred_dir}. "
            "Run YOLO prediction with save_txt=True before creating the figure."
        )
    names = load_names(Path(args.data))
    records = analyze_images(args, names)
    selected = select_cases(records, args.small_area)
    output = Path(args.output)
    create_figure(selected, names, output)
    print(f"Analyzed {len(records)} held-out test images.")
    print(f"Figure saved to: {output}")
    print(f"Selection audit saved to: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
