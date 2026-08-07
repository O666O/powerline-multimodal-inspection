"""使用当前 YOLO 数据集微调 Ultralytics RT-DETR。"""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="训练 RT-DETR 目标检测模型")
    parser.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    parser.add_argument("--model", default="rtdetr-l.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="例如 0、0,1 或 cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="RTX 5090上默认关闭AMP以避免RT-DETR损失出现NaN",
    )
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "rtdetr")
    parser.add_argument("--name", default="powerline_rtdetr_l")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"找不到数据配置：{args.data}")

    config_dir = ROOT / "runs" / ".ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    try:
        from ultralytics import RTDETR
    except ImportError as exc:
        raise SystemExit("请先安装依赖：pip install -U ultralytics") from exc

    model = RTDETR(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project.resolve()),
        name=args.name,
        resume=args.resume,
        cache=args.cache,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        amp=args.amp,
        plots=True,
        val=True,
        seed=42,
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
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=0,
    )

    best_weight = Path(model.trainer.best)
    if not best_weight.is_file():
        raise FileNotFoundError(f"找不到最佳训练权重：{best_weight}")

    best_model = RTDETR(str(best_weight))
    common_val_args = {
        "data": str(args.data.resolve()),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "plots": True,
    }
    validation = best_model.val(split="val", **common_val_args)
    test = best_model.val(split="test", **common_val_args)

    def serializable_results(metrics):
        return {
            str(key): float(value)
            for key, value in metrics.results_dict.items()
        }

    report = {
        "best_weight": str(best_weight.resolve()),
        "validation": serializable_results(validation),
        "test": serializable_results(test),
    }
    report_path = best_weight.parent.parent / "evaluation_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nRT-DETR 最终评价指标：")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"评价报告已保存：{report_path}")


if __name__ == "__main__":
    main()
