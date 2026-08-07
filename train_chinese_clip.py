"""使用生成的目标裁剪图文对微调 Hugging Face Chinese-CLIP。"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="微调 Chinese-CLIP")
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "chinese_clip_dataset"
    )
    parser.add_argument(
        "--model", default="OFA-Sys/chinese-clip-vit-base-patch16"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runs" / "chinese_clip"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-text-length", type=int, default=52)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--freeze-vision", action="store_true")
    parser.add_argument("--freeze-text", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="只读取已保存的微调权重执行测试，不重新训练",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="--eval-only 使用的权重目录；默认使用 output-dir/best",
    )
    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认按类别频率的平方根倒数进行均衡采样",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


class PairDataset:
    def __init__(self, manifest, data_dir):
        self.data_dir = data_dir
        with manifest.open("r", encoding="utf-8") as file:
            self.records = [json.loads(line) for line in file if line.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image

        record = self.records[index]
        image_path = self.data_dir / record["image"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return image, record["text"], int(record["class_id"])


def multi_positive_contrastive_loss(output, class_ids):
    """将同类别图文视为多个正样本，避免同类文本互相充当负样本。"""
    import torch
    import torch.nn.functional as functional

    positive_mask = class_ids[:, None].eq(class_ids[None, :]).float()
    positive_mask = positive_mask / positive_mask.sum(dim=1, keepdim=True)
    image_to_text = -(
        positive_mask * functional.log_softmax(output.logits_per_image, dim=1)
    ).sum(dim=1).mean()
    text_to_image = -(
        positive_mask * functional.log_softmax(output.logits_per_text, dim=1)
    ).sum(dim=1).mean()
    return (image_to_text + text_to_image) / 2


def evaluate(model, loader, device, use_amp, description="验证"):
    import torch
    from tqdm.auto import tqdm

    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        progress = tqdm(
            loader,
            desc=description,
            dynamic_ncols=True,
            leave=False,
        )
        for batch in progress:
            class_ids = batch.pop("class_ids").to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output = model(**batch, return_loss=False)
                loss = multi_positive_contrastive_loss(output, class_ids)
            total_loss += loss.item()
            total_batches += 1
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{total_loss / total_batches:.4f}",
            )
    return total_loss / max(total_batches, 1)


def feature_tensor(output):
    """兼容 Transformers 新旧版本的特征返回类型。"""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    pooled_output = getattr(output, "pooler_output", None)
    return pooled_output


def text_features_compat(model, text_batch):
    """规避部分版本 get_text_features 使用空池化输出的问题。"""
    try:
        features = feature_tensor(model.get_text_features(**text_batch))
        if features is not None:
            return features
    except TypeError as error:
        if "must be Tensor, not NoneType" not in str(error):
            raise

    text_output = model.text_model(**text_batch, return_dict=True)
    last_hidden_state = getattr(text_output, "last_hidden_state", None)
    if last_hidden_state is None:
        last_hidden_state = text_output[0]
    return model.text_projection(last_hidden_state[:, 0, :])


def image_features_compat(model, pixel_values):
    """兼容返回 Tensor 或 ModelOutput 的图像特征接口。"""
    features = feature_tensor(
        model.get_image_features(pixel_values=pixel_values)
    )
    if features is not None:
        return features
    vision_output = model.vision_model(
        pixel_values=pixel_values, return_dict=True
    )
    pooled_output = getattr(vision_output, "pooler_output", None)
    if pooled_output is None:
        pooled_output = vision_output.last_hidden_state[:, 0, :]
    return model.visual_projection(pooled_output)


def evaluate_classification(
    model,
    processor,
    manifest,
    data_dir,
    device,
    batch_size,
    num_workers,
    max_text_length,
):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class ClassificationDataset(Dataset):
        def __init__(self):
            with manifest.open("r", encoding="utf-8") as file:
                self.records = [json.loads(line) for line in file if line.strip()]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            record = self.records[index]
            with Image.open(data_dir / record["image"]) as image:
                image = image.convert("RGB")
            return image, int(record["class_id"])

    class_names = [
        line.strip()
        for line in (data_dir / "label_cn.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompts = [f"输电线路上的{name}" for name in class_names]
    text_batch = processor(
        text=prompts,
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    text_batch = {
        key: value.to(device) for key, value in text_batch.items() if key != "pixel_values"
    }

    model.eval()
    with torch.no_grad():
        text_features = text_features_compat(model, text_batch)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def collate(samples):
        images, labels = zip(*samples)
        image_batch = processor(images=list(images), return_tensors="pt")
        return image_batch["pixel_values"], torch.tensor(labels, dtype=torch.long)

    loader = DataLoader(
        ClassificationDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    confusion = torch.zeros(
        (len(class_names), len(class_names)), dtype=torch.long
    )
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            image_features = image_features_compat(model, pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
            predictions = similarities.argmax(dim=1)
            top_k = similarities.topk(min(5, len(class_names)), dim=1).indices
            top1_correct += (predictions == labels).sum().item()
            top5_correct += (top_k == labels[:, None]).any(dim=1).sum().item()
            total += labels.numel()
            for target, prediction in zip(labels.cpu(), predictions.cpu()):
                confusion[int(target), int(prediction)] += 1

    per_class = {}
    f1_scores = []
    for class_id, class_name in enumerate(class_names):
        true_positive = confusion[class_id, class_id].item()
        support = confusion[class_id].sum().item()
        predicted = confusion[:, class_id].sum().item()
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_scores.append(f1)
        per_class[class_name] = {
            "support": support,
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "samples": total,
        "top1_accuracy": top1_correct / max(total, 1),
        "top5_accuracy": top5_correct / max(total, 1),
        "macro_f1": sum(f1_scores) / max(len(f1_scores), 1),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    try:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        from transformers import (
            ChineseCLIPModel,
            ChineseCLIPProcessor,
            get_cosine_schedule_with_warmup,
        )
    except ImportError as exc:
        raise SystemExit(
            "请先安装依赖：pip install -U torch torchvision transformers pillow tqdm"
        ) from exc

    train_manifest = args.data_dir / "train_pairs.jsonl"
    valid_manifest = args.data_dir / "valid_pairs.jsonl"
    test_manifest = args.data_dir / "test_pairs.jsonl"
    if not all(
        path.is_file() for path in (train_manifest, valid_manifest, test_manifest)
    ):
        raise FileNotFoundError(
            "找不到 train_pairs.jsonl、valid_pairs.jsonl 或 test_pairs.jsonl"
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"

    print(f"运行设备：{device}", flush=True)
    if use_amp:
        print(f"GPU：{torch.cuda.get_device_name(device)}", flush=True)
    if args.eval_only:
        model_source = args.checkpoint or args.output_dir / "best"
        if not Path(model_source).is_dir():
            raise FileNotFoundError(f"找不到微调权重目录：{model_source}")
    else:
        model_source = args.model
    print(f"正在加载处理器：{model_source}", flush=True)
    processor = ChineseCLIPProcessor.from_pretrained(model_source)
    print(f"正在加载模型：{model_source}", flush=True)
    model = ChineseCLIPModel.from_pretrained(model_source)
    model.to(device)
    print("预训练模型加载完成", flush=True)

    if args.freeze_vision:
        for parameter in model.vision_model.parameters():
            parameter.requires_grad = False
    if args.freeze_text:
        for parameter in model.text_model.parameters():
            parameter.requires_grad = False

    def collate_fn(samples):
        images, texts, class_ids = zip(*samples)
        batch = processor(
            text=list(texts),
            images=list(images),
            padding=True,
            truncation=True,
            max_length=args.max_text_length,
            return_tensors="pt",
        )
        batch["class_ids"] = torch.tensor(class_ids, dtype=torch.long)
        return batch

    if args.eval_only:
        test_loader = DataLoader(
            PairDataset(test_manifest, args.data_dir),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_amp,
            collate_fn=collate_fn,
        )
        test_loss = evaluate(
            model,
            test_loader,
            device,
            use_amp,
            description="最终测试",
        )
        classification = evaluate_classification(
            model,
            processor,
            test_manifest,
            args.data_dir,
            device,
            args.batch_size,
            args.num_workers,
            args.max_text_length,
        )
        report = {
            "checkpoint": str(Path(model_source).resolve()),
            "test_contrastive_loss": test_loss,
            "test_classification": classification,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / "evaluation_metrics.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("\nChinese-CLIP 最终评价指标：")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"评价报告已保存：{report_path}")
        return

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset = PairDataset(train_manifest, args.data_dir)
    sampler = None
    if args.balanced_sampling:
        class_counts = Counter(
            int(record["class_id"]) for record in train_dataset.records
        )
        sample_weights = [
            1.0 / math.sqrt(class_counts[int(record["class_id"])])
            for record in train_dataset.records
        ]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        collate_fn=collate_fn,
        generator=generator,
    )
    valid_loader = DataLoader(
        PairDataset(valid_manifest, args.data_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )
    print(
        f"数据加载完成：训练 {len(train_dataset):,} 对，"
        f"验证 {len(valid_loader.dataset):,} 对，"
        f"每轮 {len(train_loader):,} 个批次",
        flush=True,
    )

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation
    )
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_valid_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"训练 Epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
            leave=True,
        )
        for step, batch in enumerate(progress, start=1):
            class_ids = batch.pop("class_ids").to(device, non_blocking=True)
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output = model(**batch, return_loss=False)
                loss = (
                    multi_positive_contrastive_loss(output, class_ids)
                    / args.gradient_accumulation
                )

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.gradient_accumulation

            should_update = (
                step % args.gradient_accumulation == 0
                or step == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            postfix = {
                "loss": f"{loss.item() * args.gradient_accumulation:.4f}",
                "avg": f"{running_loss / step:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
            if use_amp:
                postfix["显存"] = (
                    f"{torch.cuda.memory_allocated(device) / 1024**3:.1f}G"
                )
            progress.set_postfix(postfix)

        valid_loss = evaluate(
            model,
            valid_loader,
            device,
            use_amp,
            description=f"验证 Epoch {epoch}/{args.epochs}",
        )
        print(
            f"epoch={epoch} train_loss={running_loss / len(train_loader):.4f} "
            f"valid_loss={valid_loss:.4f}"
        )

        last_dir = args.output_dir / "last"
        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_dir = args.output_dir / "best"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"已保存最佳模型：{best_dir}")

    best_model = ChineseCLIPModel.from_pretrained(args.output_dir / "best")
    best_model.to(device)
    test_loader = DataLoader(
        PairDataset(test_manifest, args.data_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )
    test_loss = evaluate(
        best_model,
        test_loader,
        device,
        use_amp,
        description="最终测试",
    )
    classification = evaluate_classification(
        best_model,
        processor,
        test_manifest,
        args.data_dir,
        device,
        args.batch_size,
        args.num_workers,
        args.max_text_length,
    )
    report = {
        "best_validation_loss": best_valid_loss,
        "test_contrastive_loss": test_loss,
        "test_classification": classification,
    }
    report_path = args.output_dir / "evaluation_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nChinese-CLIP 最终评价指标：")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"评价报告已保存：{report_path}")


if __name__ == "__main__":
    main()
