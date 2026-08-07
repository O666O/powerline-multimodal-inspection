"""对比原版与微调 Chinese-CLIP 的图文检索指标。

输出指标：
1. Instance R@K：检索结果中是否包含与查询严格配对的图/文本。
2. label_R@K：检索结果中是否包含与查询相同类别的图/文本。
3. proto_label_R@K：类别原型文本检索的前 K 张图片中，是否包含该类别。
4. image_to_prototype：图片在类别原型中的 Top-K 分类准确率（补充指标）。
"""

import argparse
import gc
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOTYPE_TEMPLATES = [
    "输电线路上的{label}",
    "无人机巡检图像中的{label}",
    "高压输电线路设备：{label}",
    "电力巡检发现的{label}",
    "航拍画面中的{label}",
    "输电线路局部区域的{label}",
]


def parse_args():
    parser = argparse.ArgumentParser(description="评估 Chinese-CLIP 检索指标")
    parser.add_argument(
        "--baseline-model",
        required=True,
        help="原版模型的本地目录或 Hugging Face 名称",
    )
    parser.add_argument(
        "--finetuned-model",
        required=True,
        help="微调后 best 模型目录",
    )
    parser.add_argument(
        "--processor-model",
        default=None,
        help="处理器目录；默认使用 baseline-model",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "chinese_clip_dataset",
    )
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--similarity-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "retrieval_metrics.json")
    return parser.parse_args()


def load_records(data_dir, split):
    manifest = data_dir / f"{split}_pairs.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"找不到图文对清单：{manifest}")
    with manifest.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    return records


def get_feature_tensor(output):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    for name in ("text_embeds", "image_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"无法从模型输出中取得特征：{type(output)}")


def get_text_features_compat(model, text_inputs):
    """兼容部分 Transformers 版本中 ChineseCLIP pooler_output=None 的问题。"""
    try:
        return model.get_text_features(**text_inputs)
    except TypeError as error:
        if "must be Tensor, not NoneType" not in str(error):
            raise

    output = model.text_model(**text_inputs, return_dict=True)
    hidden = output.last_hidden_state
    return model.text_projection(hidden[:, 0, :])


def get_image_features_compat(model, pixel_values):
    output = model.get_image_features(pixel_values=pixel_values)
    try:
        return get_feature_tensor(output)
    except TypeError:
        vision_output = model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        return model.visual_projection(vision_output.last_hidden_state[:, 0, :])


def extract_image_features(
    model, processor, records, data_dir, device, batch_size, description
):
    import torch
    from PIL import Image
    from tqdm.auto import tqdm

    features = []
    model.eval()
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(records), batch_size),
            desc=f"{description}：提取图片特征",
            dynamic_ncols=True,
        ):
            images = []
            for record in records[start : start + batch_size]:
                with Image.open(data_dir / record["image"]) as image:
                    images.append(image.convert("RGB"))
            batch = processor(images=images, return_tensors="pt")
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            feature = get_image_features_compat(model, pixel_values)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature.float().cpu())
    return torch.cat(features)


def extract_text_features(
    model, processor, texts, device, batch_size, description
):
    import torch
    from tqdm.auto import tqdm

    features = []
    model.eval()
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc=f"{description}：提取文本特征",
            dynamic_ncols=True,
        ):
            batch = processor(
                text=texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=52,
                return_tensors="pt",
            )
            text_inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if key != "pixel_values"
            }
            output = get_text_features_compat(model, text_inputs)
            feature = get_feature_tensor(output)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature.float().cpu())
    return torch.cat(features)


def compute_direction(
    query_features,
    gallery_features,
    query_labels,
    gallery_labels,
    device,
    chunk_size,
):
    import torch
    from tqdm.auto import tqdm

    maximum_k = min(5, len(gallery_features))
    instance_hits = {1: 0, 5: 0}
    label_hits = {1: 0, 5: 0}
    gallery_on_device = gallery_features.to(device)
    gallery_labels = gallery_labels.to(device)

    with torch.inference_mode():
        for start in tqdm(
            range(0, len(query_features), chunk_size),
            desc="计算检索排名",
            dynamic_ncols=True,
            leave=False,
        ):
            end = min(start + chunk_size, len(query_features))
            query = query_features[start:end].to(device)
            similarities = query @ gallery_on_device.T
            indices = similarities.topk(maximum_k, dim=1).indices
            exact_targets = torch.arange(start, end, device=device)
            labels = query_labels[start:end].to(device)

            for k in (1, 5):
                effective_k = min(k, maximum_k)
                topk = indices[:, :effective_k]
                instance_hits[k] += (
                    topk == exact_targets[:, None]
                ).any(dim=1).sum().item()
                label_hits[k] += (
                    gallery_labels[topk] == labels[:, None]
                ).any(dim=1).sum().item()

    count = len(query_features)
    return {
        "instance_R@1": instance_hits[1] / count,
        "instance_R@5": instance_hits[5] / count,
        "label_R@1": label_hits[1] / count,
        "label_R@5": label_hits[5] / count,
    }


def compute_prototype_metrics(
    model,
    processor,
    image_features,
    image_labels,
    class_names,
    device,
    batch_size,
    description,
):
    import torch

    prototype_features = []
    for class_name in class_names:
        prompts = [
            template.format(label=class_name)
            for template in PROTOTYPE_TEMPLATES
        ]
        prompt_features = extract_text_features(
            model,
            processor,
            prompts,
            device,
            batch_size,
            f"{description}：原型 {class_name}",
        )
        prototype = prompt_features.mean(dim=0)
        prototype = prototype / prototype.norm()
        prototype_features.append(prototype)
    prototype_features = torch.stack(prototype_features)

    images_on_device = image_features.to(device)
    prototypes_on_device = prototype_features.to(device)
    labels_on_device = image_labels.to(device)
    similarities = prototypes_on_device @ images_on_device.T

    prototype_results = {}
    for k in (1, 5):
        effective_k = min(k, len(image_features))
        topk_images = similarities.topk(effective_k, dim=1).indices
        expected = torch.arange(len(class_names), device=device)[:, None]
        success = (
            labels_on_device[topk_images] == expected
        ).any(dim=1).float()
        prototype_results[f"proto_label_R@{k}"] = success.mean().item()

    image_to_prototype = images_on_device @ prototypes_on_device.T
    classification_results = {}
    for k in (1, 5):
        effective_k = min(k, len(class_names))
        predictions = image_to_prototype.topk(effective_k, dim=1).indices
        success = (
            predictions == labels_on_device[:, None]
        ).any(dim=1).float()
        classification_results[f"top{k}_accuracy"] = success.mean().item()

    return prototype_results, classification_results


def average_directions(image_to_text, text_to_image):
    return {
        key: (image_to_text[key] + text_to_image[key]) / 2
        for key in image_to_text
    }


def evaluate_model(
    model_path,
    processor,
    records,
    data_dir,
    class_names,
    device,
    batch_size,
    similarity_batch_size,
    description,
):
    import torch
    from transformers import ChineseCLIPModel

    print(f"\n正在加载{description}：{model_path}", flush=True)
    model = ChineseCLIPModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    texts = [record["text"] for record in records]
    labels = torch.tensor(
        [int(record["class_id"]) for record in records],
        dtype=torch.long,
    )
    image_features = extract_image_features(
        model,
        processor,
        records,
        data_dir,
        device,
        batch_size,
        description,
    )
    text_features = extract_text_features(
        model,
        processor,
        texts,
        device,
        batch_size,
        description,
    )

    print(f"{description}：计算图片到文本检索指标", flush=True)
    image_to_text = compute_direction(
        image_features,
        text_features,
        labels,
        labels,
        device,
        similarity_batch_size,
    )
    print(f"{description}：计算文本到图片检索指标", flush=True)
    text_to_image = compute_direction(
        text_features,
        image_features,
        labels,
        labels,
        device,
        similarity_batch_size,
    )
    bidirectional_mean = average_directions(image_to_text, text_to_image)
    prototype, image_to_prototype = compute_prototype_metrics(
        model,
        processor,
        image_features,
        labels,
        class_names,
        device,
        batch_size,
        description,
    )

    result = {
        "model": str(model_path),
        "samples": len(records),
        "image_to_text": image_to_text,
        "text_to_image": text_to_image,
        "bidirectional_mean": bidirectional_mean,
        "paper_compatible_summary": {
            **bidirectional_mean,
            **prototype,
        },
        "supplementary_image_to_prototype": image_to_prototype,
    }

    del model, image_features, text_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def add_improvement(baseline, finetuned):
    baseline_metrics = baseline["paper_compatible_summary"]
    finetuned_metrics = finetuned["paper_compatible_summary"]
    return {
        key: {
            "baseline": baseline_metrics[key],
            "finetuned": finetuned_metrics[key],
            "absolute_gain": finetuned_metrics[key] - baseline_metrics[key],
            "percentage_point_gain": (
                finetuned_metrics[key] - baseline_metrics[key]
            )
            * 100,
        }
        for key in baseline_metrics
    }


def main():
    args = parse_args()
    try:
        import torch
        from transformers import ChineseCLIPProcessor
    except ImportError as exc:
        raise SystemExit(
            "请先安装依赖：pip install -U torch transformers pillow tqdm"
        ) from exc

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"运行设备：{device}", flush=True)
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}", flush=True)

    records = load_records(args.data_dir, args.split)
    class_names = [
        line.strip()
        for line in (args.data_dir / "label_cn.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    processor_path = args.processor_model or args.baseline_model
    print(f"正在加载处理器：{processor_path}", flush=True)
    processor = ChineseCLIPProcessor.from_pretrained(processor_path)
    print(
        f"测试划分：{args.split}，图文对：{len(records):,}，"
        f"类别：{len(class_names)}",
        flush=True,
    )

    baseline = evaluate_model(
        args.baseline_model,
        processor,
        records,
        args.data_dir,
        class_names,
        device,
        args.batch_size,
        args.similarity_batch_size,
        "原版模型",
    )
    finetuned = evaluate_model(
        args.finetuned_model,
        processor,
        records,
        args.data_dir,
        class_names,
        device,
        args.batch_size,
        args.similarity_batch_size,
        "微调模型",
    )
    report = {
        "split": args.split,
        "metric_definition": {
            "paper_compatible_summary": "图到文与文到图 Recall 的算术平均",
            "instance_R@K": "前K项是否包含严格配对实例",
            "label_R@K": "前K项是否包含相同类别",
            "proto_label_R@K": "类别原型文本检索前K张图片是否命中该类别",
        },
        "baseline": baseline,
        "finetuned": finetuned,
        "comparison": add_improvement(baseline, finetuned),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n论文兼容指标对比：")
    print(
        json.dumps(
            {
                "baseline": baseline["paper_compatible_summary"],
                "finetuned": finetuned["paper_compatible_summary"],
                "comparison": report["comparison"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n完整报告已保存：{args.output}")


if __name__ == "__main__":
    main()
