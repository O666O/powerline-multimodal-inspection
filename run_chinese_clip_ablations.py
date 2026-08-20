"""顺序训练并汇总 Chinese-CLIP 微调消融实验。"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "vision_only": {
        "train_mode": "vision_only",
        "caption_mode": "original",
        "description": "仅微调图像编码塔",
    },
    "text_only": {
        "train_mode": "text_only",
        "caption_mode": "original",
        "description": "仅微调文本编码塔",
    },
    "single_template": {
        "train_mode": "full",
        "caption_mode": "single",
        "description": "双塔微调但仅使用一个训练文本模板",
    },
    "full": {
        "train_mode": "full",
        "caption_mode": "original",
        "description": "双塔和六种文本模板的完整方案",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="运行 Chinese-CLIP 微调消融实验")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument(
        "--model",
        default="OFA-Sys/chinese-clip-vit-base-patch16",
        help="原版模型名称或AutoDL上的本地模型目录",
    )
    parser.add_argument(
        "--processor-model",
        default=None,
        help="检索评价使用的处理器目录；默认等于 --model",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "chinese_clip_dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "chinese_clip_ablations",
    )
    parser.add_argument(
        "--full-checkpoint",
        type=Path,
        default=None,
        help="现有完整微调模型的best目录；提供后不再训练full配置",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=52)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--single-template", default="输电线路上的{label}")
    parser.add_argument("--similarity-batch-size", type=int, default=512)
    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使输出目录已有best权重也重新训练",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def show_command(command):
    print("\n$ " + subprocess.list2cmdline([str(item) for item in command]), flush=True)


def run_command(command, dry_run=False):
    show_command(command)
    if not dry_run:
        subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def checkpoint_for(name, args):
    if name == "full" and args.full_checkpoint is not None:
        checkpoint = args.full_checkpoint.expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"找不到完整方案权重目录：{checkpoint}")
        return checkpoint, None
    run_dir = (args.output_dir / f"seed_{args.seed}" / name).resolve()
    return run_dir / "best", run_dir


def train_experiment(name, args):
    spec = EXPERIMENTS[name]
    checkpoint, run_dir = checkpoint_for(name, args)
    if run_dir is None:
        print(f"\n[{name}] 复用现有完整模型：{checkpoint}", flush=True)
        return checkpoint, None
    metrics_path = run_dir / "evaluation_metrics.json"
    if checkpoint.is_dir() and metrics_path.is_file() and not args.force:
        print(f"\n[{name}] 已有完整结果，跳过训练：{run_dir}", flush=True)
        return checkpoint, run_dir

    command = [
        sys.executable,
        ROOT / "train_chinese_clip.py",
        "--data-dir",
        args.data_dir,
        "--model",
        args.model,
        "--output-dir",
        run_dir,
        "--epochs",
        args.epochs,
        "--batch-size",
        args.batch_size,
        "--gradient-accumulation",
        args.gradient_accumulation,
        "--learning-rate",
        args.learning_rate,
        "--weight-decay",
        args.weight_decay,
        "--warmup-ratio",
        args.warmup_ratio,
        "--num-workers",
        args.num_workers,
        "--max-text-length",
        args.max_text_length,
        "--seed",
        args.seed,
        "--device",
        args.device,
        "--train-mode",
        spec["train_mode"],
        "--caption-mode",
        spec["caption_mode"],
    ]
    if spec["caption_mode"] == "single":
        command.extend(["--single-caption-template", args.single_template])
    command.append(
        "--balanced-sampling" if args.balanced_sampling else "--no-balanced-sampling"
    )
    print(f"\n[{name}] {spec['description']}", flush=True)
    run_command(command, args.dry_run)
    return checkpoint, run_dir


def evaluate_retrieval(name, checkpoint, args):
    report_path = (
        args.output_dir / f"seed_{args.seed}" / name / "retrieval_metrics.json"
    ).resolve()
    baseline_cache = (
        args.output_dir / f"seed_{args.seed}" / "pretrained_retrieval_cache.json"
    ).resolve()
    command = [
        sys.executable,
        ROOT / "evaluate_chinese_clip_retrieval.py",
        "--baseline-model",
        args.model,
        "--finetuned-model",
        checkpoint,
        "--processor-model",
        args.processor_model or args.model,
        "--data-dir",
        args.data_dir,
        "--split",
        "test",
        "--batch-size",
        args.batch_size,
        "--similarity-batch-size",
        args.similarity_batch_size,
        "--device",
        args.device,
        "--output",
        report_path,
        "--baseline-cache",
        baseline_cache,
    ]
    run_command(command, args.dry_run)
    return report_path


def evaluate_baseline_classification(args):
    """原版模型不训练，只在同一测试集上补齐分类指标。"""

    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        print(
            "\n原版模型不是本地目录，跳过原版分类评价；"
            "检索评价仍会通过模型名称加载。",
            flush=True,
        )
        return None
    output_dir = (
        args.output_dir / f"seed_{args.seed}" / "pretrained"
    ).resolve()
    metrics_path = output_dir / "evaluation_metrics.json"
    if metrics_path.is_file() and not args.force:
        print(f"\n[pretrained] 已有分类评价：{metrics_path}", flush=True)
        return metrics_path
    command = [
        sys.executable,
        ROOT / "train_chinese_clip.py",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        output_dir,
        "--batch-size",
        args.batch_size,
        "--num-workers",
        args.num_workers,
        "--max-text-length",
        args.max_text_length,
        "--device",
        args.device,
        "--eval-only",
        "--checkpoint",
        model_path.resolve(),
    ]
    print("\n[pretrained] 在同一测试集上评价原版模型", flush=True)
    run_command(command, args.dry_run)
    return None if args.dry_run else metrics_path


def read_classification(run_dir, fallback_checkpoint):
    candidates = []
    if run_dir is not None:
        candidates.append(run_dir / "evaluation_metrics.json")
    candidates.extend(
        [
            fallback_checkpoint.parent / "evaluation_metrics.json",
            fallback_checkpoint / "evaluation_metrics.json",
        ]
    )
    for path in candidates:
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            return report.get("test_classification", {})
    return {}


def build_summary(completed, baseline_classification_path, args):
    rows = []
    metric_names = (
        "instance_R@1",
        "instance_R@5",
        "label_R@1",
        "label_R@5",
        "proto_label_R@1",
        "proto_label_R@5",
    )
    first_retrieval = json.loads(completed[0][3].read_text(encoding="utf-8"))
    baseline_classification = {}
    if baseline_classification_path and baseline_classification_path.is_file():
        baseline_report = json.loads(
            baseline_classification_path.read_text(encoding="utf-8")
        )
        baseline_classification = baseline_report.get("test_classification", {})
    baseline_summary = first_retrieval["baseline"]["paper_compatible_summary"]
    baseline_row = {
        "experiment": "pretrained",
        "description": "未进行领域微调的原版Chinese-CLIP",
        "seed": args.seed,
        "checkpoint": str(args.model),
        "top1_accuracy": baseline_classification.get("top1_accuracy"),
        "top5_accuracy": baseline_classification.get("top5_accuracy"),
        "macro_f1": baseline_classification.get("macro_f1"),
    }
    baseline_row.update({key: baseline_summary[key] for key in metric_names})
    rows.append(baseline_row)

    for name, checkpoint, run_dir, retrieval_path in completed:
        classification = read_classification(run_dir, checkpoint)
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        summary = retrieval["finetuned"]["paper_compatible_summary"]
        row = {
            "experiment": name,
            "description": EXPERIMENTS[name]["description"],
            "seed": args.seed,
            "checkpoint": str(checkpoint),
            "top1_accuracy": classification.get("top1_accuracy"),
            "top5_accuracy": classification.get("top5_accuracy"),
            "macro_f1": classification.get("macro_f1"),
        }
        row.update({key: summary[key] for key in metric_names})
        rows.append(row)

    summary_dir = (args.output_dir / f"seed_{args.seed}").resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "ablation_summary.json"
    csv_path = summary_dir / "ablation_summary.csv"
    json_path.write_text(
        json.dumps({"seed": args.seed, "results": rows}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n消融实验汇总（百分数）：")
    print(
        f"{'实验':<18}{'Top-1':>9}{'Macro-F1':>11}"
        f"{'Label R@1':>12}{'Label R@5':>12}{'Proto R@1':>12}"
    )
    for row in rows:
        def percent(key):
            value = row.get(key)
            return "--" if value is None else f"{value * 100:.2f}"

        print(
            f"{row['experiment']:<18}{percent('top1_accuracy'):>9}"
            f"{percent('macro_f1'):>11}{percent('label_R@1'):>12}"
            f"{percent('label_R@5'):>12}{percent('proto_label_R@1'):>12}"
        )
    print(f"\nJSON汇总：{json_path}")
    print(f"CSV汇总：{csv_path}")


def main():
    args = parse_args()
    if "{label}" not in args.single_template:
        raise ValueError("--single-template 必须包含 {label}")
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    baseline_classification_path = evaluate_baseline_classification(args)
    completed = []
    for name in args.experiments:
        checkpoint, run_dir = train_experiment(name, args)
        if args.dry_run:
            continue
        retrieval_path = evaluate_retrieval(name, checkpoint, args)
        completed.append((name, checkpoint, run_dir, retrieval_path))

    if completed:
        build_summary(completed, baseline_classification_path, args)
    elif args.dry_run:
        print("\n以上为预演命令，未启动训练。")


if __name__ == "__main__":
    main()
