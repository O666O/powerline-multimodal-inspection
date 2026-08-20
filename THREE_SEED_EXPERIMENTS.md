# 三随机种子目标检测实验

脚本 `run_detection_three_seeds.py` 使用完全相同的数据划分和训练参数，分别以
`42`、`3407`、`2026` 三个随机种子运行 RT-DETR-L、YOLOv8-L 和 YOLO26-L。

已有的 `runs/baselines/baseline_*` 是 seed 42 结果。脚本会先检查其中的
`args.yaml`，参数完全一致时自动复用，所以通常只需新增六次训练。

## AutoDL 运行命令

```bash
cd /root/autodl-tmp/a.yolov8

python run_detection_three_seeds.py \
  --experiments rtdetr yolov8 yolo26 \
  --seeds 42 3407 2026 \
  --rtdetr-weight ./rtdetr-l.pt \
  --yolo-weight ./yolov8l.pt \
  --yolo26-weight ./yolo26l.pt \
  --device 0 \
  --batch 8 \
  --epochs 60 \
  --imgsz 960 \
  --workers 8 \
  --patience 15 \
  --amp
```

建议正式训练前先检查运行计划：

```bash
python run_detection_three_seeds.py \
  --experiments rtdetr yolov8 yolo26 \
  --seeds 42 3407 2026 \
  --rtdetr-weight ./rtdetr-l.pt \
  --yolo-weight ./yolov8l.pt \
  --yolo26-weight ./yolo26l.pt \
  --device 0 --batch 8 --epochs 60 --imgsz 960 \
  --workers 8 --patience 15 --amp --dry-run
```

脚本默认自动续接包含 `weights/last.pt` 的中断任务。重新执行同一条命令即可，
已经生成 `best.pt` 的任务会直接复用。

如果 RT-DETR 出现 NaN，将命令最后的 `--amp` 改为 `--no-amp`。但此时不能与
AMP 训练结果混合计算三随机种子均值；应对该模型的三个种子全部使用
`--no-amp`，并为它指定一个新的输出目录。

## 输出文件

- `runs/three_seed_detection/three_seed_metrics.json`：完整运行与测试指标。
- `runs/three_seed_detection/three_seed_runs.csv`：每个种子的单次结果。
- `runs/three_seed_detection/three_seed_summary.csv`：均值和样本标准差。
- `runs/three_seed_detection/three_seed_per_class_summary.csv`：逐类别均值和标准差。
- `runs/three_seed_detection/three_seed_latex_rows.txt`：可粘贴进论文的表格行。

论文应报告 `mean ± standard deviation`，并说明标准差采用三个独立运行的样本
标准差（分母为 `n-1`）。测试集只用于最终评价，不用于挑选最佳轮次。
