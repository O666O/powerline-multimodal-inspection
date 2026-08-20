# Chinese-CLIP 微调消融实验

该实验固定数据划分、训练轮数、优化器和评测方式，仅改变可训练编码塔或训练文本模板。

| 配置 | 可训练部分 | 训练文本 |
| --- | --- | --- |
| `vision_only` | 图像编码器和图像投影层 | 原始六模板 |
| `text_only` | 文本编码器和文本投影层 | 原始六模板 |
| `single_template` | 图像塔和文本塔 | 单一模板 |
| `full` | 图像塔和文本塔 | 原始六模板 |

`vision_only` 和 `text_only` 均保留可学习的温度参数。验证集和测试集始终使用原始文本，单模板只作用于训练集，保证各配置在同一测试分布上比较。

## AutoDL运行

如果原版模型位于 `/root/autodl-tmp/models/chinese-clip`，已有完整微调权重位于 `runs/chinese_clip/best`：

```bash
cd /root/autodl-tmp/a.yolov8

python -u run_chinese_clip_ablations.py \
  --model /root/autodl-tmp/models/chinese-clip \
  --processor-model /root/autodl-tmp/models/chinese-clip \
  --full-checkpoint runs/chinese_clip/best \
  --device cuda \
  --batch-size 64 \
  --epochs 10 \
  --num-workers 4 \
  --seed 42
```

已有完整模型会被直接复用，因此实际新增训练三次。原版模型只进行一次测试，不参与训练，其检索特征结果会缓存并供全部配置复用。程序断开后可以再次执行同一命令，已有完整结果的配置会被跳过。

如显存不足，将 `--batch-size 64` 改成 `--batch-size 32 --gradient-accumulation 2`。

## 输出

- 每个配置的最佳权重：`runs/chinese_clip_ablations/seed_42/<配置>/best/`
- 分类指标：对应配置目录中的 `evaluation_metrics.json`
- 检索指标：对应配置目录中的 `retrieval_metrics.json`
- 论文汇总表：`ablation_summary.csv` 和 `ablation_summary.json`

训练前可追加 `--dry-run`，只检查即将执行的命令而不启动训练。
