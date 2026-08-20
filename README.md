# Power-Line Multimodal Inspection

Research code for a power-transmission inspection prototype combining:

- Chinese-CLIP equipment classification and bidirectional image-text retrieval;
- FAISS vector search;
- YOLOv8-L, YOLO26-L and RT-DETR-L object-detection experiments;
- a Gradio-based inspection interface with optional structured report generation.

## Repository scope

This repository contains source code, configuration files, and aggregate
evaluation results. It does **not** contain the unpublished manuscript,
inspection images, third-party datasets, trained model weights, API keys, or the
local user database. Dataset access remains subject to the terms of each upstream
source.

## Main scripts

- `train_chinese_clip.py`: fine-tune and evaluate Chinese-CLIP.
- `evaluate_chinese_clip_retrieval.py`: compare pretrained and fine-tuned retrieval.
- `run_chinese_clip_ablations.py`: run the controlled encoder/template ablations.
- `run_detection_baselines.py`: train/evaluate the three detector families.
- `run_detection_three_seeds.py`: repeat every detector with three fixed seeds.
- `run_mpcd_749_ablation.py`: isolate the 749 external conductor images.
- `train_rtdetr.py`: RT-DETR training entry point.
- `benchmark_system_latency.py`: measure local retrieval and detection latency.
- `use_model.py`: launch the integrated Gradio prototype.

## Environment

Python 3.10--3.12 and a CUDA-enabled PyTorch environment are recommended.

```bash
pip install -U torch torchvision transformers ultralytics faiss-cpu \
  gradio pillow opencv-python numpy pyyaml tqdm requests plotly wordcloud
```

## Expected local files

The ignored files must be prepared locally before training or deployment:

```text
train/images, train/labels
valid/images, valid/labels
test/images, test/labels
chinese_clip_dataset/
runs/chinese_clip/best/
runs/baselines/baseline_yolov8_l/weights/best.pt
```

## Training examples

Chinese-CLIP:

```bash
python train_chinese_clip.py --device cuda --batch-size 64 --epochs 10
```

YOLOv8-L baseline:

```bash
python run_detection_baselines.py \
  --experiments yolov8 \
  --yolo-weight ./yolov8l.pt \
  --device 0 --batch 8 --epochs 60 --imgsz 960 --workers 8 --amp
```

System latency:

```bash
python benchmark_system_latency.py --warmup 20 --repeats 100 --image-pool 50 --imgsz 960
```

The commands for controlled ablations and multi-seed experiments are documented
in `CHINESE_CLIP_ABLATIONS.md`, `THREE_SEED_EXPERIMENTS.md`, and
`MPCD_749_ABLATION.md`. De-identified aggregate outputs used by the manuscript
are stored under `results/`; they do not contain images, annotations, weights,
user records, credentials, or machine-specific checkpoint paths.

## Launch the prototype

Copy `.env.example` to a local `.env` or export the variables in your shell.
Never commit a real API key.

```bash
export DOUBAO_API_KEY="your-key"
python -u use_model.py
```

## Data and model availability

The base corpus is derived from the InsPLAD object-detection release and was
processed through an author-managed Roboflow project for class filtering,
remapping, annotation correction, and export. InsPLAD is licensed under
CC BY-NC-SA 4.0. Supplementary images retain their original source-specific
terms. Images and annotations are not redistributed here; readers should obtain
them from the cited upstream records. Trained weights are also excluded.
