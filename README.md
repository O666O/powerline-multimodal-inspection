# Power-Line Multimodal Inspection

Research code for a power-transmission inspection prototype combining:

- Chinese-CLIP equipment classification and bidirectional image-text retrieval;
- FAISS vector search;
- YOLOv8-L and RT-DETR-L object-detection experiments;
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
- `run_detection_baselines.py`: train/evaluate YOLOv8-L and RT-DETR-L baselines.
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

## Launch the prototype

Copy `.env.example` to a local `.env` or export the variables in your shell.
Never commit a real API key.

```bash
export DOUBAO_API_KEY="your-key"
python -u use_model.py
```

## Data and model availability

The base corpus was consolidated in an author-managed private Roboflow project
from multiple upstream workspaces. Roboflow served as aggregation, annotation,
and export infrastructure. The authors do not claim ownership of all underlying
images, and the images are not redistributed here. Trained weights are also
excluded pending confirmation of the applicable source and model licenses.
