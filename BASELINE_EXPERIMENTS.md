# RT-DETR baseline experiments

The baseline runner trains all selected detectors with the same 14 classes,
image size, optimizer settings, validation split and held-out test split.  It
uses every image in `train/images` once, rather than the repeated entries in
`train_quality_balanced.txt`.

## Install

```bash
pip install -U ultralytics pyyaml
```

Place `rtdetr-l.pt` and `yolov8l.pt` in the project directory when the server
cannot access GitHub.

## Recommended command

```bash
cd /root/autodl-tmp/a.yolov8
python run_detection_baselines.py \
  --experiments rtdetr yolov8 \
  --rtdetr-weight ./rtdetr-l.pt \
  --yolo-weight ./yolov8l.pt \
  --device 0 \
  --batch 8 \
  --epochs 60 \
  --imgsz 960 \
  --workers 8 \
  --patience 15 \
  --amp
```

To run only the mandatory RT-DETR baseline:

```bash
python run_detection_baselines.py \
  --experiments rtdetr \
  --rtdetr-weight ./rtdetr-l.pt \
  --device 0 --batch 8 --epochs 60 --imgsz 960 --workers 8 --amp
```

If mixed-precision training produces NaN, replace `--amp` with `--no-amp`.
Existing completed weights are reused automatically. Use `--overwrite` only
when a baseline must be trained again from scratch.

## Outputs

- `runs/baselines/baseline_metrics.json`: overall and per-class validation/test metrics.
- `runs/baselines/baseline_summary.csv`: paper-ready overall comparison rows.
- `runs/baselines/baseline_rtdetr_l/`: RT-DETR-L run.
- `runs/baselines/baseline_yolov8_l/`: YOLOv8-L run.

Do not copy new images into `valid` or `test` between runs. Model selection is
performed on validation results; the test split is only used for the final
comparison.
