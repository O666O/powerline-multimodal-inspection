# External broken-conductor data ablation

`run_mpcd_749_ablation.py` compares YOLOv8-L with and without the 749 active
MPCD broken-conductor training images. It never deletes or moves dataset files.
Instead, it creates two audited training lists whose set difference is exactly
the external subset. Validation/test splits, model initialization, optimizer,
augmentation, early stopping and evaluation settings remain fixed.

The default seeds are `42`, `3407` and `2026`. Compatible full-training-set
weights from the three-seed detector experiment are reused after protocol
checks, so a normal run trains only the three models without MPCD images.

```bash
python run_mpcd_749_ablation.py --dry-run

python run_mpcd_749_ablation.py \
  --model ./yolov8l.pt \
  --device 0 --batch 8 --epochs 60 --imgsz 960 \
  --workers 8 --patience 15 --amp
```

When the private import manifest is unavailable, the script can use the
dedicated `mpcd_train_` filename prefix. This fallback is accepted only if
there are exactly 749 current training images, every matching label exists,
and every annotation belongs exclusively to the broken-conductor class.

Public aggregate results are available in `results/mpcd_749_ablation_summary.csv`
and `results/mpcd_749_ablation_paired_deltas.csv`.
