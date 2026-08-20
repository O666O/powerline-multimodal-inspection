# Aggregate experimental results

These files are de-identified aggregate outputs used in the journal manuscript.
They contain no inspection images, annotations, trained weights, credentials,
user records, or machine-specific checkpoint paths.

- `detector_three_seed_summary.csv`: overall detector means and sample standard deviations.
- `detector_three_seed_per_class.csv`: class-wise means and sample standard deviations.
- `chinese_clip_ablation_seed42.csv`: controlled encoder/template ablation.
- `mpcd_749_ablation_summary.csv`: with/without external-data comparison.
- `mpcd_749_ablation_paired_deltas.csv`: paired with-minus-without changes by seed.
- `system_latency.csv`: 100-run local latency benchmark after 20 warm-up runs.

Detector multi-seed statistics use seeds 42, 3407 and 2026. Standard deviations
are sample standard deviations with denominator `n-1`.
