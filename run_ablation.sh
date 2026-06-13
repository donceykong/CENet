
# Sweep evidence activation (unc_act) x loss form (unc_type) x class weighting,
# holding the KL warmup/strength pinned. 2 acts x 2 types x 2 class_weight = 8
# cells. Weighted cells get a "_cw" suffix in the cell / W&B run name.
python scripts/train_ablation.py \
  --epochs 60 --perc-scans-to-use 0.99 --keyframe-dist 1.0 \
  --unc-acts relu \
  --unc-types mse \
  --warmups W10 \
  --kl-strengths 0.01 \
  --class-weights true
