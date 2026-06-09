
# Sweep evidence activation (unc_act) x loss form (unc_type), holding the KL
# warmup/strength pinned so the grid stays small. 3 acts x 3 types = 9 cells.
python scripts/train_ablation.py \
  --epochs 60 --perc-scans-to-use 0.99 --keyframe-dist 1.0 \
  --unc-acts relu softplus \
  --unc-types mse log \
  --warmups W100 --kl-strengths 0.01
