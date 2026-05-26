#!/usr/bin/env python3
"""Class-wise OOD statistics: in-domain (MCD/KTH) vs cross-domain (KITTI-360).

Each point's Mahalanobis OOD score is grouped by the model's PREDICTED class
(argmax of the semantic head). For every class we report how the score
distribution differs between the two domains -- i.e. which classes the model
recognizes confidently vs. which ones cross-domain points get forced into while
sitting far from the ID cluster.

Pools several scans per group (per-class stats need samples), prints a table,
and saves a figure:

  <out>/class_stats.png
    (top)    per-class mean OOD score, in-domain vs cross-domain (grouped bars)
    (bottom) per-class share of points (how the predicted-class mix shifts)

Example:
    python scripts/ood_class_stats.py --n 15
"""

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ce_net import CONFIG_DIR

from ood_common import STATS_FILENAME, load_yaml
from ood_test_scans import build_detector_and_groups, score_scan

GROUP_LABELS = {"mcd_kth": "in-domain (MCD/KTH)", "kitti360": "cross-domain (KITTI-360)"}


def class_name_lookup(model_dir):
    """Map a learning-map (xentropy) index -> human class name."""
    DATA = load_yaml(os.path.join(model_dir, "data_cfg.yaml"))
    labels = DATA["labels"]
    inv = DATA["learning_map_inv"]
    return lambda idx: labels.get(inv.get(idx, idx), str(idx))


def collect_group(model, hook, stats, ds, count, device):
    """Pool per-point scores and predicted classes across `count` scans."""
    scores, preds = [], []
    for i in range(count):
        _, s, p = score_scan(model, hook, stats, ds[i], device, return_pred=True)
        scores.append(s)
        preds.append(p)
        print(f"    scan {i}: {len(s)} pts")
    return np.concatenate(scores), np.concatenate(preds)


def per_class_table(scores, preds, nclasses):
    """Return dict class_idx -> {n, mean, median, p90} over points of that class."""
    out = {}
    for c in range(nclasses):
        m = preds == c
        n = int(m.sum())
        if n == 0:
            continue
        s = scores[m]
        out[c] = dict(n=n, mean=float(s.mean()), median=float(np.median(s)),
                      p90=float(np.percentile(s, 90)))
    return out


def main():
    base_cfg = load_yaml(CONFIG_DIR / "training_mcd.yaml")
    infer_cfg = load_yaml(CONFIG_DIR / "inference_mcd.yaml")
    default_model = infer_cfg.get("inference", {}).get("model_path") or os.path.join(
        base_cfg["training"]["model_path"], base_cfg["training"]["model_name"]
    )

    ap = argparse.ArgumentParser("./ood_class_stats.py")
    ap.add_argument("--model", "-m", default=default_model)
    ap.add_argument("--datasets_root",
                    default="/media/donceykong/doncey_ssd_021/datasets")
    ap.add_argument("--mcd_seq", default="kth_day_06")
    ap.add_argument("--kitti_seq", default="2013_05_28_drive_0000_sync")
    ap.add_argument("--kitti_fov_up", type=float, default=3.0)
    ap.add_argument("--kitti_fov_down", type=float, default=-25.0)
    ap.add_argument("--n", type=int, default=15, help="Scans per group to pool.")
    ap.add_argument("--min_points", type=int, default=200,
                    help="Drop classes with fewer than this many points (either group).")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--out", default="ood_test_out")
    FLAGS, _ = ap.parse_known_args()

    os.makedirs(FLAGS.out, exist_ok=True)
    name_of = class_name_lookup(FLAGS.model)

    model, hook, stats, groups, device = build_detector_and_groups(
        FLAGS.model, FLAGS.datasets_root, FLAGS.mcd_seq, FLAGS.kitti_seq, FLAGS.n,
        kitti_fov_up=FLAGS.kitti_fov_up, kitti_fov_down=FLAGS.kitti_fov_down,
        stats_path=FLAGS.stats,
    )
    nclasses = int(stats["nclasses"])

    pooled = {}
    for gname, (ds, count) in groups.items():
        print(f"\nCollecting {GROUP_LABELS.get(gname, gname)} ({count} scans)...")
        sc, pr = collect_group(model, hook, stats, ds, count, device)
        pooled[gname] = (sc, pr)
    hook.remove()

    tbl = {g: per_class_table(sc, pr, nclasses) for g, (sc, pr) in pooled.items()}
    totals = {g: len(sc) for g, (sc, _) in pooled.items()}

    # Classes present (>= min_points) in at least one group.
    keep = sorted(
        c for c in range(nclasses)
        if max(tbl["mcd_kth"].get(c, {}).get("n", 0),
                tbl["kitti360"].get(c, {}).get("n", 0)) >= FLAGS.min_points
    )

    # --- text table ---
    print("\n" + "=" * 92)
    print(f"{'class':<18}{'n_id':>8}{'mean_id':>9}{'n_ood':>9}{'mean_ood':>10}"
          f"{'Δmean':>9}{'p90_ood':>9}")
    print("-" * 92)
    for c in keep:
        idc = tbl["mcd_kth"].get(c, {})
        ood = tbl["kitti360"].get(c, {})
        mean_id = idc.get("mean", float("nan"))
        mean_ood = ood.get("mean", float("nan"))
        dmean = (mean_ood - mean_id) if (idc and ood) else float("nan")
        print(f"{name_of(c):<18}{idc.get('n', 0):>8}{mean_id:>9.2f}"
              f"{ood.get('n', 0):>9}{mean_ood:>10.2f}{dmean:>9.2f}"
              f"{ood.get('p90', float('nan')):>9.2f}")
    print("=" * 92)

    # --- figure: mean OOD per class + class share ---
    names = [name_of(c) for c in keep]
    id_mean = [tbl["mcd_kth"].get(c, {}).get("mean", 0.0) for c in keep]
    ood_mean = [tbl["kitti360"].get(c, {}).get("mean", 0.0) for c in keep]
    id_share = [100.0 * tbl["mcd_kth"].get(c, {}).get("n", 0) / max(totals["mcd_kth"], 1) for c in keep]
    ood_share = [100.0 * tbl["kitti360"].get(c, {}).get("n", 0) / max(totals["kitti360"], 1) for c in keep]

    y = np.arange(len(keep))
    h = 0.4
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, max(6, 0.42 * len(keep) + 3)))

    ax1.barh(y - h / 2, id_mean, height=h, label=GROUP_LABELS["mcd_kth"], color="#2c7fb8")
    ax1.barh(y + h / 2, ood_mean, height=h, label=GROUP_LABELS["kitti360"], color="#d95f02")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()
    ax1.set_xlabel("mean Mahalanobis OOD score")
    ax1.set_title("Per-predicted-class mean OOD score")
    ax1.legend()
    ax1.grid(axis="x", alpha=0.3)

    ax2.barh(y - h / 2, id_share, height=h, label=GROUP_LABELS["mcd_kth"], color="#2c7fb8")
    ax2.barh(y + h / 2, ood_share, height=h, label=GROUP_LABELS["kitti360"], color="#d95f02")
    ax2.set_yticks(y)
    ax2.set_yticklabels(names)
    ax2.invert_yaxis()
    ax2.set_xlabel("share of points predicted as this class (%)")
    ax2.set_title("Predicted-class distribution shift")
    ax2.legend()
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("CENet class-wise OOD: in-domain vs cross-domain", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(FLAGS.out, "class_stats.png")
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved figure -> {out_path}")

    # Quick global summary.
    for g, (sc, _) in pooled.items():
        print(f"  {GROUP_LABELS.get(g, g):28s} overall mean OOD = {sc.mean():.2f}  "
              f"median = {np.median(sc):.2f}  (n={len(sc)})")


if __name__ == "__main__":
    main()
