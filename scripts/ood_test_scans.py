#!/usr/bin/env python3
"""Cross-dataset OOD spot-check for the MCD-trained CENet detector.

Scores a few scans from two domains and compares their Mahalanobis OOD score
distributions:

  * MCD / KTH         -- in-domain (the model trained on KTH+TUHH+NTU, OS1-64)
  * KITTI-360 seq 0000 -- shifted domain (Velodyne HDL-64E, different city)

If calibration is meaningful, KITTI-360 scores should skew clearly higher than
MCD/KTH. The model and statistics live in MCD's 30-class / 128-D feature space;
the *class count is irrelevant* for OOD (features are always 128-D and we score
against MCD's class centroids), so we build the model with MCD's nclasses and
just project each dataset with its own sensor via the existing dataset classes.

Outputs (under --out, default ./ood_test_out):
  * score_histogram.png         -- overlaid score distributions
  * <group>_<name>.ply          -- first scan of each group, colored by score
and prints per-group summary statistics.

Example:
    python scripts/ood_test_scans.py --n 5
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ce_net import CONFIG_DIR
from ce_net.core.parsers.kitti360 import KITTI_360
from ce_net.core.parsers.mcd import MCD
from ce_net.utils.sensor import load_sensor

from ood_common import (
    STATS_FILENAME,
    build_model,
    load_stats,
    load_yaml,
    mahalanobis_scores,
    num_classes,
    PenultimateHook,
)

# Indices into the dataset __getitem__ tuple (shared by MCD and KITTI_360).
I_PROJ, I_PX, I_PY, I_UNPROJ_XYZ, I_NPTS = 0, 6, 7, 11, 14


def build_detector_and_groups(model_dir, datasets_root, mcd_seq, kitti_seq, n,
                              kitti_fov_up=3.0, kitti_fov_down=-25.0,
                              stats_path=None, device=None):
    """Build the MCD detector and the two test dataset groups.

    Shared by ood_test_scans.py (batch stats/export) and ood_view.py (PyVista
    viewer). The model + statistics always live in MCD's 30-class / 128-D
    feature space; each dataset is projected with its own sensor geometry.

    Returns (model, hook, stats, groups, device) where
        groups = {name: (dataset, count)}.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ARCH = load_yaml(os.path.join(model_dir, "arch_cfg.yaml"))
    DATA = load_yaml(os.path.join(model_dir, "data_cfg.yaml"))
    model_cfg = load_yaml(os.path.join(model_dir, "model_config.yaml"))
    iw, ih = model_cfg["img_width"], model_cfg["img_height"]
    nclasses = num_classes(DATA)
    max_points = ARCH["dataset"]["max_points"]

    print(f"Device: {device}  range_image={ih}x{iw}  nclasses={nclasses}")
    model = build_model(ARCH, nclasses, model_dir, device)
    stats = load_stats(stats_path or os.path.join(model_dir, STATS_FILENAME), device)
    hook = PenultimateHook(model)

    maps = dict(
        labels=DATA["labels"], color_map=DATA["color_map"],
        learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
    )

    # MCD / KTH (OS1-64), in-domain.
    os1_64 = load_sensor("OS1_64", iw, ih)
    mcd_root = os.path.join(datasets_root, "mcd")
    mcd_scans = sorted(glob.glob(
        os.path.join(mcd_root, mcd_seq, "lidar_bin", "data", "*.bin")
    ))[:n]
    if not mcd_scans:
        raise FileNotFoundError(f"No MCD scans under {mcd_root}/{mcd_seq}/lidar_bin/data")
    mcd_ds = MCD(root=mcd_root, sensor=os1_64, max_points=max_points,
                 scan_files=mcd_scans, label_files=list(mcd_scans),
                 gt=False, transform=False, **maps)

    # KITTI-360 (Velodyne HDL-64E), shifted domain. No sensor yaml exists; use
    # the (tunable) standard SemanticKITTI HDL-64E vertical FOV.
    hdl64 = {"name": "velodyne", "type": "spherical",
             "fov_up": kitti_fov_up, "fov_down": kitti_fov_down,
             "img_prop": {"width": iw, "height": ih}}
    print(f"KITTI-360 projection: {ih}x{iw}, fov_up={kitti_fov_up}, fov_down={kitti_fov_down}")
    kitti_root = os.path.join(datasets_root, "kitti360")
    kitti_ds = KITTI_360(root=kitti_root, sequences=[kitti_seq], sensor=hdl64,
                         max_points=max_points, gt=False, transform=False, **maps)

    groups = {
        "mcd_kth": (mcd_ds, min(n, len(mcd_ds))),
        "kitti360": (kitti_ds, min(n, len(kitti_ds))),
    }
    return model, hook, stats, groups, device


def score_scan(model, hook, stats, sample, device, return_pred=False):
    """Score one projected scan.

    Returns (xyz[N,3], ood_scores[N]); with return_pred=True also returns the
    model's predicted class per point (learning-map index) from the same
    forward pass: (xyz, ood_scores, pred[N]).
    """
    proj = sample[I_PROJ].unsqueeze(0).to(device)   # [1, 5, H, W]
    npoints = int(sample[I_NPTS])
    p_x = sample[I_PX][:npoints].to(device)
    p_y = sample[I_PY][:npoints].to(device)
    xyz = sample[I_UNPROJ_XYZ][:npoints].cpu().numpy()

    with torch.no_grad():
        out = model(proj)  # also populates the hook (no autocast -> fp32 features)
        ood_map = mahalanobis_scores(
            hook.features, stats["means"], stats["inv_covariance"], stats["class_ids"]
        )  # [1, H, W]
    scores = ood_map[0][p_y, p_x].cpu().numpy().astype(np.float32)
    if not return_pred:
        return xyz, scores

    logits = out[0] if isinstance(out, (list, tuple)) else out  # [1, C, H, W]
    pred_map = logits[0].argmax(dim=0)                           # [H, W]
    pred = pred_map[p_y, p_x].cpu().numpy().astype(np.int32)
    return xyz, scores, pred


def summarize(name, scores):
    print(
        f"  {name:16s} n={len(scores):>9d}  mean={scores.mean():7.2f}  "
        f"median={np.median(scores):7.2f}  p90={np.percentile(scores, 90):7.2f}  "
        f"p95={np.percentile(scores, 95):7.2f}  max={scores.max():7.2f}"
    )


def write_ply(path, xyz, scores, lo, hi):
    """ASCII PLY colored by score with a shared [lo, hi] turbo colormap."""
    norm = np.clip((scores - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = (cm.get_cmap("turbo")(norm)[:, :3] * 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


def main():
    base_cfg = load_yaml(CONFIG_DIR / "training_mcd.yaml")
    infer_cfg = load_yaml(CONFIG_DIR / "inference_mcd.yaml")
    default_model = infer_cfg.get("inference", {}).get("model_path") or os.path.join(
        base_cfg["training"]["model_path"], base_cfg["training"]["model_name"]
    )

    ap = argparse.ArgumentParser("./ood_test_scans.py")
    ap.add_argument("--model", "-m", default=default_model,
                    help="MCD-trained run dir (weights + saved configs + stats).")
    ap.add_argument("--datasets_root",
                    default="/media/donceykong/doncey_ssd_021/datasets",
                    help="Root holding both 'mcd' and 'kitti360'.")
    ap.add_argument("--mcd_seq", default="kth_day_06")
    ap.add_argument("--kitti_seq", default="2013_05_28_drive_0000_sync")
    # KITTI-360 Velodyne HDL-64E vertical FOV (standard SemanticKITTI values).
    ap.add_argument("--kitti_fov_up", type=float, default=3.0)
    ap.add_argument("--kitti_fov_down", type=float, default=-25.0)
    ap.add_argument("--n", type=int, default=5, help="Scans per group.")
    ap.add_argument("--stats", default=None,
                    help=f"OOD stats path. Default: <model>/{STATS_FILENAME}.")
    ap.add_argument("--out", default="ood_test_out")
    FLAGS, _ = ap.parse_known_args()

    os.makedirs(FLAGS.out, exist_ok=True)

    model, hook, stats, groups, device = build_detector_and_groups(
        FLAGS.model, FLAGS.datasets_root, FLAGS.mcd_seq, FLAGS.kitti_seq, FLAGS.n,
        kitti_fov_up=FLAGS.kitti_fov_up, kitti_fov_down=FLAGS.kitti_fov_down,
        stats_path=FLAGS.stats,
    )

    pooled, first_scan = {}, {}
    print("\nScoring scans...")
    for gname, (ds, count) in groups.items():
        all_scores = []
        for i in range(count):
            xyz, scores = score_scan(model, hook, stats, ds[i], device)
            all_scores.append(scores)
            if i == 0:
                first_scan[gname] = (xyz, scores)
            print(f"  [{gname}] scan {i}: {len(scores)} pts, "
                  f"mean={scores.mean():.2f} max={scores.max():.2f}")
        pooled[gname] = np.concatenate(all_scores)

    hook.remove()

    # --- summary ---
    print("\nPer-group OOD score summary:")
    for gname, scores in pooled.items():
        summarize(gname, scores)

    combined = np.concatenate(list(pooled.values()))
    lo, hi = np.percentile(combined, 1), np.percentile(combined, 99)

    # --- histogram ---
    plt.figure(figsize=(8, 5))
    for gname, scores in pooled.items():
        plt.hist(np.clip(scores, lo, hi), bins=80, density=True, alpha=0.5, label=gname)
    plt.xlabel("Mahalanobis OOD score")
    plt.ylabel("density")
    plt.title("OOD score distribution: in-domain (MCD/KTH) vs shifted (KITTI-360)")
    plt.legend()
    hist_path = os.path.join(FLAGS.out, "score_histogram.png")
    plt.savefig(hist_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved histogram -> {hist_path}")

    # --- colored point clouds (first scan of each group, shared color scale) ---
    for gname, (xyz, scores) in first_scan.items():
        ply_path = os.path.join(FLAGS.out, f"{gname}_scan0.ply")
        write_ply(ply_path, xyz, scores, lo, hi)
        print(f"Saved colored cloud -> {ply_path}  (color range [{lo:.1f}, {hi:.1f}])")

    print("\nExpectation: kitti360 mean/median should sit clearly above mcd_kth.")


if __name__ == "__main__":
    main()
