"""
Compare two CENet terrain inference outputs against gt_labels_terrain.

Both inferred directories and the GT use the MCD 30-class label space
(see config/datasets/labels/labels_mcd.yaml). All .bin files are uint32
arrays of length N (one label per point). Files are matched by filename
(GT frames 0000000011.bin .. ; predictions start earlier — we intersect).

For each method we accumulate a 30x30 confusion matrix
(rows = GT, cols = prediction), then derive per-class IoU,
accuracy/recall, and precision. mIoU and mean accuracy are computed
over the classes that have any GT presence (after ignore-mask).

Run:
  python3 compare_terrain_acc.py
  python3 compare_terrain_acc.py --ignore 0 11   # also ignore barrier+noise GT
"""

import argparse
import os
import time

import numpy as np

# ----------------------------------------------------------------------
# MCD label scheme — keep in sync with
# OSM-BKI-ROS2/config/datasets/labels/labels_mcd.yaml
# ----------------------------------------------------------------------
MCD_LABELS = {
    0:  "barrier",         1:  "bike",          2:  "building",
    3:  "chair",           4:  "cliff",         5:  "container",
    6:  "curb",            7:  "fence",         8:  "hydrant",
    9:  "infosign",        10: "lanemarking",   11: "noise",
    12: "other",           13: "parkinglot",    14: "pedestrian",
    15: "pole",            16: "road",          17: "shelter",
    18: "sidewalk",        19: "stairs",        20: "structure-other",
    21: "traffic-cone",    22: "traffic-sign",  23: "trashbin",
    24: "treetrunk",       25: "vegetation",    26: "vehicle-dynamic",
    27: "vehicle-other",   28: "vehicle-static", 29: "terrain",
}
NUM_CLASSES = 30
TERRAIN_CLS = 29


# ----------------------------------------------------------------------
# Default paths
# ----------------------------------------------------------------------
SEQ_DIR = "/media/donceykong/doncey_ssd_021/datasets/mcd/kth_day_09"
GT_DIR = os.path.join(SEQ_DIR, "gt_labels_terrain")
METHOD_DIRS = {
    "cenet_mcd_terrain":          os.path.join(SEQ_DIR, "inferred_labels/cenet_mcd_terrain"),
    "cenet_mcd_terrain_ntu_tuhh": os.path.join(SEQ_DIR, "inferred_labels/cenet_mcd_terrain_ntu_tuhh"),
}


def load_labels(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint32)


def accumulate_confusion(
    gt_dir: str,
    pred_dir: str,
    filenames: list[str],
    ignore: set[int],
) -> tuple[np.ndarray, int, int]:
    """Returns (confusion[30,30], total_points, kept_points)."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total = 0
    kept = 0
    t0 = time.time()
    for i, fname in enumerate(filenames):
        gt = load_labels(os.path.join(gt_dir, fname))
        pr = load_labels(os.path.join(pred_dir, fname))
        if gt.shape != pr.shape:
            raise ValueError(
                f"shape mismatch on {fname}: gt={gt.shape} pred={pr.shape}"
            )
        total += gt.size

        # Discard out-of-range labels (defensive) and ignore classes.
        valid = (gt < NUM_CLASSES) & (pr < NUM_CLASSES)
        if ignore:
            ig_mask = np.zeros(NUM_CLASSES, dtype=bool)
            for c in ignore:
                if 0 <= c < NUM_CLASSES:
                    ig_mask[c] = True
            valid &= ~ig_mask[gt.clip(max=NUM_CLASSES - 1)]
        gt_v = gt[valid].astype(np.int64, copy=False)
        pr_v = pr[valid].astype(np.int64, copy=False)
        kept += gt_v.size

        idx = gt_v * NUM_CLASSES + pr_v
        cm += np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES) \
                .reshape(NUM_CLASSES, NUM_CLASSES)

        if (i + 1) % 500 == 0 or (i + 1) == len(filenames):
            print(f"  [{i+1:>5d}/{len(filenames)}] kept={kept:>12d}  "
                  f"elapsed={time.time()-t0:6.1f}s")
    return cm, total, kept


def metrics_from_cm(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(np.float64)
    gt_counts = cm.sum(axis=1).astype(np.float64)
    pred_counts = cm.sum(axis=0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom_iou = gt_counts + pred_counts - tp
        iou = np.where(denom_iou > 0, tp / denom_iou, 0.0)
        recall = np.where(gt_counts > 0, tp / gt_counts, 0.0)
        precision = np.where(pred_counts > 0, tp / pred_counts, 0.0)
    return {
        "tp": tp, "gt": gt_counts, "pred": pred_counts,
        "iou": iou, "recall": recall, "precision": precision,
    }


def print_per_class(name: str, m: dict, kept: int) -> None:
    iou, rec, pre = m["iou"], m["recall"], m["precision"]
    gt, pred = m["gt"], m["pred"]
    present = gt > 0

    print(f"\n=== {name} ===")
    print(f"{'id':>3}  {'name':<18s}  {'IoU':>8s}  {'Acc':>8s}  "
          f"{'Prec':>8s}  {'GT':>12s}  {'Pred':>12s}")
    print("-" * 80)
    for c in range(NUM_CLASSES):
        if not present[c] and pred[c] == 0:
            continue
        print(f"{c:>3d}  {MCD_LABELS[c]:<18s}  "
              f"{iou[c]:>8.4f}  {rec[c]:>8.4f}  {pre[c]:>8.4f}  "
              f"{int(gt[c]):>12d}  {int(pred[c]):>12d}")
    miou = iou[present].mean() if present.any() else 0.0
    mrec = rec[present].mean() if present.any() else 0.0
    mpre = pre[present].mean() if present.any() else 0.0
    oacc = m["tp"].sum() / kept if kept > 0 else 0.0
    print(f"  mIoU (over {int(present.sum())} present cls): {miou:.4f}")
    print(f"  mean recall:                       {mrec:.4f}")
    print(f"  mean precision:                    {mpre:.4f}")
    print(f"  overall accuracy:                  {oacc:.4f}")


def print_side_by_side(metrics: dict[str, dict]) -> None:
    names = list(metrics.keys())
    if len(names) != 2:
        return
    a, b = names
    ma, mb = metrics[a], metrics[b]
    present = (ma["gt"] > 0) | (mb["gt"] > 0)

    print("\n" + "=" * 96)
    print(f"Side-by-side: {a}  vs  {b}")
    print("=" * 96)
    print(f"{'id':>3}  {'name':<18s}  "
          f"{'IoU_A':>8s} {'IoU_B':>8s} {'ΔIoU':>8s}  "
          f"{'Acc_A':>8s} {'Acc_B':>8s} {'ΔAcc':>8s}")
    print("-" * 96)
    for c in range(NUM_CLASSES):
        if not present[c]:
            continue
        marker = "  ←TERRAIN" if c == TERRAIN_CLS else ""
        diou = mb["iou"][c] - ma["iou"][c]
        dacc = mb["recall"][c] - ma["recall"][c]
        print(f"{c:>3d}  {MCD_LABELS[c]:<18s}  "
              f"{ma['iou'][c]:>8.4f} {mb['iou'][c]:>8.4f} {diou:>+8.4f}  "
              f"{ma['recall'][c]:>8.4f} {mb['recall'][c]:>8.4f} {dacc:>+8.4f}"
              f"{marker}")

    pa = ma["gt"] > 0
    pb = mb["gt"] > 0
    miou_a = ma["iou"][pa].mean() if pa.any() else 0.0
    miou_b = mb["iou"][pb].mean() if pb.any() else 0.0
    mrec_a = ma["recall"][pa].mean() if pa.any() else 0.0
    mrec_b = mb["recall"][pb].mean() if pb.any() else 0.0
    print("-" * 96)
    print(f"  mIoU:        A={miou_a:.4f}  B={miou_b:.4f}  Δ={miou_b-miou_a:+.4f}")
    print(f"  mean recall: A={mrec_a:.4f}  B={mrec_b:.4f}  Δ={mrec_b-mrec_a:+.4f}")
    print(f"  TERRAIN IoU: A={ma['iou'][TERRAIN_CLS]:.4f}  "
          f"B={mb['iou'][TERRAIN_CLS]:.4f}  "
          f"Δ={mb['iou'][TERRAIN_CLS]-ma['iou'][TERRAIN_CLS]:+.4f}")
    print(f"  TERRAIN Acc: A={ma['recall'][TERRAIN_CLS]:.4f}  "
          f"B={mb['recall'][TERRAIN_CLS]:.4f}  "
          f"Δ={mb['recall'][TERRAIN_CLS]-ma['recall'][TERRAIN_CLS]:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", default=GT_DIR)
    p.add_argument("--method-a", default=METHOD_DIRS["cenet_mcd_terrain"])
    p.add_argument("--method-b", default=METHOD_DIRS["cenet_mcd_terrain_ntu_tuhh"])
    p.add_argument(
        "--ignore", type=int, nargs="*", default=[],
        help="GT class ids to ignore (e.g. --ignore 11 to drop 'noise').",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of frames (for quick smoke runs).",
    )
    args = p.parse_args()

    gt_files = set(os.listdir(args.gt_dir))
    a_files = set(os.listdir(args.method_a))
    b_files = set(os.listdir(args.method_b))
    common = sorted(gt_files & a_files & b_files)
    if args.limit:
        common = common[: args.limit]
    if not common:
        raise SystemExit("No common files between GT and the two methods.")
    print(f"Matching frames: {len(common)} "
          f"(gt={len(gt_files)}, A={len(a_files)}, B={len(b_files)})")
    print(f"Ignored GT classes: {sorted(args.ignore) if args.ignore else 'none'}")

    methods = {
        os.path.basename(args.method_a.rstrip('/')): args.method_a,
        os.path.basename(args.method_b.rstrip('/')): args.method_b,
    }

    results = {}
    for name, pred_dir in methods.items():
        print(f"\n--- accumulating: {name} ---")
        cm, total, kept = accumulate_confusion(
            args.gt_dir, pred_dir, common, set(args.ignore),
        )
        m = metrics_from_cm(cm)
        m["_kept"] = kept
        m["_total"] = total
        results[name] = m
        print(f"  total points: {total}   kept after ignore: {kept}")

    for name, m in results.items():
        print_per_class(name, m, m["_kept"])

    print_side_by_side(results)


if __name__ == "__main__":
    main()
