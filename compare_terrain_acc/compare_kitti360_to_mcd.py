"""
Evaluate KITTI-360-trained CENet predictions on the MCD kth_day_09 sequence
by mapping both predictions and GT into the 9-class common taxonomy from
config/datasets/labels/labels_common.yaml.

Predictions live in:
  inferred_labels/cenet_kitti360{,_notta}/<frame>.bin   uint32, KITTI-360 ids
GT lives in:
  gt_labels_terrain/<frame>.bin                         uint32, MCD ids

Mapping:
  KITTI-360 raw  --kitti360_to_common-->  common [0..8]
  MCD raw        --mcd_to_common------->  common [0..8]
Then we accumulate a 9x9 confusion matrix per method and report IoU,
accuracy, precision. mIoU is computed over the semantic classes 1..8
(class 0 = 'unlabeled' is treated as ignore by default).

Run:
  python3 compare_kitti360_to_mcd.py
  python3 compare_kitti360_to_mcd.py cenet_kitti360 cenet_kitti360_notta
  python3 compare_kitti360_to_mcd.py --gt-name gt_labels --limit 500
"""

import argparse
import os
import time

import numpy as np
import yaml

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SEQ_DIR = "/media/donceykong/doncey_ssd_021/datasets/mcd/kth_day_09"
LABELS_COMMON_YAML = (
    "/home/donceykong/Desktop/ARPG/projects/OSM-BKI-FULL/OSM-BKI-ROS2/"
    "OSM-BKI/OSM-BKI-ROS2/config/datasets/labels/labels_common.yaml"
)

# Default methods to evaluate (must be subdirs of <SEQ_DIR>/inferred_labels).
DEFAULT_METHODS = ["cenet_kitti360", "cenet_kitti360_notta"]

# ----------------------------------------------------------------------
# Common taxonomy
# ----------------------------------------------------------------------
COMMON_CLASSES = [
    "unlabeled",   # 0  (ignored in mIoU by default)
    "road",        # 1
    "sidewalk",    # 2
    "parking",     # 3
    "building",    # 4
    "fence",       # 5
    "vegetation",  # 6
    "vehicle",     # 7
    "terrain",     # 8
]
NUM_COMMON = len(COMMON_CLASSES)
TERRAIN_COMMON = 8


def load_common_lut(yaml_path: str) -> dict:
    """Parses labels_common.yaml and returns the kitti360 / mcd LUTs as
    numpy uint8 arrays of length max_id+1 (default-filled with 0 = unlabeled).
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("num_class", NUM_COMMON) != NUM_COMMON:
        raise ValueError(
            f"labels_common.yaml has num_class={cfg.get('num_class')}, "
            f"expected {NUM_COMMON}; update COMMON_CLASSES if the taxonomy changed."
        )

    def to_lut(mapping: dict[int, int]) -> np.ndarray:
        max_id = max(mapping.keys())
        # cap to a sane size; KITTI-360 has 65535 but that's just sentinel.
        lut_size = max_id + 1
        lut = np.zeros(lut_size, dtype=np.uint8)
        for k, v in mapping.items():
            lut[k] = int(v)
        return lut

    return {
        "kitti360": to_lut(cfg["kitti360_to_common"]),
        "mcd":      to_lut(cfg["mcd_to_common"]),
        "semkitti": to_lut(cfg["semkitti_to_common"]),
    }


def remap(labels: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """LUT-remap a uint32 label array, defaulting any out-of-range id to 0."""
    lab = labels.astype(np.int64, copy=False)
    out = np.zeros_like(lab, dtype=np.uint8)
    in_range = lab < lut.size
    out[in_range] = lut[lab[in_range]]
    return out


# ----------------------------------------------------------------------
# Accumulation + metrics
# ----------------------------------------------------------------------
def accumulate_confusion_common(
    gt_dir: str,
    pred_dir: str,
    filenames: list[str],
    pred_lut: np.ndarray,
    gt_lut: np.ndarray,
    ignore_common: set[int],
) -> tuple[np.ndarray, int, int]:
    cm = np.zeros((NUM_COMMON, NUM_COMMON), dtype=np.int64)
    total = 0
    kept = 0
    t0 = time.time()
    for i, fname in enumerate(filenames):
        gt_raw = np.fromfile(os.path.join(gt_dir,  fname), dtype=np.uint32)
        pr_raw = np.fromfile(os.path.join(pred_dir, fname), dtype=np.uint32)
        if gt_raw.shape != pr_raw.shape:
            raise ValueError(
                f"shape mismatch on {fname}: gt={gt_raw.shape} pred={pr_raw.shape}"
            )
        total += gt_raw.size

        gt = remap(gt_raw, gt_lut).astype(np.int64, copy=False)
        pr = remap(pr_raw, pred_lut).astype(np.int64, copy=False)

        # Build an ignore mask over GT classes.
        if ignore_common:
            ig = np.zeros(NUM_COMMON, dtype=bool)
            for c in ignore_common:
                if 0 <= c < NUM_COMMON:
                    ig[c] = True
            keep = ~ig[gt]
            gt = gt[keep]; pr = pr[keep]
        kept += gt.size

        idx = gt * NUM_COMMON + pr
        cm += np.bincount(idx, minlength=NUM_COMMON * NUM_COMMON) \
                .reshape(NUM_COMMON, NUM_COMMON)

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
        "cm": cm,
    }


def print_per_class(name: str, m: dict, kept: int, semantic: list[int]) -> None:
    iou, rec, pre = m["iou"], m["recall"], m["precision"]
    gt, pred = m["gt"], m["pred"]

    print(f"\n=== {name} (common 9-class space) ===")
    print(f"{'id':>3}  {'name':<14s}  {'IoU':>8s}  {'Acc':>8s}  "
          f"{'Prec':>8s}  {'GT':>12s}  {'Pred':>12s}")
    print("-" * 78)
    for c in range(NUM_COMMON):
        print(f"{c:>3d}  {COMMON_CLASSES[c]:<14s}  "
              f"{iou[c]:>8.4f}  {rec[c]:>8.4f}  {pre[c]:>8.4f}  "
              f"{int(gt[c]):>12d}  {int(pred[c]):>12d}")

    sem = np.asarray(semantic, dtype=np.int64)
    present = gt[sem] > 0
    sem_present = sem[present]
    miou = iou[sem_present].mean() if sem_present.size else 0.0
    mrec = rec[sem_present].mean() if sem_present.size else 0.0
    mpre = pre[sem_present].mean() if sem_present.size else 0.0
    sem_tp_sum = m["tp"][sem].sum()
    sem_gt_sum = gt[sem].sum()
    oacc = sem_tp_sum / sem_gt_sum if sem_gt_sum > 0 else 0.0
    print(f"  mIoU (over {int(present.sum())} of {sem.size} semantic cls): {miou:.4f}")
    print(f"  mean recall:        {mrec:.4f}")
    print(f"  mean precision:     {mpre:.4f}")
    print(f"  overall accuracy:   {oacc:.4f}  (kept={kept})")


def print_confusion(name: str, cm: np.ndarray) -> None:
    print(f"\nConfusion matrix for {name}  (rows=GT common, cols=pred common):")
    header = "       " + "  ".join(f"{COMMON_CLASSES[c][:7]:>7s}" for c in range(NUM_COMMON))
    print(header)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    for r in range(NUM_COMMON):
        cells = "  ".join(f"{norm[r, c]:>7.3f}" for c in range(NUM_COMMON))
        print(f"{COMMON_CLASSES[r][:6]:>6s}  {cells}   (n={int(row_sums[r,0])})")


def print_side_by_side(results: dict[str, dict], semantic: list[int]) -> None:
    names = list(results.keys())
    if len(names) < 2:
        return
    sem = np.asarray(semantic, dtype=np.int64)

    print("\n" + "=" * (24 + 18 * len(names)))
    print("Side-by-side (common space)")
    print("=" * (24 + 18 * len(names)))
    print(f"{'id':>3}  {'name':<14s}  " +
          "  ".join(f"{'IoU_' + chr(65+i):>8s}" for i in range(len(names))) +
          "  " +
          "  ".join(f"{'Acc_' + chr(65+i):>8s}" for i in range(len(names))))
    print("-" * (24 + 18 * len(names)))
    for c in range(NUM_COMMON):
        marker = "  ← TERRAIN" if c == TERRAIN_COMMON else ""
        ious = "  ".join(f"{results[n]['iou'][c]:>8.4f}" for n in names)
        accs = "  ".join(f"{results[n]['recall'][c]:>8.4f}" for n in names)
        print(f"{c:>3d}  {COMMON_CLASSES[c]:<14s}  {ious}  {accs}{marker}")

    print("-" * (24 + 18 * len(names)))
    for n in names:
        gt = results[n]["gt"][sem]
        present = gt > 0
        if present.any():
            miou = results[n]["iou"][sem[present]].mean()
            mrec = results[n]["recall"][sem[present]].mean()
        else:
            miou = mrec = 0.0
        terr_iou = results[n]["iou"][TERRAIN_COMMON]
        terr_acc = results[n]["recall"][TERRAIN_COMMON]
        print(f"  {n:<35s} mIoU={miou:.4f}  mRec={mrec:.4f}  "
              f"terrain IoU={terr_iou:.4f}  terrain Acc={terr_acc:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("methods", nargs="*", default=DEFAULT_METHODS,
                   help="Subdirs under inferred_labels/ to evaluate.")
    p.add_argument("--seq-dir", default=SEQ_DIR)
    p.add_argument("--gt-name", default="gt_labels_terrain",
                   choices=["gt_labels", "gt_labels_terrain"],
                   help="Which GT folder to use (both are MCD label space).")
    p.add_argument("--labels-common", default=LABELS_COMMON_YAML)
    p.add_argument("--pred-lut", default="kitti360",
                   choices=["kitti360", "mcd", "semkitti"],
                   help="Which *_to_common LUT applies to the predictions.")
    p.add_argument("--ignore", type=int, nargs="*", default=[0],
                   help="Common class ids to ignore in GT (default: [0] = unlabeled).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of frames.")
    p.add_argument("--show-confusion", action="store_true",
                   help="Print row-normalized 9x9 confusion matrix for each method.")
    args = p.parse_args()

    print(f"Loading common LUTs from: {args.labels_common}")
    luts = load_common_lut(args.labels_common)
    pred_lut = luts[args.pred_lut]
    gt_lut = luts["mcd"]
    print(f"  pred LUT '{args.pred_lut}' size={pred_lut.size}, "
          f"unique targets={sorted(set(pred_lut.tolist()))}")
    print(f"  gt LUT   'mcd' size={gt_lut.size}, "
          f"unique targets={sorted(set(gt_lut.tolist()))}")

    gt_dir = os.path.join(args.seq_dir, args.gt_name)
    gt_files = set(os.listdir(gt_dir))

    # Build common file set across all chosen methods.
    method_dirs = {m: os.path.join(args.seq_dir, "inferred_labels", m) for m in args.methods}
    for m, d in method_dirs.items():
        if not os.path.isdir(d):
            raise SystemExit(f"Missing method directory: {d}")
    common = gt_files.copy()
    for d in method_dirs.values():
        common &= set(os.listdir(d))
    common = sorted(common)
    if args.limit:
        common = common[: args.limit]
    if not common:
        raise SystemExit("No frames common to GT and all chosen methods.")

    # Semantic classes (everything except ignored).
    ignore_set = set(args.ignore)
    semantic = [c for c in range(NUM_COMMON) if c not in ignore_set]

    print(f"\nGT dir:           {gt_dir}")
    print(f"Methods:          {args.methods}")
    print(f"Matching frames:  {len(common)}  (gt={len(gt_files)})")
    print(f"Ignored common:   {sorted(ignore_set)}  -> semantic={semantic}")

    results: dict[str, dict] = {}
    for m, d in method_dirs.items():
        print(f"\n--- accumulating: {m} ---")
        cm, total, kept = accumulate_confusion_common(
            gt_dir, d, common, pred_lut, gt_lut, ignore_set,
        )
        mx = metrics_from_cm(cm)
        mx["_kept"] = kept
        mx["_total"] = total
        results[m] = mx
        print(f"  total points: {total}   kept after ignore: {kept}")

    for m, mx in results.items():
        print_per_class(m, mx, mx["_kept"], semantic)
        if args.show_confusion:
            print_confusion(m, mx["cm"])

    print_side_by_side(results, semantic)


if __name__ == "__main__":
    main()
