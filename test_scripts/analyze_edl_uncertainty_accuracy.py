"""
Analyze correlation between EDL uncertainty (vacuity) and prediction accuracy.

Compares ground truth labels with CENet EDL inferred labels and uncertainty scores.
Both GT and predictions are mapped to a common 13-class taxonomy before comparison.

Memory-efficient: accumulates statistics incrementally per scan instead of
concatenating all points into giant arrays.

Produces:
  1. Per-scan accuracy vs mean uncertainty (scatter + regression)
  2. Binned uncertainty vs accuracy curve (reliability diagram)
  3. Per-class accuracy vs mean uncertainty
  4. Per-class regression slopes
  5. Per-class uncertainty distributions
"""

import argparse
import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# =============================================================================
# Common label taxonomy (13 classes)
# =============================================================================
COMMON_LABELS = {
    0: "unlabeled",
    1: "road",
    2: "sidewalk",
    3: "parking",
    4: "other-ground",
    5: "building",
    6: "fence",
    7: "pole",
    8: "traffic-sign",
    9: "vegetation",
    10: "two-wheeler",
    11: "vehicle",
    12: "other-object",
}
NUM_COMMON = 13
IGNORE_LABEL = 0  # unlabeled

# MCD raw label -> common class
MCD_TO_COMMON = {
    0: 6, 1: 10, 2: 5, 3: 12, 4: 4, 5: 12, 6: 4, 7: 6, 8: 12, 9: 8,
    10: 1, 11: 0, 12: 12, 13: 3, 14: 12, 15: 7, 16: 1, 17: 5, 18: 2,
    19: 4, 20: 12, 21: 8, 22: 8, 23: 12, 24: 9, 25: 9, 26: 11, 27: 11, 28: 11,
}

# KITTI-360 raw label -> common class
KITTI360_TO_COMMON = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 12, 6: 4, 7: 1, 8: 2, 9: 3,
    10: 1, 11: 5, 12: 5, 13: 6, 14: 6, 15: 5, 16: 5, 17: 7, 18: 7,
    19: 8, 20: 8, 21: 9, 22: 9, 23: 0, 24: 12, 25: 12, 26: 11, 27: 11,
    28: 11, 29: 11, 30: 11, 31: 11, 32: 10, 33: 10, 34: 5, 35: 6,
    36: 8, 37: 7, 38: 7, 39: 12, 40: 12, 41: 12, 42: 5, 43: 11, 44: 12,
    65535: 0,
}

# =============================================================================
# Dataset root paths
# =============================================================================
DATASET_ROOTS = {
    "kitti360": Path("/media/donceykong/doncey_ssd_011/datasets/KITTI360"),
    "mcd": Path("/media/donceykong/doncey_ssd_011/datasets/MCD"),
}

DATASET_GT_MAPS = {
    "kitti360": KITTI360_TO_COMMON,
    "mcd": MCD_TO_COMMON,
}

INFER_SUBDIR = "inferred_labels/cenet_mcd_EDL"


def resolve_dataset(dataset, seq):
    """Resolve dataset + sequence to paths and label mappings."""
    if dataset not in DATASET_ROOTS:
        print(f"Error: Unknown dataset '{dataset}'. Choose from: {list(DATASET_ROOTS.keys())}")
        sys.exit(1)

    root = DATASET_ROOTS[dataset]

    if dataset == "kitti360":
        seq_dir = root / f"2013_05_28_drive_{seq}_sync"
        title = f"KITTI-360 seq {seq} (cross-domain)"
    else:
        seq_dir = root / seq
        title = f"MCD {seq}"

    if not seq_dir.exists():
        print(f"Error: Sequence directory not found: {seq_dir}")
        sys.exit(1)

    edl_dir = seq_dir / INFER_SUBDIR
    if not edl_dir.exists():
        print(f"Error: EDL inferred labels not found: {edl_dir}")
        print("Run CENet EDL inference on this sequence first.")
        sys.exit(1)

    conf_dir = edl_dir / "confidence_scores"
    if not conf_dir.exists():
        print(f"Error: Confidence scores not found: {conf_dir}")
        sys.exit(1)

    return {
        "base": seq_dir,
        "infer_subdir": INFER_SUBDIR,
        "gt_map": DATASET_GT_MAPS[dataset],
        "pred_map": MCD_TO_COMMON,
        "title_prefix": title,
    }


def build_label_lut(mapping, max_key=None):
    """Build a numpy lookup table from a dict mapping for fast vectorized remapping."""
    if max_key is None:
        max_key = max(mapping.keys())
    lut = np.full(max_key + 1, IGNORE_LABEL, dtype=np.int32)
    for src, dst in mapping.items():
        if src <= max_key:
            lut[src] = dst
    return lut


def apply_label_lut(labels, lut):
    """Fast vectorized label remapping via LUT."""
    # Clamp out-of-range labels to IGNORE_LABEL
    safe = np.where((labels >= 0) & (labels < len(lut)), labels, 0)
    return lut[safe]


def load_scan(gt_dir, pred_dir, conf_dir, name, gt_lut, pred_lut):
    """Load one scan, map both GT and pred to common labels."""
    gt_raw = np.fromfile(gt_dir / name, dtype=np.int32)
    pred_raw = np.fromfile(pred_dir / name, dtype=np.int32)
    vacuity = np.fromfile(conf_dir / name, dtype=np.float16).astype(np.float32)

    gt = apply_label_lut(gt_raw, gt_lut)
    pred = apply_label_lut(pred_raw, pred_lut)
    return gt, pred, vacuity


def main():
    parser = argparse.ArgumentParser(description="EDL uncertainty vs accuracy analysis")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_ROOTS.keys()),
                        help="Dataset name (e.g. kitti360, mcd)")
    parser.add_argument("--seq", required=True,
                        help="Sequence ID (e.g. 0009 for kitti360, kth_night_05 for mcd)")
    args = parser.parse_args()

    cfg = resolve_dataset(args.dataset, args.seq)
    base = cfg["base"]
    gt_dir = base / "gt_labels"
    pred_dir = base / cfg["infer_subdir"]
    conf_dir = pred_dir / "confidence_scores"
    out_dir = Path(__file__).parent / "edl_analysis_outputs" / f"{args.dataset}_{args.seq}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {cfg['title_prefix']}")
    print(f"GT dir:   {gt_dir}")
    print(f"Pred dir: {pred_dir}")
    print(f"Conf dir: {conf_dir}")

    # Build fast LUTs
    gt_lut = build_label_lut(cfg["gt_map"], max_key=max(cfg["gt_map"].keys()))
    pred_lut = build_label_lut(cfg["pred_map"], max_key=max(cfg["pred_map"].keys()))

    # Find common scans
    gt_files = {f.name for f in gt_dir.glob("*.bin")}
    pred_files = {f.name for f in pred_dir.glob("*.bin")}
    conf_files = {f.name for f in conf_dir.glob("*.bin")}
    common = sorted(gt_files & pred_files & conf_files)
    print(f"Found {len(common)} common scans (GT: {len(gt_files)}, Pred: {len(pred_files)})")

    # =========================================================================
    # Pass 1: Accumulate statistics incrementally (no giant arrays)
    # =========================================================================
    N_BINS = 20
    bin_edges = np.linspace(0, 1, N_BINS + 1)

    # Per-scan arrays (small: one entry per scan)
    scan_accs, scan_uncs = [], []

    # Global accumulators
    total_correct = 0
    total_points = 0
    total_vacuity_sum = 0.0

    # Per-bin accumulators: [correct_count, total_count] per bin
    bin_correct = np.zeros(N_BINS, dtype=np.float64)
    bin_total = np.zeros(N_BINS, dtype=np.int64)

    # Per-class accumulators
    class_correct_count = np.zeros(NUM_COMMON, dtype=np.int64)
    class_total_count = np.zeros(NUM_COMMON, dtype=np.int64)
    class_vacuity_sum = np.zeros(NUM_COMMON, dtype=np.float64)
    # For per-class regression: Welford-style running sums for slope computation
    # slope = (sum(xy) - n*mean_x*mean_y) / (sum(x^2) - n*mean_x^2)
    # where x=vacuity, y=correct (0 or 1)
    class_sum_vac = np.zeros(NUM_COMMON, dtype=np.float64)      # sum(x)
    class_sum_vac2 = np.zeros(NUM_COMMON, dtype=np.float64)     # sum(x^2)
    class_sum_vac_cor = np.zeros(NUM_COMMON, dtype=np.float64)  # sum(x*y)

    # Reservoir sample for violin plots (max 50k per class)
    RESERVOIR_SIZE = 50_000
    class_reservoirs = [[] for _ in range(NUM_COMMON)]
    class_reservoir_seen = np.zeros(NUM_COMMON, dtype=np.int64)

    rng = np.random.default_rng(42)

    for scan_idx, name in enumerate(common):
        if scan_idx % 1000 == 0 and scan_idx > 0:
            print(f"  Processing scan {scan_idx}/{len(common)}...")

        gt, pred, vacuity = load_scan(gt_dir, pred_dir, conf_dir, name, gt_lut, pred_lut)

        valid = gt != IGNORE_LABEL
        gt_v = gt[valid]
        pred_v = pred[valid]
        vac_v = vacuity[valid]

        if len(gt_v) == 0:
            continue

        correct = (gt_v == pred_v)
        correct_f = correct.astype(np.float32)

        # Per-scan stats
        scan_accs.append(correct_f.mean())
        scan_uncs.append(vac_v.mean())

        # Global stats
        n = len(gt_v)
        total_correct += correct.sum()
        total_points += n
        total_vacuity_sum += vac_v.sum()

        # Bin stats
        bin_idx = np.clip(np.searchsorted(bin_edges, vac_v, side="right") - 1, 0, N_BINS - 1)
        for b in range(N_BINS):
            mask_b = bin_idx == b
            bin_total[b] += mask_b.sum()
            bin_correct[b] += correct[mask_b].sum()

        # Per-class stats
        for cid in range(NUM_COMMON):
            if cid == IGNORE_LABEL:
                continue
            mask_c = gt_v == cid
            nc = mask_c.sum()
            if nc == 0:
                continue

            vac_c = vac_v[mask_c]
            cor_c = correct_f[mask_c]

            class_total_count[cid] += nc
            class_correct_count[cid] += cor_c.sum()
            class_vacuity_sum[cid] += vac_c.sum()
            class_sum_vac[cid] += vac_c.sum()
            class_sum_vac2[cid] += (vac_c * vac_c).sum()
            class_sum_vac_cor[cid] += (vac_c * cor_c).sum()

            # Reservoir sampling for violin plot
            prev_seen = class_reservoir_seen[cid]
            class_reservoir_seen[cid] += nc
            reservoir = class_reservoirs[cid]

            if len(reservoir) < RESERVOIR_SIZE:
                need = RESERVOIR_SIZE - len(reservoir)
                reservoir.extend(vac_c[:need].tolist())
                remaining = vac_c[need:]
                start_idx = prev_seen + need
            else:
                remaining = vac_c
                start_idx = prev_seen

            # Standard reservoir sampling for the rest
            if len(remaining) > 0 and len(reservoir) >= RESERVOIR_SIZE:
                total_seen = class_reservoir_seen[cid]
                indices = rng.integers(0, total_seen, size=len(remaining))
                for k, idx in enumerate(indices):
                    if idx < RESERVOIR_SIZE:
                        reservoir[idx] = remaining[k]

    scan_accs = np.array(scan_accs)
    scan_uncs = np.array(scan_uncs)

    overall_acc = total_correct / total_points if total_points > 0 else 0
    overall_vac = total_vacuity_sum / total_points if total_points > 0 else 0

    print(f"Total valid points: {total_points:,}")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f"Mean vacuity: {overall_vac:.4f}")

    title_pfx = cfg["title_prefix"]

    # =========================================================================
    # Figure 1: Per-scan accuracy vs mean uncertainty with regression
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(scan_uncs, scan_accs, alpha=0.3, s=10, color="steelblue")

    slope, intercept, r, p, se = stats.linregress(scan_uncs, scan_accs)
    x_fit = np.linspace(scan_uncs.min(), scan_uncs.max(), 100)
    p_str = f"{p:.2e}" if p > 0 else "<1e-300"
    ax.plot(x_fit, slope * x_fit + intercept, "r-", linewidth=2,
            label=f"y = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r**2:.4f}, p = {p_str}")

    ax.set_xlabel("Mean Vacuity (uncertainty)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title_pfx}\nPer-Scan: Accuracy vs Mean Uncertainty", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_scan_accuracy_vs_uncertainty.png", dpi=150)
    plt.close(fig)
    print(f"\nPer-scan regression: slope={slope:.4f}, R^2={r**2:.4f}, p={p_str}")

    # =========================================================================
    # Figure 2: Binned uncertainty vs accuracy (reliability diagram)
    # =========================================================================
    valid_bins = bin_total > 0
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)[valid_bins]
    bin_accs = (bin_correct[valid_bins] / bin_total[valid_bins]).astype(np.float64)
    bin_counts = bin_total[valid_bins]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(bin_centers, bin_accs, width=1.0 / N_BINS * 0.8, color="steelblue",
            alpha=0.7, label="Accuracy")

    slope_b, intercept_b, r_b, p_b, _ = stats.linregress(bin_centers, bin_accs)
    ax1.plot(bin_centers, slope_b * bin_centers + intercept_b, "r-", linewidth=2,
             label=f"Regression: $R^2$={r_b**2:.4f}")

    ax1.set_xlabel("Vacuity (uncertainty)", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(f"{title_pfx}\nBinned Uncertainty vs Accuracy (Reliability Diagram)", fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(bin_centers, bin_counts / bin_counts.sum(), "k--", alpha=0.5, label="Point fraction")
    ax2.set_ylabel("Fraction of points", fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "binned_uncertainty_vs_accuracy.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Compute per-class derived stats
    # =========================================================================
    class_ids = sorted(set(COMMON_LABELS.keys()) - {IGNORE_LABEL})
    pc_accs, pc_uncs, pc_names, pc_sizes = [], [], [], []
    pc_slopes, pc_r2s, pc_names_slope = [], [], []

    for cid in class_ids:
        n = class_total_count[cid]
        if n < 100:
            continue

        acc = class_correct_count[cid] / n
        mean_vac = class_vacuity_sum[cid] / n

        pc_accs.append(acc)
        pc_uncs.append(mean_vac)
        pc_names.append(COMMON_LABELS[cid])
        pc_sizes.append(n)

        # Compute regression slope from running sums:
        # slope = (sum_xy - n*mean_x*mean_y) / (sum_x2 - n*mean_x^2)
        mean_x = class_sum_vac[cid] / n
        mean_y = class_correct_count[cid] / n
        denom = class_sum_vac2[cid] - n * mean_x * mean_x
        if abs(denom) < 1e-12:
            continue
        numer = class_sum_vac_cor[cid] - n * mean_x * mean_y
        sl = numer / denom

        # R^2 = slope^2 * var_x / var_y
        var_x = denom / n
        var_y = mean_y * (1 - mean_y)  # binary variance
        r2 = (sl * sl * var_x / var_y) if var_y > 1e-12 else 0.0
        r2 = min(r2, 1.0)

        pc_slopes.append(sl)
        pc_r2s.append(r2)
        pc_names_slope.append(COMMON_LABELS[cid])

    pc_accs = np.array(pc_accs)
    pc_uncs = np.array(pc_uncs)
    pc_sizes = np.array(pc_sizes)
    pc_slopes = np.array(pc_slopes)
    pc_r2s = np.array(pc_r2s)

    # =========================================================================
    # Figure 3: Per-class accuracy vs mean uncertainty
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(pc_uncs, pc_accs,
                         s=np.log10(pc_sizes) * 30, c=pc_accs,
                         cmap="RdYlGn", edgecolors="black", linewidth=0.5,
                         vmin=0, vmax=1)

    for i, name in enumerate(pc_names):
        ax.annotate(name, (pc_uncs[i], pc_accs[i]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    if len(pc_uncs) > 2:
        slope_c, intercept_c, r_c, p_c, _ = stats.linregress(pc_uncs, pc_accs)
        x_fit_c = np.linspace(min(pc_uncs), max(pc_uncs), 100)
        ax.plot(x_fit_c, slope_c * x_fit_c + intercept_c, "r--", linewidth=2,
                label=f"$R^2$={r_c**2:.4f}, p={p_c:.3f}")
    else:
        slope_c, r_c = float("nan"), float("nan")

    ax.set_xlabel("Mean Vacuity (uncertainty)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title_pfx}\nPer-Class: Accuracy vs Mean Uncertainty\n(point size = log10 class count)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_accuracy_vs_uncertainty.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Figure 4: Per-class regression slopes (uncertainty vs correctness)
    # =========================================================================
    if len(pc_slopes) > 0:
        sort_idx = np.argsort(pc_slopes)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["green" if s < 0 else "red" for s in pc_slopes[sort_idx]]
        ax.barh(range(len(pc_slopes)), pc_slopes[sort_idx], color=colors, alpha=0.7,
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(pc_slopes)))
        ax.set_yticklabels([pc_names_slope[i] for i in sort_idx], fontsize=9)
        ax.set_xlabel("Regression Slope (vacuity vs correctness)", fontsize=12)
        ax.set_title(f"{title_pfx}\nPer-Class: Slope of Uncertainty-Correctness Regression\n"
                     "(green = negative = well-calibrated)", fontsize=13)
        ax.axvline(0, color="black", linewidth=0.8)
        for j, idx in enumerate(sort_idx):
            ax.text(pc_slopes[idx], j, f" R\u00b2={pc_r2s[idx]:.3f}", va="center",
                    fontsize=7, color="black")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(out_dir / "per_class_regression_slopes.png", dpi=150)
        plt.close(fig)

    # =========================================================================
    # Figure 5: Per-class uncertainty distributions (violin plot)
    # =========================================================================
    violin_data, violin_names, violin_medians = [], [], []
    for cid in class_ids:
        reservoir = class_reservoirs[cid]
        if len(reservoir) < 100:
            continue
        arr = np.array(reservoir, dtype=np.float32)
        violin_data.append(arr)
        violin_names.append(COMMON_LABELS[cid])
        violin_medians.append(np.median(arr))

    if len(violin_data) > 0:
        median_order = np.argsort(violin_medians)
        violin_data = [violin_data[i] for i in median_order]
        violin_names = [violin_names[i] for i in median_order]

        fig, ax = plt.subplots(figsize=(12, 8))
        parts = ax.violinplot(violin_data, positions=range(len(violin_data)),
                              showmeans=True, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("red")
        parts["cmedians"].set_color("black")

        ax.set_xticks(range(len(violin_names)))
        ax.set_xticklabels(violin_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Vacuity (uncertainty)", fontsize=12)
        ax.set_title(f"{title_pfx}\nPer-Class Uncertainty Distribution\n"
                     "(ordered by median vacuity, red=mean, black=median)", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "per_class_uncertainty_distribution.png", dpi=150)
        plt.close(fig)

    # =========================================================================
    # Summary stats
    # =========================================================================
    print(f"Binned regression: slope={slope_b:.4f}, R^2={r_b**2:.4f}")
    if not np.isnan(r_c):
        print(f"Per-class regression: slope={slope_c:.4f}, R^2={r_c**2:.4f}")

    # Per-class summary table
    slope_lookup = dict(zip(pc_names_slope, zip(pc_slopes, pc_r2s)))
    print(f"\n{'Class':<16} {'Count':>12} {'Accuracy':>10} {'Mean Vacuity':>14} {'Slope':>8} {'R\u00b2':>8}")
    print("-" * 72)
    for i in np.argsort(pc_uncs):
        name = pc_names[i]
        sl, r2 = slope_lookup.get(name, (float("nan"), float("nan")))
        print(f"{name:<16} {pc_sizes[i]:>12,} {pc_accs[i]:>10.4f} {pc_uncs[i]:>14.4f} {sl:>8.4f} {r2:>8.4f}")

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
