"""
Plot per-class accuracy vs EDL vacuity for cenet_mcd_terrain_ntu_tuhh_EDL
applied to KITTI-360 sequence 2013_05_28_drive_0000_sync.

Cross-domain twist: the EDL model is *MCD-trained* but evaluated on
KITTI-360. So:
  GT      (KITTI-360 raw 0..44 + 65535) --kitti360_to_common--> common [0..8]
  PRED    (MCD raw 0..29)                --mcd_to_common------> common [0..8]
Both LUTs come from config/datasets/labels/labels_common.yaml.

Layout (one figure, two stacked panels sharing x-axis):
  Top:    accuracy per vacuity bin -- one line per "major" common class
          (GT presence >= --min-frac), plus an overall line in black.
          Each line is the *pooled* accuracy; behind it a translucent
          "error band" (a.k.a. confidence band / fan chart) shows the
          per-frame spread of accuracy in that bin (--band: pct/std/sem).
  Bottom: histogram of point counts per vacuity bin (log y).

Vacuity = K/S (K=30 MCD classes, S=sum of Dirichlet alphas), stored as
float16 in <EDL>/confidence_scores/<name>.bin. 0 = certain, higher = less
sure. NOTE: the K=30 is fixed by the *model* (MCD class count), so the
absolute vacuity scale is identical to the MCD evaluation and the two
plots are directly comparable.

Run:
  python3 plot_edl_vacuity_vs_acc_kitti360.py
  python3 plot_edl_vacuity_vs_acc_kitti360.py --bins 30 --vmax 1.0 --limit 500
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ----------------------------------------------------------------------
# Common 9-class taxonomy (must match labels_common.yaml)
# ----------------------------------------------------------------------
COMMON_CLASSES = [
    "unlabeled",   # 0  (ignored by default)
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

LABELS_COMMON_YAML = (
    "/home/donceykong/Desktop/ARPG/projects/OSM-BKI-FULL/OSM-BKI-ROS2/"
    "OSM-BKI/OSM-BKI-ROS2/config/datasets/labels/labels_common.yaml"
)
KITTI360_SEQ = "/media/donceykong/donceys_data_ssd/datasets/kitti360/2013_05_28_drive_0009_sync"
EDL_SUBDIR = "inferred_labels/cenet_mcd_relu_mse_W10_kl0p01"


def load_common_lut(yaml_path: str) -> dict[str, np.ndarray]:
    """Parses labels_common.yaml; returns {'kitti360', 'mcd', 'semkitti'} LUTs
    as uint8 arrays of length max_id+1, defaulting to 0 (unlabeled)."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("num_class", NUM_COMMON) != NUM_COMMON:
        raise ValueError(
            f"labels_common.yaml has num_class={cfg.get('num_class')}, "
            f"expected {NUM_COMMON}."
        )

    def to_lut(mapping: dict[int, int]) -> np.ndarray:
        max_id = max(mapping.keys())
        lut = np.zeros(max_id + 1, dtype=np.uint8)
        for k, v in mapping.items():
            lut[k] = int(v)
        return lut

    return {
        "kitti360": to_lut(cfg["kitti360_to_common"]),
        "mcd":      to_lut(cfg["mcd_to_common"]),
        "semkitti": to_lut(cfg["semkitti_to_common"]),
    }


def remap(labels: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """LUT-remap a uint32 label array. Any id >= lut.size becomes 0."""
    lab = labels.astype(np.int64, copy=False)
    out = np.zeros_like(lab, dtype=np.uint8)
    in_range = lab < lut.size
    out[in_range] = lut[lab[in_range]]
    return out


def accumulate(
    gt_dir: str,
    pred_dir: str,
    vac_dir: str,
    files: list[str],
    pred_lut: np.ndarray,
    gt_lut: np.ndarray,
    nbins: int,
    vmax: float,
    ignore: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (correct[C, nbins], total[C, nbins], edges,
    pf_correct[F, C, nbins], pf_totals[F, C, nbins]) where C=NUM_COMMON and
    F=len(files). The per-frame (pf_*) arrays let the plot show a spread band."""
    edges = np.linspace(0.0, vmax, nbins + 1, dtype=np.float64)
    correct = np.zeros((NUM_COMMON, nbins), dtype=np.int64)
    totals  = np.zeros((NUM_COMMON, nbins), dtype=np.int64)
    nframes = len(files)
    pf_correct = np.zeros((nframes, NUM_COMMON, nbins), dtype=np.int32)
    pf_totals  = np.zeros((nframes, NUM_COMMON, nbins), dtype=np.int32)

    t0 = time.time()
    for i, fname in enumerate(files):
        gt_raw = np.fromfile(os.path.join(gt_dir,  fname), dtype=np.uint32)
        pr_raw = np.fromfile(os.path.join(pred_dir, fname), dtype=np.uint32)
        vc     = np.fromfile(os.path.join(vac_dir, fname), dtype=np.float16).astype(np.float32)
        if not (gt_raw.shape == pr_raw.shape == vc.shape):
            raise ValueError(
                f"shape mismatch on {fname}: gt={gt_raw.shape} "
                f"pred={pr_raw.shape} vac={vc.shape}"
            )

        gt = remap(gt_raw, gt_lut).astype(np.int64, copy=False)
        pr = remap(pr_raw, pred_lut).astype(np.int64, copy=False)

        # Filter: finite vacuity and not ignored.
        valid = np.isfinite(vc)
        if ignore:
            ig = np.zeros(NUM_COMMON, dtype=bool)
            for c in ignore:
                if 0 <= c < NUM_COMMON:
                    ig[c] = True
            valid &= ~ig[gt]
        gt = gt[valid]; pr = pr[valid]
        vc = np.clip(vc[valid], 0.0, vmax)

        b = np.searchsorted(edges, vc, side="right") - 1
        np.clip(b, 0, nbins - 1, out=b)

        cls_bin = gt * nbins + b
        tot_f = np.bincount(cls_bin, minlength=NUM_COMMON * nbins) \
                  .reshape(NUM_COMMON, nbins)
        ok = gt == pr
        cor_f = (np.bincount(cls_bin[ok], minlength=NUM_COMMON * nbins)
                   .reshape(NUM_COMMON, nbins)
                 if ok.any() else np.zeros((NUM_COMMON, nbins), dtype=np.int64))

        pf_totals[i]  = tot_f
        pf_correct[i] = cor_f
        totals  += tot_f
        correct += cor_f

        if (i + 1) % 500 == 0 or (i + 1) == len(files):
            print(f"  [{i+1:>5d}/{len(files)}] kept={int(totals.sum()):>12d}  "
                  f"elapsed={time.time()-t0:6.1f}s")
    return correct, totals, edges, pf_correct, pf_totals


def pick_major_classes(totals: np.ndarray, min_frac: float) -> list[int]:
    per_cls = totals.sum(axis=1)
    grand = per_cls.sum()
    if grand <= 0:
        return []
    keep = [c for c in range(NUM_COMMON) if per_cls[c] >= min_frac * grand]
    keep.sort(key=lambda c: -per_cls[c])
    return keep


def print_text_summary(
    correct: np.ndarray, totals: np.ndarray, edges: np.ndarray, major: list[int]
) -> None:
    nbins = totals.shape[1]
    print(f"\n{'bin_lo':>8s} {'bin_hi':>8s} {'n_total':>10s} {'acc_overall':>12s}",
          end="")
    for c in major:
        print(f"  acc[{c}:{COMMON_CLASSES[c][:8]}]", end="")
    print()
    print("-" * (40 + 20 * len(major)))
    tot_all = totals.sum(axis=0)
    cor_all = correct.sum(axis=0)
    for j in range(nbins):
        n = int(tot_all[j])
        oa = cor_all[j] / tot_all[j] if tot_all[j] > 0 else float("nan")
        print(f"{edges[j]:>8.3f} {edges[j+1]:>8.3f} {n:>10d} {oa:>12.4f}", end="")
        for c in major:
            tt = totals[c, j]
            ca = correct[c, j] / tt if tt > 0 else float("nan")
            print(f"  {ca:>14.4f}", end="")
        print()


def band_bounds(
    pf_corr: np.ndarray, pf_tot: np.ndarray, mode: str, pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin spread of per-frame accuracy.

    pf_corr / pf_tot: (nframes, nbins) correct / total counts. Returns
    (lo[nbins], hi[nbins]) clipped to [0, 1]; bins with no frames -> NaN.
      mode="pct": [pct, 100-pct] percentile band across frames.
      mode="std": mean +/- 1 std.   mode="sem": mean +/- std/sqrt(n).
    """
    nbins = pf_tot.shape[1]
    lo = np.full(nbins, np.nan)
    hi = np.full(nbins, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(pf_tot > 0, pf_corr / pf_tot, np.nan)
    for j in range(nbins):
        col = acc[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        if mode == "pct":
            lo[j] = np.percentile(col, pct)
            hi[j] = np.percentile(col, 100.0 - pct)
        else:
            m = col.mean()
            s = col.std()
            if mode == "sem":
                s = s / np.sqrt(col.size)
            lo[j] = m - s
            hi[j] = m + s
    np.clip(lo, 0.0, 1.0, out=lo)
    np.clip(hi, 0.0, 1.0, out=hi)
    return lo, hi


def make_plot(
    correct: np.ndarray, totals: np.ndarray, edges: np.ndarray,
    major: list[int], out_path: str, title_suffix: str,
    pf_correct: np.ndarray, pf_totals: np.ndarray,
    band_mode: str = "pct", band_pct: float = 25.0,
) -> None:
    nbins = totals.shape[1]
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, (ax_acc, ax_hist) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(major):
        tot = totals[c]
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(tot > 0, correct[c] / tot, np.nan)
        color = cmap(i % 10)
        if band_mode != "none":
            lo, hi = band_bounds(pf_correct[:, c, :], pf_totals[:, c, :],
                                 band_mode, band_pct)
            ax_acc.fill_between(centers, lo, hi, color=color, alpha=0.15,
                                linewidth=0)
        ax_acc.plot(
            centers, acc,
            label=f"{c} {COMMON_CLASSES[c]} ({int(tot.sum()):,})",
            color=color, linewidth=1.8, marker="o", markersize=4,
            alpha=0.9,
        )

    tot_all = totals.sum(axis=0)
    cor_all = correct.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc_all = np.where(tot_all > 0, cor_all / tot_all, np.nan)
    if band_mode != "none":
        lo, hi = band_bounds(pf_correct.sum(axis=1), pf_totals.sum(axis=1),
                             band_mode, band_pct)
        ax_acc.fill_between(centers, lo, hi, color="black", alpha=0.12,
                            linewidth=0)
    ax_acc.plot(
        centers, acc_all, label=f"OVERALL ({int(tot_all.sum()):,})",
        color="black", linewidth=2.8, marker="s", markersize=5,
    )

    ax_acc.set_ylabel("Accuracy (pred_common == gt_common)")
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_title(
        "EDL vacuity vs accuracy — cenet_mcd_relu_mse_W10_kl0p01 on KITTI-360 "
        f"seq 0009 (common 9-class){title_suffix}"
    )
    band_desc = {
        "pct": f"{band_pct:.0f}-{100 - band_pct:.0f} pct band",
        "std": "mean +/- 1 std band",
        "sem": "mean +/- SEM band",
        "none": "",
    }[band_mode]
    legend_title = "common class (n points)"
    if band_desc:
        legend_title += f"\nshaded = per-frame {band_desc}"
    ax_acc.legend(
        loc="upper right", fontsize=9, ncol=1, framealpha=0.9,
        title=legend_title,
    )

    widths = edges[1:] - edges[:-1]
    ax_hist.bar(
        centers, tot_all, width=widths * 0.95,
        color="#3b7dd8", edgecolor="black", linewidth=0.4,
    )
    ax_hist.set_yscale("log")
    ax_hist.set_ylabel("# points (log)")
    ax_hist.set_xlabel("EDL vacuity  K/S   (0 = certain, higher = more uncertain)")
    ax_hist.grid(True, alpha=0.3, which="both")
    ax_hist.set_xlim(edges[0], edges[-1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved: {out_path}")


def make_class_plot(
    correct: np.ndarray, totals: np.ndarray, edges: np.ndarray,
    c: int, out_path: str, title_suffix: str,
    pf_correct: np.ndarray, pf_totals: np.ndarray,
    band_mode: str = "pct", band_pct: float = 25.0,
) -> None:
    """Single-class figure: that class's pooled accuracy + per-frame spread
    band, with the overall line as a faint reference and a histogram of the
    class's own per-bin point counts."""
    nbins = totals.shape[1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    name = COMMON_CLASSES[c]

    fig, (ax_acc, ax_hist) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    tot = totals[c]
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(tot > 0, correct[c] / tot, np.nan)

    # Faint overall reference line.
    tot_all = totals.sum(axis=0)
    cor_all = correct.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc_all = np.where(tot_all > 0, cor_all / tot_all, np.nan)
    ax_acc.plot(
        centers, acc_all, label=f"OVERALL ({int(tot_all.sum()):,})",
        color="0.6", linewidth=1.5, linestyle="--", alpha=0.8,
    )

    color = "#1f77b4"
    if band_mode != "none":
        lo, hi = band_bounds(pf_correct[:, c, :], pf_totals[:, c, :],
                             band_mode, band_pct)
        ax_acc.fill_between(centers, lo, hi, color=color, alpha=0.20,
                            linewidth=0)
    ax_acc.plot(
        centers, acc,
        label=f"{c} {name} ({int(tot.sum()):,})",
        color=color, linewidth=2.4, marker="o", markersize=4.5,
    )

    band_desc = {
        "pct": f"{band_pct:.0f}-{100 - band_pct:.0f} pct band",
        "std": "mean +/- 1 std band",
        "sem": "mean +/- SEM band",
        "none": "",
    }[band_mode]
    legend_title = "common class (n points)"
    if band_desc:
        legend_title += f"\nshaded = per-frame {band_desc}"

    ax_acc.set_ylabel("Accuracy (pred_common == gt_common)")
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_title(
        f"EDL vacuity vs accuracy — {name} — cenet_mcd_relu_mse_W10_kl0p01 "
        f"on KITTI-360 seq 0009{title_suffix}"
    )
    ax_acc.legend(loc="upper right", fontsize=9, framealpha=0.9,
                  title=legend_title)

    widths = edges[1:] - edges[:-1]
    ax_hist.bar(
        centers, tot, width=widths * 0.95,
        color=color, edgecolor="black", linewidth=0.4,
    )
    ax_hist.set_yscale("log")
    ax_hist.set_ylabel(f"# {name} points (log)")
    ax_hist.set_xlabel("EDL vacuity  K/S   (0 = certain, higher = more uncertain)")
    ax_hist.grid(True, alpha=0.3, which="both")
    ax_hist.set_xlim(edges[0], edges[-1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-dir", default=KITTI360_SEQ)
    p.add_argument("--gt-name", default="gt_labels",
                   help="GT folder under seq (KITTI-360 raw label space).")
    p.add_argument("--edl-subdir", default=EDL_SUBDIR,
                   help="EDL model subdir under seq (MCD raw label space).")
    p.add_argument("--vac-subdir", default="confidence_scores",
                   help="Vacuity folder under --edl-subdir.")
    p.add_argument("--labels-common", default=LABELS_COMMON_YAML)
    p.add_argument("--pred-lut", default="mcd",
                   choices=["mcd", "kitti360", "semkitti"],
                   help="Which *_to_common LUT to apply to predictions "
                        "(EDL model was MCD-trained -> 'mcd').")
    p.add_argument("--gt-lut", default="kitti360",
                   choices=["mcd", "kitti360", "semkitti"],
                   help="LUT to apply to GT.")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--vmax", type=float, default=None,
                   help="Upper edge for vacuity binning. Auto from data if omitted.")
    p.add_argument("--min-frac", type=float, default=0.005,
                   help="Min GT fraction (in common space) for a class to be plotted.")
    p.add_argument("--ignore", type=int, nargs="*", default=[0],
                   help="Common class ids to ignore in GT (default: [0] unlabeled).")
    p.add_argument("--band", default="pct", choices=["pct", "std", "sem", "none"],
                   help="Spread band behind each accuracy line: percentile band "
                        "(pct), mean+/-std (std), mean+/-SEM (sem), or off (none).")
    p.add_argument("--band-pct", type=float, default=25.0,
                   help="Lower percentile for --band pct (band = [p, 100-p]).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="edl_vacuity_vs_accuracy_kitti360_0009.png")
    p.add_argument("--per-class", action="store_true",
                   help="Also save one figure per major class next to --out, "
                        "named <out_stem>_<id>_<class>.png.")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    luts = load_common_lut(args.labels_common)
    pred_lut = luts[args.pred_lut]
    gt_lut = luts[args.gt_lut]
    print(f"Pred LUT '{args.pred_lut}' size={pred_lut.size}, "
          f"targets={sorted(set(pred_lut.tolist()))}")
    print(f"GT   LUT '{args.gt_lut}' size={gt_lut.size}, "
          f"targets={sorted(set(gt_lut.tolist()))}")

    gt_dir   = os.path.join(args.seq_dir, args.gt_name)
    edl_dir  = os.path.join(args.seq_dir, args.edl_subdir)
    vac_dir  = os.path.join(edl_dir, args.vac_subdir)
    for d in (gt_dir, edl_dir, vac_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"Missing directory: {d}")

    gt_files = set(os.listdir(gt_dir))
    pr_files = {f for f in os.listdir(edl_dir) if f.endswith(".bin")}
    vc_files = set(os.listdir(vac_dir))
    common = sorted(gt_files & pr_files & vc_files)
    if args.limit:
        common = common[: args.limit]
    if not common:
        raise SystemExit("No frames common to GT / preds / vacuity.")
    print(f"\nMatching frames: {len(common)} "
          f"(gt={len(gt_files)}, pred={len(pr_files)}, vac={len(vc_files)})")
    print(f"Ignored common classes: {sorted(args.ignore) if args.ignore else 'none'}")

    if args.vmax is None:
        sample = np.fromfile(os.path.join(vac_dir, common[0]), dtype=np.float16) \
                    .astype(np.float32)
        vmax = float(np.quantile(sample[np.isfinite(sample)], 0.999))
        vmax = max(vmax, 0.05)
        vmax = min(vmax * 1.1, 1.0)
        print(f"Auto vmax = {vmax:.4f} (99.9th pct of first scan, capped at 1.0)")
    else:
        vmax = args.vmax

    print(f"\n--- accumulating {args.bins} vacuity bins over [0, {vmax:.4f}] ---")
    correct, totals, edges, pf_correct, pf_totals = accumulate(
        gt_dir, edl_dir, vac_dir, common,
        pred_lut, gt_lut, args.bins, vmax, set(args.ignore),
    )
    grand = int(totals.sum())
    print(f"total kept points: {grand}")
    if grand == 0:
        raise SystemExit("No points survived filtering.")

    major = pick_major_classes(totals, args.min_frac)
    # Always include terrain if present, even below threshold.
    if totals[TERRAIN_COMMON].sum() > 0 and TERRAIN_COMMON not in major:
        major.append(TERRAIN_COMMON)
    print(f"Major common classes (>= {args.min_frac*100:.2f}% GT + terrain): "
          f"{[(c, COMMON_CLASSES[c]) for c in major]}")

    print_text_summary(correct, totals, edges, major)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["bin_lo", "bin_hi", "n_total", "acc_overall"]
            for c in major:
                header += [f"n_cls_{c}", f"acc_cls_{c}_{COMMON_CLASSES[c]}"]
            w.writerow(header)
            tot_all = totals.sum(axis=0)
            cor_all = correct.sum(axis=0)
            for j in range(args.bins):
                row = [
                    float(edges[j]), float(edges[j + 1]),
                    int(tot_all[j]),
                    float(cor_all[j] / tot_all[j]) if tot_all[j] else float("nan"),
                ]
                for c in major:
                    tt = totals[c, j]
                    row += [
                        int(tt),
                        float(correct[c, j] / tt) if tt else float("nan"),
                    ]
                w.writerow(row)
        print(f"wrote CSV: {args.csv}")

    title_suffix = f"  (n={len(common)} frames, {grand:,} pts)"
    make_plot(correct, totals, edges, major, args.out, title_suffix,
              pf_correct, pf_totals, band_mode=args.band, band_pct=args.band_pct)

    if args.per_class:
        stem, ext = os.path.splitext(args.out)
        for c in major:
            cls_path = f"{stem}_{c}_{COMMON_CLASSES[c]}{ext or '.png'}"
            make_class_plot(
                correct, totals, edges, c, cls_path, title_suffix,
                pf_correct, pf_totals,
                band_mode=args.band, band_pct=args.band_pct,
            )


if __name__ == "__main__":
    main()
