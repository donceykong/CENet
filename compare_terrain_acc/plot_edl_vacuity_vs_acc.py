"""
Plot per-class accuracy vs EDL vacuity for cenet_mcd_relu_mse_W10_kl0p01.

Layout (one figure, two stacked panels sharing x-axis):

  Top:    accuracy per vacuity bin -- one line per "major" class
          (i.e. classes with GT presence >= --min-frac of all kept points),
          plus a thick black "overall" line. Each line is the *pooled*
          accuracy; behind it a translucent "error band" (a.k.a. confidence
          band / fan chart) shows the per-frame spread of accuracy in that
          bin (--band: pct/std/sem).

  Bottom: histogram of point counts per vacuity bin (log y).

Pass --per-class to also save one figure per major class next to --out.

Vacuity comes from <EDL>/confidence_scores/<name>.bin as float16 — despite
the directory name, the value stored is K/S (K=30, S=sum of Dirichlet
alphas), i.e. EDL epistemic uncertainty: 0 = certain, 1 = no evidence.

GT labels come from gt_labels_terrain (MCD 30-class scheme; vegetation
under terrain is relabeled 25 -> 29). Predictions come from <EDL>/*.bin
(uint32 per point, MCD scheme).

Run:
  python3 plot_edl_vacuity_vs_acc.py
  python3 plot_edl_vacuity_vs_acc.py --bins 30 --min-frac 0.005 --limit 500
  python3 plot_edl_vacuity_vs_acc.py --per-class --band pct --band-pct 25
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Keep in sync with config/datasets/labels/labels_mcd.yaml
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

SEQ_DIR  = "/media/donceykong/donceys_data_ssd/datasets/mcd/kth/kth_day_09"
GT_DIR   = os.path.join(SEQ_DIR, "gt_labels_terrain")
EDL_DIR  = os.path.join(SEQ_DIR, "inferred_labels/cenet_mcd_relu_mse_W10_kl0p01")
CONF_DIR = os.path.join(EDL_DIR, "confidence_scores")  # actually vacuity


def accumulate(
    gt_dir: str,
    edl_dir: str,
    vac_dir: str,
    files: list[str],
    nbins: int,
    vmax: float,
    ignore: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Streams files; returns:
      correct_hist[NUM_CLASSES, nbins]  # correct points by (gt_class, bin)
      total_hist  [NUM_CLASSES, nbins]  # total points   by (gt_class, bin)
      bin_edges   [nbins+1]
      pf_correct  [F, NUM_CLASSES, nbins]  # per-frame correct counts
      pf_totals   [F, NUM_CLASSES, nbins]  # per-frame total counts
    The per-frame (pf_*) arrays let the plot show a per-frame spread band.
    """
    edges = np.linspace(0.0, vmax, nbins + 1, dtype=np.float64)
    correct = np.zeros((NUM_CLASSES, nbins), dtype=np.int64)
    totals  = np.zeros((NUM_CLASSES, nbins), dtype=np.int64)
    nframes = len(files)
    pf_correct = np.zeros((nframes, NUM_CLASSES, nbins), dtype=np.int32)
    pf_totals  = np.zeros((nframes, NUM_CLASSES, nbins), dtype=np.int32)

    t0 = time.time()
    for i, fname in enumerate(files):
        gt = np.fromfile(os.path.join(gt_dir,  fname), dtype=np.uint32)
        pr = np.fromfile(os.path.join(edl_dir, fname), dtype=np.uint32)
        vc = np.fromfile(os.path.join(vac_dir, fname), dtype=np.float16).astype(np.float32)
        if not (gt.shape == pr.shape == vc.shape):
            raise ValueError(
                f"shape mismatch on {fname}: gt={gt.shape} pred={pr.shape} vac={vc.shape}"
            )

        # Defensive filter + ignore list.
        valid = (gt < NUM_CLASSES) & (pr < NUM_CLASSES) & np.isfinite(vc)
        if ignore:
            ig = np.zeros(NUM_CLASSES, dtype=bool)
            for c in ignore:
                if 0 <= c < NUM_CLASSES:
                    ig[c] = True
            valid &= ~ig[gt.clip(max=NUM_CLASSES - 1)]
        gt = gt[valid].astype(np.int64, copy=False)
        pr = pr[valid].astype(np.int64, copy=False)
        vc = np.clip(vc[valid], 0.0, vmax)

        # Bin assignment in [0, nbins-1].
        b = np.searchsorted(edges, vc, side="right") - 1
        np.clip(b, 0, nbins - 1, out=b)

        # 2-D bincount via flattening: (class * nbins + bin).
        cls_bin = gt * nbins + b
        tot_f = np.bincount(cls_bin, minlength=NUM_CLASSES * nbins) \
                  .reshape(NUM_CLASSES, nbins)
        ok = gt == pr
        cor_f = (np.bincount(cls_bin[ok], minlength=NUM_CLASSES * nbins)
                   .reshape(NUM_CLASSES, nbins)
                 if ok.any() else np.zeros((NUM_CLASSES, nbins), dtype=np.int64))

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
    keep = [c for c in range(NUM_CLASSES) if per_cls[c] >= min_frac * grand]
    keep.sort(key=lambda c: -per_cls[c])
    return keep


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
    correct: np.ndarray,
    totals: np.ndarray,
    edges: np.ndarray,
    major: list[int],
    out_path: str,
    title_suffix: str,
    pf_correct: np.ndarray,
    pf_totals: np.ndarray,
    band_mode: str = "pct",
    band_pct: float = 25.0,
) -> None:
    nbins = totals.shape[1]
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, (ax_acc, ax_hist) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # ---- Top: per-class accuracy vs vacuity --------------------------
    # Color via a perceptually-ordered colormap so neighbouring class
    # ids don't end up the same hue.
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(major):
        tot = totals[c]
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(tot > 0, correct[c] / tot, np.nan)
        color = cmap(i % 20)
        if band_mode != "none":
            lo, hi = band_bounds(pf_correct[:, c, :], pf_totals[:, c, :],
                                 band_mode, band_pct)
            ax_acc.fill_between(centers, lo, hi, color=color, alpha=0.15,
                                linewidth=0)
        # Skip bins with tiny support so the line doesn't jitter.
        ax_acc.plot(
            centers, acc,
            label=f"{c} {MCD_LABELS[c]} ({int(tot.sum()):,})",
            color=color, linewidth=1.6, marker="o", markersize=3,
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
        color="black", linewidth=2.6, marker="s", markersize=4,
    )

    band_desc = {
        "pct": f"{band_pct:.0f}-{100 - band_pct:.0f} pct band",
        "std": "mean +/- 1 std band",
        "sem": "mean +/- SEM band",
        "none": "",
    }[band_mode]
    legend_title = "class (n points)"
    if band_desc:
        legend_title += f"\nshaded = per-frame {band_desc}"

    ax_acc.set_ylabel("Accuracy (pred == gt)")
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_title(
        f"EDL vacuity (K/S) vs accuracy — cenet_mcd_relu_mse_W10_kl0p01{title_suffix}"
    )
    ax_acc.legend(
        loc="upper right", fontsize=8, ncol=2, framealpha=0.9,
        title=legend_title,
    )

    # ---- Bottom: histogram of points per vacuity bin -----------------
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
    correct: np.ndarray,
    totals: np.ndarray,
    edges: np.ndarray,
    c: int,
    out_path: str,
    title_suffix: str,
    pf_correct: np.ndarray,
    pf_totals: np.ndarray,
    band_mode: str = "pct",
    band_pct: float = 25.0,
) -> None:
    """Single-class figure: that class's pooled accuracy + per-frame spread
    band, with the overall line as a faint reference and a histogram of the
    class's own per-bin point counts."""
    nbins = totals.shape[1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    name = MCD_LABELS[c]

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
    legend_title = "class (n points)"
    if band_desc:
        legend_title += f"\nshaded = per-frame {band_desc}"

    ax_acc.set_ylabel("Accuracy (pred == gt)")
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_title(
        f"EDL vacuity (K/S) vs accuracy — {name} — "
        f"cenet_mcd_relu_mse_W10_kl0p01{title_suffix}"
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


def print_text_summary(
    correct: np.ndarray, totals: np.ndarray, edges: np.ndarray, major: list[int]
) -> None:
    nbins = totals.shape[1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    print(f"\n{'bin_lo':>8s} {'bin_hi':>8s} {'n_total':>10s} {'acc_overall':>12s}",
          end="")
    for c in major:
        print(f"  acc[{c}:{MCD_LABELS[c][:10]}]", end="")
    print()
    print("-" * (40 + 22 * len(major)))
    tot_all = totals.sum(axis=0)
    cor_all = correct.sum(axis=0)
    for j in range(nbins):
        n = int(tot_all[j])
        oa = cor_all[j] / tot_all[j] if tot_all[j] > 0 else float("nan")
        print(f"{edges[j]:>8.3f} {edges[j+1]:>8.3f} {n:>10d} {oa:>12.4f}", end="")
        for c in major:
            tt = totals[c, j]
            ca = correct[c, j] / tt if tt > 0 else float("nan")
            print(f"  {ca:>16.4f}", end="")
        print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir",  default=GT_DIR)
    p.add_argument("--edl-dir", default=EDL_DIR,
                   help="Top-level dir with uint32 prediction .bin files.")
    p.add_argument("--vac-dir", default=CONF_DIR,
                   help="Vacuity dir (float16 per point). Defaults to "
                        "<edl-dir>/confidence_scores.")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--vmax", type=float, default=None,
                   help="Upper edge for vacuity binning. Auto-detected from "
                        "data if omitted (99th percentile of first scan).")
    p.add_argument("--min-frac", type=float, default=0.01,
                   help="Min fraction of total GT points for a class to be "
                        "considered 'major' and plotted.")
    p.add_argument("--ignore", type=int, nargs="*", default=[],
                   help="GT class ids to drop from the analysis (e.g. 0 11).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of frames.")
    p.add_argument("--band", default="pct", choices=["pct", "std", "sem", "none"],
                   help="Spread band behind each accuracy line: percentile band "
                        "(pct), mean+/-std (std), mean+/-SEM (sem), or off (none).")
    p.add_argument("--band-pct", type=float, default=25.0,
                   help="Lower percentile for --band pct (band = [p, 100-p]).")
    p.add_argument("--per-class", action="store_true",
                   help="Also save one figure per major class next to --out, "
                        "named <out_stem>_<id>_<class>.png.")
    p.add_argument("--out", default="edl_vacuity_vs_accuracy.png")
    p.add_argument("--csv", default=None,
                   help="Optional CSV path to dump per-bin per-class accuracy.")
    args = p.parse_args()

    gt_files  = set(os.listdir(args.gt_dir))
    pred_files = {f for f in os.listdir(args.edl_dir) if f.endswith(".bin")}
    vac_files = set(os.listdir(args.vac_dir))
    common = sorted(gt_files & pred_files & vac_files)
    if args.limit:
        common = common[: args.limit]
    if not common:
        raise SystemExit("No common files between GT / EDL preds / vacuity.")
    print(f"Matching frames: {len(common)}  "
          f"(gt={len(gt_files)}, pred={len(pred_files)}, vac={len(vac_files)})")
    print(f"Ignored GT classes: {sorted(args.ignore) if args.ignore else 'none'}")

    # Auto-detect vmax if needed.
    if args.vmax is None:
        sample = np.fromfile(
            os.path.join(args.vac_dir, common[0]), dtype=np.float16,
        ).astype(np.float32)
        vmax = float(np.quantile(sample[np.isfinite(sample)], 0.999))
        # Round up to a clean value.
        vmax = max(vmax, 0.05)
        vmax = min(vmax * 1.1, 1.0)
        print(f"Auto vmax = {vmax:.4f} (from 99.9th pct of first scan, capped at 1.0)")
    else:
        vmax = args.vmax

    print(f"\n--- accumulating {args.bins} vacuity bins over [0, {vmax:.4f}] ---")
    correct, totals, edges, pf_correct, pf_totals = accumulate(
        args.gt_dir, args.edl_dir, args.vac_dir, common,
        args.bins, vmax, set(args.ignore),
    )
    grand = int(totals.sum())
    print(f"total kept points: {grand}")
    if grand == 0:
        raise SystemExit("No points survived filtering.")

    major = pick_major_classes(totals, args.min_frac)
    if TERRAIN_CLS in [c for c in range(NUM_CLASSES) if totals[c].sum() > 0] \
            and TERRAIN_CLS not in major:
        major.append(TERRAIN_CLS)
    print(f"Major classes (>={args.min_frac*100:.2f}% of GT, plus terrain): "
          f"{[(c, MCD_LABELS[c]) for c in major]}")

    print_text_summary(correct, totals, edges, major)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["bin_lo", "bin_hi", "n_total", "acc_overall"]
            for c in major:
                header += [f"n_cls_{c}", f"acc_cls_{c}_{MCD_LABELS[c]}"]
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
            cls_path = f"{stem}_{c}_{MCD_LABELS[c]}{ext or '.png'}"
            make_class_plot(
                correct, totals, edges, c, cls_path, title_suffix,
                pf_correct, pf_totals,
                band_mode=args.band, band_pct=args.band_pct,
            )


if __name__ == "__main__":
    main()
