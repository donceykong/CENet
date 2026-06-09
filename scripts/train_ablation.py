#!/usr/bin/env python3
"""EDL ablation launcher: evidence-act x loss-form x KL-warmup x KL-strength.

Sweeps the single-stage 512p EDL model over the cross-product of four axes,
holding everything else (split, seed, LR schedule, batch size) fixed:

    unc_act          in {exp, relu, softplus}  # evidence activation; default exp only
    unc_type         in {log, digamma, mse}    # mse = paper Eq.5, reported as most stable
    kl_warmup_epochs in {10, null}      # 10 = paper's min(1,epoch/10); null = anneal over max_epochs
    kl_strength      in {0.05,0.1,0.5}  # KL weight; MSE's loss is ~3x smaller so the
                                        #   same kl_strength bites ~3x harder than for log

Each cell trains FROM SCRATCH into its own run dir so runs never clobber, and
logs to Aim with unc_act / unc_type / kl_warmup_epochs / kl_strength hparams for
side-by-side compare. The calibration figure (reliability + rejection curve) is
produced on the final epoch of each run and uploaded to Aim.

Cell names encode all four axes, e.g. `exp_mse_W10_kl0p1` (dots -> 'p').

Usage:
    python scripts/train_ablation.py                          # default grid, full epochs
    python scripts/train_ablation.py --epochs 1               # 1-epoch smoke test
    python scripts/train_ablation.py --kl-strengths 0.1       # pin one KL value
    python scripts/train_ablation.py --unc-types mse          # only the MSE half
    python scripts/train_ablation.py --unc-acts exp relu softplus  # sweep activation
    python scripts/train_ablation.py --cells exp_mse_W10_kl0p1     # explicit cell subset
    python scripts/train_ablation.py --dry-run               # resolve + print, no training
"""

import argparse
import copy
import os
import sys
from collections import defaultdict

# ce_net imports (repo root) + sibling train.py helpers (scripts/)
_HERE = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(_HERE, "..")))
sys.path.append(_HERE)

from ce_net import CONFIG_DIR
from ce_net.models.trainer_wb import Trainer
from ce_net.core.parsers.mcd import get_keyframe_scans as _mcd_keyframes
from ce_net.core.parsers.cumulti import get_keyframe_scans as _cumulti_keyframes
from ce_net.core.parsers.kitti360 import get_keyframe_scans as _kitti360_keyframes
from train import (  # sibling scripts/train.py
    load_yaml,
    parse_yaml,
    _resolve_dataset,
    _persist_run_artifacts,
)

# Ablation axes.
UNC_ACTS = ["exp", "relu", "softplus"]  # evidence activation (loss reads unc_act)
UNC_TYPES = ["log", "digamma", "mse"]   # loss form (mse = paper Eq.5)
WARMUPS = {"W10": 10, "Wmax": None}  # cell-name key -> kl_warmup_epochs value
DEFAULT_KL_STRENGTHS = [0.05, 0.1, 0.5]
# Default to exp-only so adding the unc_act axis doesn't silently 3x the grid;
# pass --unc-acts exp relu softplus to sweep it.
DEFAULT_UNC_ACTS = ["exp"]
DEFAULT_UNC_TYPES = ["log", "mse"]

# Per-dataset keyframe selectors (only MCD is implemented; others raise).
_KEYFRAME_FNS = {
    "MCD": _mcd_keyframes,
    "CU-MULTI": _cumulti_keyframes,
    "KITTI-360": _kitti360_keyframes,
}


def _seq_root(scan_path):
    """Sequence root that owns one pose file. All datasets store scans at
    <seq_root>/<subdir>/data/<stem>.<ext>, so it's three levels up."""
    return os.path.dirname(os.path.dirname(os.path.dirname(scan_path)))


def apply_keyframe_subset(train_shards, dataset_name, keyframe_dist, perc):
    """Replace each train shard's scans with a per-sequence, proportional,
    keyframe-spaced subset (valid/test are left untouched by the caller).

    The split pools+shuffles whole sequences into a shard, so we first regroup
    each shard's scans by sequence root, then hand each sequence to the
    dataset's `get_keyframe_scans` (which reads that dataset's pose format).
    Operating on the train portion only avoids val/test leakage; the dropped
    val/test frames just leave small temporal gaps the spacing rule tolerates.
    """
    if dataset_name not in _KEYFRAME_FNS:
        raise SystemExit(f"No keyframe selector registered for dataset '{dataset_name}'.")
    kf_fn = _KEYFRAME_FNS[dataset_name]

    new_shards = []
    for shard in train_shards:
        by_seq = defaultdict(lambda: ([], []))
        for scan, label in zip(shard["scan_files"], shard["label_files"]):
            sc, lb = by_seq[_seq_root(scan)]
            sc.append(scan)
            lb.append(label)

        sel_scans, sel_labels = [], []
        for seq_root, (sc, lb) in sorted(by_seq.items()):
            ss, ll = kf_fn(sc, lb, keyframe_dist, perc)
            sel_scans.extend(ss)
            sel_labels.extend(ll)
            print(f"    {os.path.basename(seq_root):16s} {len(sc):6d} -> {len(ss):5d}")
        new_shards.append({**shard, "scan_files": sel_scans, "label_files": sel_labels})
    return new_shards


def _kl_key(kl):
    # 0.05 -> "kl0p05", 0.1 -> "kl0p1", 0.5 -> "kl0p5"
    return ("kl%g" % kl).replace(".", "p")


def build_cells(unc_acts, unc_types, warmup_keys, kl_strengths):
    """Cross-product of the four axes -> {cell_name: overrides}."""
    cells = {}
    for act in unc_acts:
        for unc in unc_types:
            for wkey in warmup_keys:
                for kl in kl_strengths:
                    name = f"{act}_{unc}_{wkey}_{_kl_key(kl)}"
                    cells[name] = {
                        "unc_act": act,
                        "unc_type": unc,
                        "kl_warmup_epochs": WARMUPS[wkey],
                        "kl_strength": kl,
                    }
    return cells


def seed_torch(seed=1024):
    import random as _random
    import numpy as _np
    import torch as _torch

    _random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed(seed)
    print(f"Seeded RNGs with {seed}")


def _build_flags(base_config, model_cfg):
    # parse_yaml reads sys.argv via parse_known_args; neutralise it so this
    # script's own flags (--epochs/--cells/...) don't leak into it.
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        return parse_yaml(base_config, model_cfg)
    finally:
        sys.argv = saved


def main():
    ap = argparse.ArgumentParser("./train_ablation.py")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override max_epochs for every cell (e.g. 1 for a smoke test).")
    ap.add_argument("--unc-acts", nargs="*", choices=UNC_ACTS, default=DEFAULT_UNC_ACTS,
                    help="evidence-activation axis. Default: exp. "
                         "Pass 'exp relu softplus' to sweep it.")
    ap.add_argument("--unc-types", nargs="*", choices=UNC_TYPES, default=DEFAULT_UNC_TYPES,
                    help="loss-form axis. Default: log mse (digamma also available).")
    ap.add_argument("--warmups", nargs="*", choices=list(WARMUPS), default=list(WARMUPS),
                    help="KL-warmup axis (cell-name keys). Default: W10 Wmax.")
    ap.add_argument("--kl-strengths", nargs="*", type=float, default=DEFAULT_KL_STRENGTHS,
                    help=f"KL-strength axis. Default: {DEFAULT_KL_STRENGTHS}.")
    ap.add_argument("--cells", nargs="*", default=None,
                    help="Explicit cell-name subset (e.g. mse_W10_kl0p1). "
                         "Overrides the axis flags when given.")
    ap.add_argument("--perc-scans-to-use", type=float, default=None,
                    help="Keyframe-subset the TRAIN set to this fraction of each "
                         "sequence (proportional). Omit to train on the full split.")
    ap.add_argument("--keyframe-dist", type=float, default=1.0,
                    help="Min spacing in metres between kept train scans "
                         "(only used when --perc-scans-to-use is set). Default 1.0.")
    ap.add_argument("--ablation-root", type=str, default=None,
                    help="Output root. Default: <training.model_path>__ablation.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve configs and print the plan without training.")
    args = ap.parse_args()

    if args.perc_scans_to_use is not None and not (0 < args.perc_scans_to_use <= 1.0):
        raise SystemExit("--perc-scans-to-use must be in (0, 1].")

    # Build the grid from the axes, then optionally filter to an explicit list.
    all_cells = build_cells(args.unc_acts, args.unc_types, args.warmups, args.kl_strengths)
    if args.cells:
        unknown = [c for c in args.cells if c not in all_cells]
        if unknown:
            raise SystemExit(
                f"Unknown cell(s): {unknown}\nAvailable from current axes: "
                f"{list(all_cells)}"
            )
        cells = {c: all_cells[c] for c in args.cells}
    else:
        cells = all_cells

    seed_torch(1024)

    base_config = load_yaml(CONFIG_DIR / "training_mcd_EDL.yaml")
    dataset_name = base_config["dataset_name"]
    stage = base_config["training"]["model_name"]  # e.g. cenet_mcd-512-EDL
    model_configs = base_config["training"]["model_configs"]
    if stage not in model_configs:
        raise KeyError(
            f"training.model_name '{stage}' is not in training.model_configs."
        )
    base_model_cfg = model_configs[stage]

    ablation_root = args.ablation_root or (
        base_config["training"]["model_path"].rstrip("/") + "__ablation"
    )
    print(f"\nAblation stage: {stage}")
    print(f"Ablation root:  {ablation_root}")
    print(f"Grid:           {len(cells)} cells -> {list(cells)}")
    if args.epochs is not None:
        print(f"max_epochs override: {args.epochs}")
    if args.perc_scans_to_use is not None:
        print(f"keyframe subset:     perc={args.perc_scans_to_use} dist={args.keyframe_dist}m")

    for cell, overrides in cells.items():
        print(f"\n===== CELL {cell}: {overrides} =====")

        model_cfg = copy.deepcopy(base_model_cfg)
        model_cfg["model_name"] = f"{stage}-{cell}"
        model_cfg["pretrained_model_path"] = None  # always from scratch

        FLAGS = _build_flags(base_config, model_cfg)
        FLAGS.pretrained = None
        FLAGS.name = cell
        FLAGS.log = os.path.join(ablation_root, cell)

        ARCH, DATA = _resolve_dataset(FLAGS, base_config, dataset_name)

        # Apply the cell's evidential overrides onto the resolved ARCH.
        ARCH["train"]["evidential_loss"] = True
        ev = ARCH["train"].setdefault("evidential", {})
        ev["unc_act"] = overrides["unc_act"]
        ev["unc_type"] = overrides["unc_type"]
        ev["kl_warmup_epochs"] = overrides["kl_warmup_epochs"]
        ev["kl_strength"] = overrides["kl_strength"]
        if args.epochs is not None:
            ARCH["train"]["max_epochs"] = args.epochs

        print(
            f"  unc_act={ev['unc_act']}  unc_type={ev['unc_type']}  kl_warmup_epochs={ev['kl_warmup_epochs']}  "
            f"kl_strength={ev.get('kl_strength')}  max_epochs={ARCH['train']['max_epochs']}  "
            f"batch_size={ARCH['train']['batch_size']}"
        )
        print(f"  log dir: {FLAGS.log}")

        # Keyframe-subset the TRAIN split (proportional, per sequence). valid/
        # test stay as the normal split. Deterministic, so every cell trains on
        # the identical subset.
        if args.perc_scans_to_use is not None:
            before = sum(len(s["scan_files"]) for s in DATA["split"]["train"])
            print(f"  keyframe subset (dist={args.keyframe_dist}m perc={args.perc_scans_to_use}):")
            DATA["split"]["train"] = apply_keyframe_subset(
                DATA["split"]["train"], dataset_name,
                args.keyframe_dist, args.perc_scans_to_use,
            )
            after = sum(len(s["scan_files"]) for s in DATA["split"]["train"])
            print(f"  train scans: {before} -> {after}")
            # Persisted into arch_cfg.yaml and logged to Aim hparams for provenance.
            ARCH["train"]["keyframe_subset"] = {
                "keyframe_dist": args.keyframe_dist,
                "perc_scans_to_use": args.perc_scans_to_use,
            }

        if args.dry_run:
            print("  [dry-run] config resolved OK; skipping training.")
            continue

        # _persist_run_artifacts rmtree's FLAGS.log first; each cell has its
        # own dir so this is safe and gives a clean run.
        _persist_run_artifacts(FLAGS, ARCH, DATA, model_cfg)
        trainer = Trainer(
            ARCH, DATA, dataset_name, FLAGS.dataset, FLAGS.log, FLAGS.pretrained
        )
        trainer.train()

    print("\nAblation launcher finished.")


if __name__ == "__main__":
    main()
