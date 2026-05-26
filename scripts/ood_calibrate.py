#!/usr/bin/env python3
"""Phase 1 (offline): calibrate Mahalanobis OOD statistics for CENet.

Runs the in-distribution (ID) training split through the trained model, taps
the penultimate feature map (``model.conv_2``, [B, 128, H, W]), and computes:

  * a per-class mean   mu_c            ([num_classes, 128])
  * a shared (tied) inverse covariance Sigma^-1   ([128, 128])

These are written to ``<model_dir>/ood_mahalanobis.pt`` for ood_detect.py.

Config sources mirror scripts/train.py: paths come from config/training_mcd.yaml
and the per-run configs saved in the model dir at train time
(arch_cfg.yaml, data_cfg.yaml, model_config.yaml). The split is reproduced
with the same seed (1024) used during training, so calibration sees exactly
the training train-split -- with augmentation disabled.

Example:
    python scripts/ood_calibrate.py \
        --model /media/.../mcd_terrain_ntu_tuhh/cenet_mcd-1024 \
        --max-batches 500
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ce_net import CONFIG_DIR
from ce_net.core.parsers.mcd import split_mcd_sensor_groups
from ce_net.core.parsers.parser import _build_mcd_concat
from ce_net.utils.sensor import materialize_sensor_groups

from ood_common import (
    STATS_FILENAME,
    build_model,
    ignored_class_ids,
    load_yaml,
    num_classes,
    save_stats,
    valid_pixel_mask,
    PenultimateHook,
)


def build_calibration_loader(ARCH, DATA, dataset_path, img_width, img_height):
    """Labeled, un-augmented DataLoader over the MCD training split.

    Reuses train.py's deterministic per-sensor split (seed=1024) so the ID
    statistics come from the same scans the model was trained on. Unlike the
    Parser's train loader, augmentation is OFF (transform=False) -- we want
    clean ID feature statistics, not jittered ones.
    """
    if DATA.get("dataset_name", "MCD") != "MCD" and "sensor_groups" not in DATA:
        raise NotImplementedError(
            "ood_calibrate.py currently supports MCD (sensor_groups). For other "
            "datasets, build a labeled (gt=True) loader for their train split."
        )

    groups = materialize_sensor_groups(
        DATA["sensor_groups"], img_width=img_width, img_height=img_height
    )
    split_ratios = DATA.get("split", [0.8, 0.1, 0.1])
    if not isinstance(split_ratios, list):
        raise ValueError("data_cfg.split must be a [train, valid, test] ratio list.")
    split = split_mcd_sensor_groups(dataset_path, groups, split_ratios, seed=1024)
    train_shards = split["train"]

    n_scans = sum(len(s["scan_files"]) for s in train_shards)
    print(f"Calibration train scans: {n_scans}")
    if n_scans == 0:
        raise RuntimeError("No training scans resolved; check dataset_path / sensor_groups.")

    train_dataset = _build_mcd_concat(
        train_shards,
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        max_points=ARCH["dataset"]["max_points"],
        gt=True,
        transform=False,  # no augmentation for calibration
        root=dataset_path,
    )

    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=ARCH["train"]["batch_size"],
        shuffle=False,
        num_workers=ARCH["train"]["workers"],
        drop_last=False,
    )


def calibrate(model, loader, nclasses, ignore_ids, device, feature_dim, max_batches=None,
              shrinkage=0.2, min_cov_samples=None):
    """Two-pass calibration: class means, then shared + per-class covariance.

    Accumulators are kept in float64 for numerical stability over the large
    pixel counts; per-batch feature math stays in float32.
    """
    hook = PenultimateHook(model)

    class_sums = torch.zeros((nclasses, feature_dim), dtype=torch.float64, device=device)
    class_counts = torch.zeros(nclasses, dtype=torch.float64, device=device)

    print("Phase 1: gathering class means...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            proj_in, proj_mask, proj_labels = batch[0], batch[1], batch[2]
            proj_in = proj_in.to(device)

            model(proj_in)  # populates the hook (output unused)
            feats = hook.features  # [B, C, H, W] float32
            B, C, H, W = feats.shape
            feats = feats.permute(0, 2, 3, 1).reshape(-1, C)

            labels = proj_labels.to(device).reshape(-1).long()
            mask = proj_mask.to(device).reshape(-1)
            keep = valid_pixel_mask(labels, mask, nclasses, ignore_ids)

            feats, labels = feats[keep], labels[keep]
            class_sums.index_add_(0, labels, feats.double())
            class_counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))

            if (i + 1) % 50 == 0:
                print(f"  means pass: {i + 1} batches")

    # Classes never seen get NaN means; they're excluded via class_counts>0.
    means = (class_sums / class_counts.clamp(min=1).unsqueeze(1)).float()
    seen = int((class_counts > 0).sum())
    print(f"  classes with samples: {seen}/{nclasses}")

    print("Phase 2: gathering shared covariance...")
    covariance = torch.zeros((feature_dim, feature_dim), dtype=torch.float64, device=device)
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            proj_in, proj_mask, proj_labels = batch[0], batch[1], batch[2]
            proj_in = proj_in.to(device)

            model(proj_in)
            feats = hook.features
            B, C, H, W = feats.shape
            feats = feats.permute(0, 2, 3, 1).reshape(-1, C)

            labels = proj_labels.to(device).reshape(-1).long()
            mask = proj_mask.to(device).reshape(-1)
            keep = valid_pixel_mask(labels, mask, nclasses, ignore_ids)

            feats, labels = feats[keep], labels[keep]
            # Center each pixel by its own class mean (tied-covariance pooling).
            centered = (feats - means[labels]).double()
            covariance += centered.t() @ centered
            total += centered.shape[0]

            if (i + 1) % 50 == 0:
                print(f"  covariance pass: {i + 1} batches")

    hook.remove()

    if total == 0:
        raise RuntimeError("No valid pixels found during calibration.")
    covariance /= total

    # Diagonal regularization guarantees invertibility.
    eye = torch.eye(feature_dim, dtype=torch.float64, device=device)
    inv_covariance = torch.inverse(covariance + 1e-5 * eye).float()

    return means, inv_covariance, class_counts


def main():
    base_cfg = load_yaml(CONFIG_DIR / "training_mcd.yaml")
    infer_cfg = load_yaml(CONFIG_DIR / "inference_mcd.yaml")

    # The trained run dir is <training.model_path>/<model_name>; the inference
    # yaml already records that full path, so reuse it as the default.
    default_model = infer_cfg.get("inference", {}).get("model_path")
    if not default_model:
        default_model = os.path.join(
            base_cfg["training"]["model_path"], base_cfg["training"]["model_name"]
        )

    ap = argparse.ArgumentParser("./ood_calibrate.py")
    ap.add_argument(
        "--model", "-m",
        default=default_model,
        help="Trained model run dir (holds weights + saved configs).",
    )
    ap.add_argument(
        "--dataset_path", "-dataset_path",
        default=base_cfg["dataset_path"],
        help="Dataset root for the ID training data (defaults to the TRAINING root).",
    )
    ap.add_argument(
        "--data_config", "-data_config",
        default=None,
        help="data_cfg yaml. Default: the data_cfg.yaml saved in the model dir.",
    )
    ap.add_argument(
        "--max-batches", type=int, default=None,
        help="Cap batches per pass for a quick/representative calibration.",
    )
    ap.add_argument(
        "--out", default=None,
        help=f"Output path. Default: <model_dir>/{STATS_FILENAME}.",
    )
    FLAGS, _ = ap.parse_known_args()

    if not os.path.isdir(FLAGS.model):
        raise FileNotFoundError(f"Model dir not found: {FLAGS.model}")

    arch_cfg_path = os.path.join(FLAGS.model, "arch_cfg.yaml")
    model_cfg_path = os.path.join(FLAGS.model, "model_config.yaml")
    data_cfg_path = FLAGS.data_config or os.path.join(FLAGS.model, "data_cfg.yaml")

    print("----------")
    print("OOD CALIBRATION")
    print("  model:       ", FLAGS.model)
    print("  dataset_path:", FLAGS.dataset_path)
    print("  arch_cfg:    ", arch_cfg_path)
    print("  data_cfg:    ", data_cfg_path)
    print("----------\n")

    ARCH = load_yaml(arch_cfg_path)
    DATA = load_yaml(data_cfg_path)
    model_cfg = load_yaml(model_cfg_path)
    img_width, img_height = model_cfg["img_width"], model_cfg["img_height"]
    ARCH.setdefault("dataset", {})["sensor"] = None  # MCD: per-shard sensors

    nclasses = num_classes(DATA)
    ignore_ids = ignored_class_ids(DATA)
    print(f"num_classes={nclasses}  ignore_ids={ignore_ids}  range_image={img_height}x{img_width}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader = build_calibration_loader(
        ARCH, DATA, FLAGS.dataset_path, img_width, img_height
    )
    model = build_model(ARCH, nclasses, FLAGS.model, device)

    from ood_common import FEATURE_DIM

    means, inv_covariance, class_counts = calibrate(
        model, loader, nclasses, ignore_ids, device, FEATURE_DIM,
        max_batches=FLAGS.max_batches,
    )

    out_path = FLAGS.out or os.path.join(FLAGS.model, STATS_FILENAME)
    save_stats(out_path, means, inv_covariance, class_counts, ignore_ids, nclasses)
    print(f"\nSaved OOD statistics -> {out_path}")


if __name__ == "__main__":
    main()
