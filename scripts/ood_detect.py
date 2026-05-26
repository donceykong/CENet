#!/usr/bin/env python3
"""Phase 2 (online): per-point Mahalanobis OOD detection with CENet.

Loads the statistics calibrated by ood_calibrate.py, runs CENet over a split,
taps the penultimate feature map (``model.conv_2``), scores every pixel by its
Mahalanobis distance to the nearest ID class cluster, then un-projects the
range-image score map back to 3D points (using the same p_x/p_y indices CENet
already uses for predictions) and writes one score per point.

Higher score == more out-of-distribution.

This mirrors scripts/infer.py for config + data setup (it reuses the ``User``
class to build the model and parser exactly as production inference does), then
runs its own OOD loop instead of saving class predictions. The model itself is
untouched -- features are captured with a forward hook.

Output: ``<dataset>/<seq>/<relative_infer_dir>/ood_scores/<name>`` as a flat
float32 binary (np.tofile), one value per point, aligned with the scan's
points -- matching the layout of the existing confidence_scores output.

Example:
    python scripts/ood_detect.py \
        --model /media/.../mcd_terrain_ntu_tuhh/cenet_mcd-1024 \
        --split test
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ce_net import CONFIG_DIR
from ce_net.core.parsers.mcd import build_mcd_inference_shards
from ce_net.models.user import User
from ce_net.utils.sensor import load_sensor, materialize_sensor_groups

from ood_common import (
    STATS_FILENAME,
    load_stats,
    load_yaml,
    mahalanobis_scores,
    PenultimateHook,
)


def _pick_loader(parser, split):
    if split == "train":
        return parser.get_train_set()
    if split == "valid":
        return parser.get_valid_set()
    return parser.get_test_set()


def run_ood(user, stats, split, dataset_path, dataset_name, relative_infer_dir):
    """Iterate a split, compute per-point OOD scores, save them to disk."""
    model = user.model
    model.eval()
    device = user.device
    means = stats["means"]
    inv_cov = stats["inv_covariance"]
    class_ids = stats["class_ids"]

    loader = _pick_loader(user.parser, split)
    hook = PenultimateHook(model)

    times = []
    n_frames = 0
    with torch.no_grad():
        for batch in loader:
            (proj_in, _proj_mask, _proj_labels, _unproj_labels, path_seq,
             path_name, p_x, p_y, *_rest) = batch
            npoints = batch[14]

            p_x = p_x[0, :npoints].to(device)
            p_y = p_y[0, :npoints].to(device)
            path_seq = path_seq[0]
            path_name = path_name[0]
            proj_in = proj_in.to(device)

            end = time.time()
            # No autocast: keep penultimate features in fp32 for stable distances.
            model(proj_in)
            features = hook.features  # [B, C, H, W]

            ood_map = mahalanobis_scores(features, means, inv_cov, class_ids)  # [B, H, W]
            # Un-project the range-image scores to points (nearest pixel), the
            # same indexing CENet uses for predictions/confidences.
            ood_points = ood_map[0][p_y, p_x]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - end)
            n_frames += 1

            # Save alongside the inferred labels, mirroring confidence_scores.
            label_dir = os.path.join(dataset_path, path_seq, relative_infer_dir)
            ood_dir = os.path.join(label_dir, "ood_scores")
            os.makedirs(ood_dir, exist_ok=True)
            ood_path = os.path.join(ood_dir, path_name)

            ood_np = ood_points.cpu().numpy().reshape(-1).astype(np.float32)
            ood_np.tofile(ood_path)
            print(f"seq {path_seq} scan {path_name}: "
                  f"{n_frames} | points={ood_np.shape[0]} "
                  f"min={ood_np.min():.2f} max={ood_np.max():.2f} -> {ood_path}")

    hook.remove()
    if times:
        print(f"\nMean OOD inference time: {np.mean(times):.4f}s  std: {np.std(times):.4f}s")
    print(f"Total frames: {n_frames}\nFinished OOD detection.")


def main():
    config = load_yaml(CONFIG_DIR / "inference_mcd.yaml")
    splits = ["train", "valid", "test"]

    ap = argparse.ArgumentParser("./ood_detect.py")
    ap.add_argument("--dataset_path", "-dataset_path",
                    default=config.get("dataset_path"))
    ap.add_argument("--dataset_name", "-d_name",
                    default=config.get("dataset_name"))
    ap.add_argument("--model", "-m",
                    default=config.get("inference", {}).get("model_path"))
    ap.add_argument("--split", "-s", choices=splits,
                    default=config.get("inference", {}).get("split", "test"))
    ap.add_argument("--data_config", "-data_config",
                    default=config.get("data_config"))
    ap.add_argument("--stats", default=None,
                    help=f"OOD stats path. Default: <model>/{STATS_FILENAME}.")
    FLAGS, _ = ap.parse_known_args()

    print("----------")
    print("OOD DETECTION")
    print("  dataset_name:", FLAGS.dataset_name)
    print("  dataset_path:", FLAGS.dataset_path)
    print("  model:       ", FLAGS.model)
    print("  split:       ", FLAGS.split)
    print("----------\n")

    if not os.path.isdir(FLAGS.model):
        raise FileNotFoundError(f"Model dir not found: {FLAGS.model}")

    ARCH = load_yaml(os.path.join(FLAGS.model, "arch_cfg.yaml"))
    DATA = load_yaml(FLAGS.data_config)
    model_cfg = load_yaml(os.path.join(FLAGS.model, "model_config.yaml"))
    img_width, img_height = model_cfg["img_width"], model_cfg["img_height"]
    print(f"Range image: {img_height} x {img_width}")

    relative_infer_dir = config.get("relative_infer_dir", "inferred_labels/cenet_mcd")
    DATA["relative_infer_dir"] = relative_infer_dir

    # --- Build the inference data setup, mirroring scripts/infer.py ---
    if FLAGS.dataset_name == "MCD":
        if "sensor_groups" not in DATA:
            raise ValueError("data_cfg is missing 'sensor_groups' (required for MCD).")
        groups = materialize_sensor_groups(
            DATA["sensor_groups"], img_width=img_width, img_height=img_height
        )
        shards = build_mcd_inference_shards(
            FLAGS.dataset_path, groups, sequences=DATA.get("infer_sequences")
        )
        if not shards:
            raise RuntimeError("No MCD inference sequences resolved.")
        for s in shards:
            print(f"  {s['seq']:35s} sensor={s['sensor_name']:8s} scans={len(s['scan_files'])}")
        DATA["split"] = {"train": [], "valid": [], "test": shards}
        ARCH.setdefault("dataset", {})["sensor"] = None
        # MCD inference data only populates the test loader.
        split = "test"
        if FLAGS.split != "test":
            print(f"[MCD] inference shards populate the test loader; using 'test' "
                  f"instead of '{FLAGS.split}'.")
    else:
        if not ARCH.get("dataset", {}).get("sensor"):
            sensor_ref = DATA.get("sensor_config")
            if sensor_ref is None:
                raise ValueError("ARCH has no sensor and data_cfg has no 'sensor_config'.")
            ARCH.setdefault("dataset", {})["sensor"] = load_sensor(
                sensor_ref, img_width, img_height
            )
        DATA.setdefault("sequences", [DATA.get("seq")] if DATA.get("seq") else [])
        split = FLAGS.split

    user = User(
        ARCH, DATA, FLAGS.dataset_name, FLAGS.dataset_path, FLAGS.model, split
    )

    stats_path = FLAGS.stats or os.path.join(FLAGS.model, STATS_FILENAME)
    stats = load_stats(stats_path, user.device)
    print(f"Loaded OOD stats from {stats_path} "
          f"(classes={stats['class_ids'].numel()}, feature_dim={stats['feature_dim']})")

    run_ood(user, stats, split, FLAGS.dataset_path, FLAGS.dataset_name, relative_infer_dir)


if __name__ == "__main__":
    main()
