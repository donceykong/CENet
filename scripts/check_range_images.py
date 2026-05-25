#!/usr/bin/env python3
"""Sanity-check range-image projection per sensor group.

For every sensor_groups entry in data_cfg_mcd.yaml, take a handful of scans
from each sequence, project them with that sensor's (fov_up, fov_down) and
the model's target (img_height, img_width), and save a PNG strip.

Run it before training to confirm:
  - the right SENSORS yaml is associated with each sequence,
  - the projection covers the FoV without obvious clipping or wrap-around,
  - the resulting range image looks structured (rings/columns visible),
    not noise.

Usage:
    python scripts/check_range_images.py
    python scripts/check_range_images.py --num_scans 5 --img_width 1024
    python scripts/check_range_images.py --out /tmp/range_check
"""

import argparse
import os
import sys

import numpy as np
import yaml

# Internal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ce_net import CONFIG_DIR
from ce_net.core.pointcloud.laserscan import LaserScan
from ce_net.utils.sensor import materialize_sensor_groups


SCAN_SUBDIR = os.path.join("lidar_bin", "data")


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _to_uint8(img, valid_mask):
    """Linearly stretch valid range pixels to [0, 255]."""
    out = np.zeros(img.shape, dtype=np.uint8)
    if not valid_mask.any():
        return out
    vmin = float(img[valid_mask].min())
    vmax = float(img[valid_mask].max())
    if vmax <= vmin:
        return out
    norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    out = (norm * 255).astype(np.uint8)
    out[~valid_mask] = 0
    return out


def _save_strip(out_path, frames, gap=4):
    """Stack frames vertically with a small black gap between them."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required for saving range-image strips. "
            "pip install pillow"
        ) from e

    h = sum(f.shape[0] for f in frames) + gap * (len(frames) - 1)
    w = max(f.shape[1] for f in frames)
    canvas = np.zeros((h, w), dtype=np.uint8)
    y = 0
    for i, f in enumerate(frames):
        canvas[y : y + f.shape[0], : f.shape[1]] = f
        y += f.shape[0] + gap
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def _sample_scan_files(seq_dir, n):
    scan_dir = os.path.join(seq_dir, SCAN_SUBDIR)
    if not os.path.isdir(scan_dir):
        return []
    files = sorted(f for f in os.listdir(scan_dir) if f.endswith(".bin"))
    if not files:
        return []
    if len(files) <= n:
        idx = list(range(len(files)))
    else:
        idx = np.linspace(0, len(files) - 1, n).round().astype(int).tolist()
    return [os.path.join(scan_dir, files[i]) for i in idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_cfg",
        type=str,
        default=str(CONFIG_DIR / "training_mcd.yaml"),
        help="Top-level training yaml; used for dataset_path + data_config defaults.",
    )
    parser.add_argument(
        "--data_cfg",
        type=str,
        default=None,
        help="Override data_cfg path (otherwise read from training yaml).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override dataset root (otherwise read from training yaml).",
    )
    parser.add_argument(
        "--img_width", type=int, default=1024,
        help="Range-image width to project at.",
    )
    parser.add_argument(
        "--img_height", type=int, default=64,
        help="Range-image height to project at.",
    )
    parser.add_argument(
        "--num_scans", type=int, default=3,
        help="Scans to sample per sequence (evenly spaced).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "range_images"),
        help="Output directory for PNG strips. Default: scripts/range_images.",
    )
    args = parser.parse_args()

    training_cfg = load_yaml(args.training_cfg)
    data_cfg_path = args.data_cfg or training_cfg["data_config"]
    dataset_path = args.dataset_path or training_cfg["dataset_path"]
    print(f"training_cfg : {args.training_cfg}")
    print(f"data_cfg     : {data_cfg_path}")
    print(f"dataset_path : {dataset_path}")
    print(f"img size     : {args.img_height} x {args.img_width}")
    print(f"out dir      : {args.out}\n")

    data_cfg = load_yaml(data_cfg_path)
    if "sensor_groups" not in data_cfg:
        raise ValueError(f"data_cfg {data_cfg_path} has no 'sensor_groups'.")

    groups = materialize_sensor_groups(
        data_cfg["sensor_groups"], args.img_width, args.img_height
    )

    out_root = os.path.abspath(args.out)
    n_total = 0
    for grp in groups:
        sensor = grp["sensor"]
        sensor_name = grp["sensor_name"]
        H = sensor["img_prop"]["height"]
        W = sensor["img_prop"]["width"]
        print(
            f"[{sensor_name}] fov_up={sensor['fov_up']} "
            f"fov_down={sensor['fov_down']} -> H={H} W={W}"
        )
        for seq in grp["sequences"]:
            seq_dir = os.path.join(dataset_path, seq)
            scan_files = _sample_scan_files(seq_dir, args.num_scans)
            if not scan_files:
                print(f"  {seq:35s} SKIP (no scans found)")
                continue

            frames = []
            for scan_file in scan_files:
                ls = LaserScan(
                    project=True,
                    H=H,
                    W=W,
                    fov_up=sensor["fov_up"],
                    fov_down=sensor["fov_down"],
                )
                ls.open_scan(scan_file)
                valid = ls.proj_range > 0
                frames.append(_to_uint8(ls.proj_range, valid))

            out_path = os.path.join(
                out_root, sensor_name, f"{seq.replace(os.sep, '_')}.png"
            )
            _save_strip(out_path, frames)
            n_total += len(frames)
            print(f"  {seq:35s} -> {out_path} ({len(frames)} scans)")

    print(f"\nWrote {n_total} range image(s) under {out_root}")


if __name__ == "__main__":
    main()
