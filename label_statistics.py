#!/usr/bin/env python3
"""
Compute label statistics over all .bin label files in a sequence (or multiple).
Reads only gt_labels/*.bin (no pointcloud .bin files). Reports each label's
count and percentage using label names from ce_net/config/data_cfg_mcd.yaml.
"""

import os
import sys
import yaml
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Import dataset_binarize package to set up sys.path for lidar2osm imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ce_net.utils.file_io import read_bin_file


def load_label_names(config_path=None):
    """Load label id -> name from data_cfg_mcd.yaml."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "ce_net", "config", "data_cfg_mcd.yaml"
        )
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    labels = data.get("labels", {})
    # YAML may load keys as int or str
    return {int(k): str(v) for k, v in labels.items()}


def collect_label_files(root_path, sequences=None):
    """
    Collect all .bin label file paths under root_path.

    If sequences is given, only collect from root_path/<seq>/gt_labels/ for each
    seq in sequences. Otherwise: if root_path has gt_labels/, use that; else
    look for subdirs that have gt_labels/.
    """
    if sequences is not None:
        all_files = []
        for seq in sequences:
            labels_dir = os.path.join(root_path, seq, "gt_labels")
            if not os.path.isdir(labels_dir):
                continue
            for f in sorted(os.listdir(labels_dir)):
                if f.endswith(".bin"):
                    all_files.append(os.path.join(labels_dir, f))
        return sorted(all_files)

    labels_dir = os.path.join(root_path, "gt_labels")
    if os.path.isdir(labels_dir):
        # Single sequence root
        files = sorted(
            [
                os.path.join(labels_dir, f)
                for f in os.listdir(labels_dir)
                if f.endswith(".bin")
            ]
        )
        return files

    # Multi-sequence: root_path contains e.g. day_06, day_09, ...
    all_files = []
    for name in sorted(os.listdir(root_path)):
        sub = os.path.join(root_path, name)
        if not os.path.isdir(sub):
            continue
        sub_labels = os.path.join(sub, "gt_labels")
        if not os.path.isdir(sub_labels):
            continue
        for f in sorted(os.listdir(sub_labels)):
            if f.endswith(".bin"):
                all_files.append(os.path.join(sub_labels, f))
    return all_files


def compute_label_statistics(root_path, config_path=None, sequences=None):
    """
    Read all .bin label files under root_path, aggregate counts per label,
    and return (label_id_to_name, counts_dict, total_points, num_files).

    If sequences is given, only root_path/<seq>/gt_labels/*.bin are used.
    """
    label_names = load_label_names(config_path)
    label_files = collect_label_files(root_path, sequences=sequences)

    if not label_files:
        print(f"No .bin label files found under: {root_path}")
        return label_names, {}, 0, 0

    counts = defaultdict(int)
    total = 0

    for path in tqdm(label_files, desc="Reading label files", unit="file"):
        try:
            labels = read_bin_file(path, dtype=np.int32, shape=(-1))
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}", file=sys.stderr)
            continue
        total += len(labels)
        for lid in labels:
            counts[int(lid)] += 1

    return label_names, dict(counts), total, len(label_files)


def print_statistics(label_names, counts, total):
    """Print per-label count and percentage."""
    if total == 0:
        print("No points in any label file.")
        return

    # All label IDs that appear in data or in config
    all_ids = sorted(set(counts.keys()) | set(label_names.keys()))

    print(f"\n{'Label':<6} {'Name':<22} {'Count':>12} {'Percent':>10}")
    print("-" * 54)

    for lid in all_ids:
        name = label_names.get(lid, f"unknown({lid})")
        cnt = counts.get(lid, 0)
        pct = 100.0 * cnt / total
        print(f"{lid:<6} {name:<22} {cnt:>12} {pct:>9.2f}%")

    print("-" * 54)
    print(f"{'Total':<6} {'':<22} {total:>12} {100.0:>9.2f}%")


def main():
    # Root path: either a single sequence dir (with gt_labels/) or a parent with multiple sequences
    root_path = "/media/donceykong/doncey_ssd_02/datasets/MCD"
    sequences = [
        "kth_day_06",
        "kth_day_09",
        "kth_day_10",
        "kth_night_01",
        "kth_night_04",
        "kth_night_05",
    ]

    if len(sys.argv) > 1:
        root_path = sys.argv[1]

    config_path = None
    if len(sys.argv) > 2:
        config_path = sys.argv[2]

    print(f"Root path: {root_path}")
    print(f"Sequences: {sequences}")
    label_names, counts, total, num_files = compute_label_statistics(
        root_path, config_path, sequences=sequences
    )
    print(f"Total label files: {num_files}")
    print(f"Total points: {total}")
    print_statistics(label_names, counts, total)


if __name__ == "__main__":
    main()
