#!/usr/bin/env python3
"""
Compute per-class content ratios for the MCD dataset using gt_labels_terrain.

Usage:
    python compute_mcd_content_ratios.py
"""

import os
import numpy as np
from collections import defaultdict

DATASET_ROOT = "/media/donceykong/doncey_ssd_011/datasets/mcd"
SEQUENCES = ["kth_day_06", "kth_day_10", "kth_night_01", "kth_night_04"]
LABEL_DIR = "gt_labels_terrain"
NUM_CLASSES = 30  # 0-29

def main():
    class_counts = defaultdict(int)
    total_points = 0

    for seq in SEQUENCES:
        label_path = os.path.join(DATASET_ROOT, seq, LABEL_DIR)
        if not os.path.isdir(label_path):
            print(f"WARNING: {label_path} does not exist, skipping.")
            continue

        label_files = sorted([
            os.path.join(label_path, f)
            for f in os.listdir(label_path)
            if f.endswith(".bin")
        ])
        print(f"Sequence {seq}: {len(label_files)} label files")

        for lf in label_files:
            labels = np.fromfile(lf, dtype=np.int32)
            total_points += labels.shape[0]
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                class_counts[int(u)] += int(c)

    if total_points == 0:
        print("No points found.")
        return

    print(f"\nTotal points: {total_points}")
    print(f"\nContent ratios (for data_cfg_mcd.yaml):")
    print("content:")
    for cls_id in range(NUM_CLASSES):
        ratio = class_counts.get(cls_id, 0) / total_points
        print(f"  {cls_id}: {ratio:.10f}")

    print(f"\nRaw counts:")
    for cls_id in range(NUM_CLASSES):
        print(f"  {cls_id}: {class_counts.get(cls_id, 0)}")

    unexpected = set(class_counts.keys()) - set(range(NUM_CLASSES))
    if unexpected:
        print(f"\nWARNING: Unexpected class IDs: {unexpected}")
        for cls_id in sorted(unexpected):
            print(f"  {cls_id}: count={class_counts[cls_id]}, ratio={class_counts[cls_id]/total_points:.10f}")


if __name__ == "__main__":
    main()
