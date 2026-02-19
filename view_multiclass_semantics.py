#!/usr/bin/env python3
"""
View multiclass semantic labels for a single scan.

This script loads:
1. A .bin scan file (point cloud: x, y, z, intensity)
2. The corresponding multiclass confidence scores file (probabilities per point per class)

Default: colors by argmax class, with brightness modulated by confidence.
With --single-label NAME_OR_ID: colors show one label's confidence (full color = high, black = zero).
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
import open3d as o3d
import yaml

# Add parent directory to path to import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ce_net.utils.file_io import read_bin_file

# Default config path: ce_net/config/data_cfg_mcd.yaml (relative to this script's directory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "ce_net", "config", "data_cfg_mcd.yaml")


def load_label_config(config_path):
    """
    Load label definitions (names, color_map, learning_map_inv) from a YAML config.
    Expects format like ce_net/config/data_cfg_mcd.yaml with keys:
      labels, color_map (BGR 0-255), learning_map_inv.
    
    Returns:
        dict with keys: labels (id -> name), color_map_rgb (id -> (r,g,b) 0-255),
                        learning_map_inv (class_idx -> label_id)
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    labels = cfg.get("labels", {})
    color_map_bgr = cfg.get("color_map", {})
    learning_map_inv = cfg.get("learning_map_inv", {})
    # Convert BGR to RGB for Open3D
    color_map_rgb = {}
    for k, bgr in color_map_bgr.items():
        color_map_rgb[int(k)] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    return {
        "labels": {int(k): v for k, v in labels.items()},
        "color_map_rgb": color_map_rgb,
        "learning_map_inv": {int(k): int(v) for k, v in learning_map_inv.items()},
    }


def _apply_value_curve(t, value_floor=0.0, gamma=1.0):
    """
    Map values in [0, 1] to output brightness. value_floor avoids pure black; gamma < 1 brightens mid-tones.
    """
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    if gamma != 1.0:
        t = np.power(t, gamma)
    if value_floor > 0:
        t = value_floor + (1.0 - value_floor) * t
    return t


def labels_to_colors(labels, label_id_to_color, confidences=None, value_floor=0.15, gamma=1.0):
    """
    Convert semantic label IDs to RGB colors; optionally modulate brightness by confidence.
    Normalizes confidence range; value_floor and gamma improve visibility.
    """
    if confidences is None:
        confidences = np.ones(len(labels), dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)
    max_c = np.max(confidences)
    min_c = np.min(confidences)
    rng = max_c - min_c
    if rng <= 0:
        rng = 1.0
    t = (confidences - min_c) / rng
    t = _apply_value_curve(t, value_floor=value_floor, gamma=gamma)
    colors = np.zeros((len(labels), 3), dtype=np.float32)
    for i, label_id in enumerate(labels):
        label_id_int = int(label_id)
        if label_id_int in label_id_to_color:
            base_color = np.array(label_id_to_color[label_id_int], dtype=np.float32) / 255.0
            colors[i] = base_color * t[i]
    return colors


def single_label_confidence_to_colors(confidences, label_id, label_id_to_color,
                                    normalize_range=True, value_floor=0.12, gamma=0.65,
                                    grayscale=False):
    """
    Color by one label's confidence.
    grayscale=True: high = white, low = black (use value_floor=0). Scene background gray is set in the viewer.
    grayscale=False: high = label color, low = dark.
    """
    colors = np.zeros((len(confidences), 3), dtype=np.float32)
    c = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if normalize_range:
        c_min, c_max = np.min(c), np.max(c)
        rng = c_max - c_min
        if rng <= 0:
            rng = 1.0
        c = (c - c_min) / rng
    t = _apply_value_curve(c, value_floor=value_floor, gamma=gamma)
    if grayscale:
        colors[:] = t[:, np.newaxis]
    else:
        if label_id not in label_id_to_color:
            return colors
        base = np.array(label_id_to_color[label_id], dtype=np.float32) / 255.0
        colors[:] = base * t[:, np.newaxis]
    return colors


def get_label_id_to_class_index(learning_map_inv):
    """
    Inverse of learning_map_inv: map semantic label ID -> class index (0 to n_classes-1).
    Used to index into multiclass_probs[:, class_index] for a given label.
    """
    return {int(v): int(k) for k, v in learning_map_inv.items()}


def map_class_indices_to_labels(class_indices, learning_map_inv):
    """
    Map class indices (0 to n_classes-1) to semantic label IDs using learning_map_inv.
    This is equivalent to MCD.map(class_indices, learning_map_inv).
    
    Args:
        class_indices: (N,) array of class indices (0 to n_classes-1)
        learning_map_inv: Dictionary mapping class index to semantic label ID
    
    Returns:
        label_ids: (N,) array of semantic label IDs
    """
    # Create lookup table (same approach as MCD.map)
    maxkey = 0
    for key in learning_map_inv.keys():
        if key > maxkey:
            maxkey = key
    
    # Create lookup table
    lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, value in learning_map_inv.items():
        try:
            lut[int(key)] = int(value)
        except IndexError:
            print(f"Warning: Wrong key {key}")
    
    # Apply mapping
    return lut[class_indices]


def load_scan_and_multiclass_labels(scan_path, multiclass_path, learning_map_inv):
    """
    Load a scan file and its corresponding multiclass probabilities.
    
    Args:
        scan_path: Path to .bin scan file (format: float32, shape (-1, 4) -> [x, y, z, intensity])
        multiclass_path: Path to multiclass probabilities file (format: float16, shape (N_points, N_classes))
    
    Returns:
        points_xyz: (N, 3) array of point coordinates
        intensities: (N,) array of intensities
        multiclass_probs: (N, C) array of probabilities per class
        semantic_labels: (N,) array of class indices (argmax over classes)
        confidences: (N,) array of maximum probabilities
    """
    print(f"Loading scan from: {scan_path}")
    if not os.path.exists(scan_path):
        raise FileNotFoundError(f"Scan file not found: {scan_path}")
    
    # Load point cloud (KITTI format: x, y, z, intensity)
    points = read_bin_file(scan_path, dtype=np.float32, shape=(-1, 4))
    points_xyz = points[:, :3]  # Extract xyz coordinates
    intensities = points[:, 3]  # Extract intensity
    
    print(f"  Loaded {len(points_xyz)} points")
    
    print(f"\nLoading multiclass probabilities from: {multiclass_path}")
    if not os.path.exists(multiclass_path):
        raise FileNotFoundError(f"Multiclass probabilities file not found: {multiclass_path}")
    
    # Load multiclass probabilities (shape: [N_points, N_classes], dtype: float16)
    multiclass_probs = read_bin_file(multiclass_path, dtype=np.float16)
    
    # Reshape based on number of points
    n_points = len(points_xyz)
    n_classes = len(multiclass_probs) // n_points
    
    # Verify that the data divides evenly
    if len(multiclass_probs) % n_points != 0:
        raise ValueError(
            f"Cannot reshape multiclass probabilities: {len(multiclass_probs)} elements "
            f"cannot be evenly divided into {n_points} points. Expected {n_points * n_classes} elements."
        )
    
    multiclass_probs = multiclass_probs.reshape(n_points, n_classes)
    
    print(f"  Loaded probabilities for {n_points} points and {n_classes} classes")
    print(f"  Shape: {multiclass_probs.shape}")
    
    # For each point, find the class index by taking argmax (0 to n_classes-1)
    class_indices = np.argmax(multiclass_probs, axis=1)
    confidences = np.max(multiclass_probs, axis=1)
    
    # Map class indices to semantic label IDs using learning_map_inv
    semantic_label_ids = map_class_indices_to_labels(class_indices, learning_map_inv)
    
    print(f"\nComputed labels:")
    print(f"  Class indices shape: {class_indices.shape}")
    print(f"  Semantic label IDs shape: {semantic_label_ids.shape}")
    print(f"  Unique class indices: {np.unique(class_indices)}")
    print(f"  Unique semantic label IDs: {np.unique(semantic_label_ids)}")
    print(f"  Confidence scores shape: {confidences.shape}")
    print(f"  Mean confidence: {np.mean(confidences):.4f}")
    print(f"  Min confidence: {np.min(confidences):.4f}")
    print(f"  Max confidence: {np.max(confidences):.4f}")
    
    return points_xyz, intensities, multiclass_probs, semantic_label_ids, confidences


def _parse_single_label(single_label_arg, label_config):
    """
    Parse --single-label: accept label name (e.g. 'vehicle-static') or id (e.g. '28').
    Returns (label_id, label_name) or (None, None) if not found.
    """
    labels_by_id = label_config["labels"]
    labels_by_name = {v: k for k, v in labels_by_id.items()}
    # Try as integer id first
    try:
        lid = int(single_label_arg)
        if lid in labels_by_id:
            return lid, labels_by_id[lid]
        return None, None
    except ValueError:
        pass
    # Try as name (case-insensitive)
    for name, lid in labels_by_name.items():
        if name.lower() == single_label_arg.lower():
            return lid, name
    return None, None


def main():
    """
    Load and display multiclass semantic labels for the first scan from kth_night_05.
    """
    parser = argparse.ArgumentParser(
        description="View multiclass semantic labels. Optionally view confidence of a single label (color = label color * confidence, 0 = black)."
    )
    parser.add_argument(
        "--single-label",
        type=str,
        default=2,
        metavar="NAME_OR_ID",
        help="View confidence for one label only: pass label name (e.g. vehicle-static) or id (e.g. 28). High confidence = label color, zero = black.",
    )
    parser.add_argument("--dataset-path", type=str, default="/media/donceykong/doncey_ssd_02/datasets/MCD")
    parser.add_argument("--seq", type=str, default="kth_day_09")
    parser.add_argument("--scan-idx", type=int, default=11)
    parser.add_argument("--no-normalize", action="store_true", help="Single-label: use raw probability instead of normalizing range")
    parser.add_argument("--value-floor", type=float, default=0.12, metavar="F", help="Minimum brightness 0..1 (default 0.12); with --grayscale points use 0 (white to black)")
    parser.add_argument("--gamma", type=float, default=0.65, metavar="G", help="Gamma for mid-tone contrast (default 0.65)")
    parser.add_argument("--grayscale", action="store_true", help="Single-label: points white (high) to black (low); scene background gray")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    seq_name = args.seq
    scan_idx = args.scan_idx

    # Load label config (colors and learning_map_inv) from MCD config
    config_path = DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        print(f"Error: Config not found: {config_path}")
        return 1
    print(f"Loading config from: {config_path}")
    label_config = load_label_config(config_path)
    learning_map_inv = label_config["learning_map_inv"]
    label_id_to_color = label_config["color_map_rgb"]
    print(f"Loaded learning_map_inv with {len(learning_map_inv)} mappings, "
          f"color_map with {len(label_id_to_color)} entries")
    
    # Get all scan files
    data_dir = os.path.join(dataset_path, seq_name, "lidar_bin/data")
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    bin_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.bin')])
    if len(bin_files) == 0:
        print(f"Error: No .bin files found in {data_dir}")
        return 1
    
    scan_name = bin_files[scan_idx]
    print(f"Using first scan from {seq_name}: {scan_name}")
    
    # Construct paths for MCD dataset
    scan_path = os.path.join(dataset_path, seq_name, "lidar_bin/data", scan_name)
    multiclass_path = os.path.join(
        dataset_path, seq_name, 
        "inferred_labels/cenet_mcd", 
        "multiclass_confidence_scores", 
        scan_name
    )
    
    try:
        points_xyz, intensities, multiclass_probs, semantic_label_ids, confidences = \
            load_scan_and_multiclass_labels(scan_path, multiclass_path, learning_map_inv)
        
        # Display some statistics
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total points: {len(points_xyz)}")
        print(f"Number of classes: {multiclass_probs.shape[1]}")
        label_names = label_config["labels"]
        print(f"\nPoints per class (semantic label IDs):")
        unique_labels, counts = np.unique(semantic_label_ids, return_counts=True)
        for label_id, count in zip(unique_labels, counts):
            percentage = 100 * count / len(semantic_label_ids)
            name = label_names.get(int(label_id), "?")
            print(f"  {name} (id={label_id}): {count} points ({percentage:.2f}%)")
        
        print(f"\nConfidence statistics:")
        print(f"  Mean: {np.mean(confidences):.4f}")
        print(f"  Std: {np.std(confidences):.4f}")
        print(f"  Min: {np.min(confidences):.4f}")
        print(f"  Max: {np.max(confidences):.4f}")
        print(f"  Points with confidence < 0.5: {np.sum(confidences < 0.5)} ({100*np.sum(confidences < 0.5)/len(confidences):.2f}%)")
        print(f"  Points with confidence < 0.7: {np.sum(confidences < 0.7)} ({100*np.sum(confidences < 0.7)/len(confidences):.2f}%)")
        
        # Show first few points as examples
        print(f"\nFirst 10 points (x, y, z, label_id, confidence):")
        for i in range(min(10, len(points_xyz))):
            print(f"  Point {i}: ({points_xyz[i,0]:.2f}, {points_xyz[i,1]:.2f}, {points_xyz[i,2]:.2f}), "
                  f"Label ID: {semantic_label_ids[i]}, Confidence: {confidences[i]:.4f}")
        
        # Visualize point cloud with semantic labels
        print("\n" + "="*80)
        print("VISUALIZATION")
        print("="*80)
        print("Creating point cloud visualization...")

        single_label_id = None
        single_label_name = None
        if args.single_label is not None:
            single_label_id, single_label_name = _parse_single_label(
                args.single_label, label_config
            )
            if single_label_id is None:
                print(f"Error: Unknown label '{args.single_label}'. Use a label name or id from the config.")
                return 1
            print(f"Single-label mode: showing confidence for '{single_label_name}' (id={single_label_id})")
            label_id_to_class_idx = get_label_id_to_class_index(learning_map_inv)
            if single_label_id not in label_id_to_class_idx:
                print(f"Error: No class index for label id {single_label_id} in learning_map_inv.")
                return 1
            class_idx = label_id_to_class_idx[single_label_id]
            single_confidences = np.asarray(
                multiclass_probs[:, class_idx], dtype=np.float32
            )
            value_floor = 0.0 if args.grayscale else args.value_floor
            colors = single_label_confidence_to_colors(
                single_confidences, single_label_id, label_id_to_color,
                normalize_range=not args.no_normalize,
                value_floor=value_floor, gamma=args.gamma,
                grayscale=args.grayscale,
            )
            print(f"  Confidence for this label: min={np.min(single_confidences):.4f}, max={np.max(single_confidences):.4f}, mean={np.mean(single_confidences):.4f}")
        else:
            # Convert semantic label IDs to colors, modulated by confidence
            # Lower confidence = darker/more transparent appearance
            colors = labels_to_colors(
                semantic_label_ids, label_id_to_color, confidences=confidences,
                value_floor=args.value_floor, gamma=args.gamma,
            )

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        print(f"Point cloud created with {len(points_xyz)} points")
        if single_label_id is not None:
            print(f"Single-label view: {single_label_name} (id={single_label_id})")
        else:
            print(f"Number of unique classes: {len(unique_labels)}")
        print(f"\nVisualization controls:")
        print(f"  - Mouse: Rotate view")
        print(f"  - Shift + Mouse: Pan view")
        print(f"  - Mouse wheel: Zoom")
        print(f"  - Q or ESC: Quit")
        print("="*80)
        print("\nOpening visualization window...")
        
        # Visualize (gray scene background when --grayscale)
        if args.grayscale and single_label_id is not None:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
            opt = vis.get_render_option()
            opt.background_color = np.array([0.5, 0.5, 0.5], dtype=np.float64)
            vis.run()
            vis.destroy_window()
        else:
            o3d.visualization.draw_geometries([pcd])
        
        print("\n" + "="*80)
        print("Done!")
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

