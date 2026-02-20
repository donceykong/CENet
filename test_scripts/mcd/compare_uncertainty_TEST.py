#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
import matplotlib.pyplot as plt
import open3d as o3d
import yaml
from tqdm import tqdm

# Resolve project root (repo root containing ce_net): two levels up from test_scripts/mcd/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
from ce_net.utils.file_io import read_bin_file

# MCD label config path relative to project root
DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "ce_net", "config", "data_cfg_mcd.yaml")

# Body to LiDAR transformation matrix
BODY_TO_LIDAR_TF = np.array([
    [0.9999135040741837, -0.011166365511073898, -0.006949579221822984, -0.04894521120494695],
    [-0.011356389542502144, -0.9995453006865824, -0.02793249526856565, -0.03126929060348084],
    [-0.006634514801117132, 0.02800900135032654, -0.999585653686922, -0.01755515794222565],
    [0.0, 0.0, 0.0, 1.0]
])

def read_bin_file(file_path, dtype, shape=None):
    """
    Reads a .bin file and reshapes the data according to the provided shape.

    Args:
        file_path (str): The path to the .bin file.
        dtype (data-type): The data type of the file content (e.g., np.float32, np.int16).
        shape (tuple, optional): The desired shape of the output array. If None, the data is returned as a 1D array.

    Returns:
        np.ndarray: The data read from the .bin file, reshaped according to the provided shape.
    """
    data = np.fromfile(file_path, dtype=dtype)
    if shape:
        return data.reshape(shape)
    return data


def load_label_config(config_path):
    """
    Load label definitions (names, color_map, learning_map_inv) from MCD YAML config.
    Returns dict with labels, color_map_rgb (id -> (r,g,b) 0-255), learning_map_inv.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    labels = cfg.get("labels", {})
    color_map_bgr = cfg.get("color_map", {})
    learning_map_inv = cfg.get("learning_map_inv", {})
    color_map_rgb = {}
    for k, bgr in color_map_bgr.items():
        color_map_rgb[int(k)] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    return {
        "labels": {int(k): v for k, v in labels.items()},
        "color_map_rgb": color_map_rgb,
        "learning_map_inv": {int(k): int(v) for k, v in learning_map_inv.items()},
    }


def get_label_id_to_class_index(learning_map_inv):
    """Map semantic label ID -> class index for indexing multiclass_probs."""
    return {int(v): int(k) for k, v in learning_map_inv.items()}


def map_class_indices_to_labels(class_indices, learning_map_inv):
    """Map class indices (0..n_classes-1) to semantic label IDs."""
    maxkey = max(learning_map_inv.keys(), default=0)
    lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, value in learning_map_inv.items():
        try:
            lut[int(key)] = int(value)
        except IndexError:
            pass
    return lut[class_indices]


# Viridis colormap (standard key points, interpolated to 256 entries): dark purple/blue -> green -> yellow
_VIRIDIS_KEY = np.array([
    [0.267004, 0.004874, 0.329415],
    [0.282327, 0.140926, 0.457517],
    [0.127568, 0.566949, 0.550556],
    [0.369214, 0.788888, 0.383287],
    [0.993248, 0.906157, 0.143936],
], dtype=np.float32)
_VIRIDIS_LUT = np.zeros((256, 3), dtype=np.float32)
for i in range(256):
    t = i / 255.0
    idx = t * 4  # 4 segments between 5 key points
    j = int(np.clip(np.floor(idx), 0, 3))
    u = idx - j
    _VIRIDIS_LUT[i] = (1 - u) * _VIRIDIS_KEY[j] + u * _VIRIDIS_KEY[j + 1]


def scalar_to_viridis_rgb(values, normalize_range=True):
    """
    Map scalar values to RGB using the Viridis colormap (dark = low, yellow = high).
    values: (N,) array. If normalize_range=True, map [min, max] to [0, 1] before lookup.
    Returns (N, 3) RGB in [0, 1].
    """
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if normalize_range:
        vmin, vmax = np.min(v), np.max(v)
        rng = vmax - vmin
        if rng <= 0:
            rng = 1.0
        v = (v - vmin) / rng
    v = np.clip(v, 0.0, 1.0)
    idx = (v * 255).astype(np.int32)
    idx = np.clip(idx, 0, 255)
    return _VIRIDIS_LUT[idx].copy()


def _apply_value_curve(t, value_floor=0.0, gamma=1.0):
    """
    Map values in [0, 1] to output brightness for better visibility.
    value_floor: minimum output (e.g. 0.15 so low confidence isn't pure black).
    gamma: < 1 brightens mid-tones, > 1 darkens them.
    """
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    if gamma != 1.0:
        t = np.power(t, gamma)
    if value_floor > 0:
        t = value_floor + (1.0 - value_floor) * t
    return t


def labels_to_colors(labels, label_id_to_color, confidences=None, value_floor=0.15, gamma=1.0):
    """
    Convert semantic label IDs to RGB colors. If confidences is given, modulate brightness
    (lower confidence = darker). Normalizes confidence range per batch; optional floor/gamma
    improve visibility (avoid pure black, better mid-tone contrast).
    """
    n = len(labels)
    colors = np.zeros((n, 3), dtype=np.float32)
    if confidences is None:
        confidences = np.ones(n, dtype=np.float32)
    else:
        confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)
    max_c = np.max(confidences)
    min_c = np.min(confidences)
    rng = max_c - min_c
    if rng <= 0:
        # Uniform confidences (e.g. confidences=None): full brightness
        t = np.ones(n, dtype=np.float32)
    else:
        t = (confidences - min_c) / rng
        t = _apply_value_curve(t, value_floor=value_floor, gamma=gamma)
    for i, label_id in enumerate(labels):
        lid = int(label_id)
        if lid in label_id_to_color:
            base = np.array(label_id_to_color[lid], dtype=np.float32) / 255.0
            colors[i] = base * t[i]
        else:
            colors[i] = [0.5, 0.5, 0.5]
    return colors


def single_label_confidence_to_colors(confidences, label_id, label_id_to_color,
                                      normalize_range=True, value_floor=0.12, gamma=0.65,
                                      grayscale=False):
    """
    Color by one label's confidence.
    - grayscale=False: high = label color, low = dark (with value_floor).
    - grayscale=True: high = white, low = black (use value_floor=0 for full range). Scene background gray is set in the viewer.
    - normalize_range, value_floor, gamma: as in labels_to_colors.
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
        # White (high) to black (low); use value_floor=0 for full range
        colors[:] = t[:, np.newaxis]
    else:
        if label_id not in label_id_to_color:
            return colors
        base = np.array(label_id_to_color[label_id], dtype=np.float32) / 255.0
        colors[:] = base * t[:, np.newaxis]
    return colors


def _parse_single_label(single_label_arg, label_config):
    """Parse --single-label: accept label name or id. Return (label_id, label_name) or (None, None)."""
    labels_by_id = label_config["labels"]
    labels_by_name = {v: k for k, v in labels_by_id.items()}
    try:
        lid = int(single_label_arg)
        if lid in labels_by_id:
            return lid, labels_by_id[lid]
        return None, None
    except ValueError:
        pass
    for name, lid in labels_by_name.items():
        if name.lower() == single_label_arg.lower():
            return lid, name
    return None, None


def load_poses(poses_file):
    """
    Load poses from CSV file.
    
    Args:
        poses_file: Path to CSV file with poses (num, t, x, y, z, qx, qy, qz, qw)
    
    Returns:
        poses: Dictionary mapping timestamp to [index, x, y, z, qx, qy, qz, qw]
        index_to_timestamp: Dictionary mapping index to timestamp
    """
    print(f"\nLoading poses from {poses_file}")
    
    # First, inspect the CSV file to understand its format
    try:
        # Read first few lines to inspect format
        with open(poses_file, 'r') as f:
            first_lines = [f.readline().strip() for _ in range(5)]
        
        print(f"\nCSV file inspection:")
        print(f"  First 3 lines of file:")
        for i, line in enumerate(first_lines[:3]):
            print(f"    Line {i+1}: {line[:100]}")  # Print first 100 chars
        
        # Try different reading strategies
        df = None
        
        # Strategy 1: Try with header, handling comment lines
        try:
            df = pd.read_csv(poses_file, comment='#', skipinitialspace=True)
            print(f"\n  Attempted to read with header (comment='#')")
            print(f"  Columns found: {list(df.columns)}")
            print(f"  Number of columns: {len(df.columns)}")
            
            # Check if we have the expected columns (handle various naming conventions)
            col_names = [str(col).strip().lower().replace('#', '').replace(' ', '') for col in df.columns]
            has_timestamp_col = any('timestamp' in col or col == 't' for col in col_names)
            has_pose_cols = all(any(coord in col for col in col_names) for coord in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
            
            # Check for num column (optional)
            has_num_col = any('num' in col for col in col_names)
            
            if has_timestamp_col and has_pose_cols and (len(df.columns) >= 8 or (has_num_col and len(df.columns) >= 9)):
                print(f"  ✓ Format detected: Has header with column names")
                if has_num_col:
                    print(f"    Note: Found 'num' column, will be ignored")
            elif len(df.columns) == 8 or len(df.columns) == 9:
                print(f"  ⚠ Format: {len(df.columns)} columns but unclear header format, trying positional")
                df = None  # Will try positional
            else:
                print(f"  ⚠ Format: Unexpected column count ({len(df.columns)}), trying positional")
                df = None
        except Exception as e:
            print(f"  Failed to read with header: {e}")
            df = None
        
        # Strategy 2: Try without header (positional)
        if df is None:
            try:
                df = pd.read_csv(poses_file, comment='#', header=None, skipinitialspace=True)
                print(f"\n  Attempted to read without header (positional)")
                print(f"  Number of columns: {len(df.columns)}")
                if len(df.columns) >= 8:
                    print(f"  ✓ Format detected: No header, using positional indexing")
                else:
                    print(f"  ✗ Error: Only {len(df.columns)} columns found, need at least 8")
                    return {}
            except Exception as e:
                print(f"  Failed to read without header: {e}")
                return {}
        
        print(f"\n  Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"  First few data rows:")
        print(df.head(3))
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    poses = {}
    index_to_timestamp = {}
    
    for idx, row in df.iterrows():
        try:
            # Try to get timestamp and index - handle various column name formats
            timestamp = None
            pose_index = None
            x, y, z, qx, qy, qz, qw = None, None, None, None, None, None, None
            
            # Build column map for pose values
            col_map = {}
            for col in df.columns:
                col_clean = str(col).strip().lower().replace('#', '').replace(' ', '')
                col_map[col_clean] = col
            
            # Check for timestamp column (handle various naming: 't', 'timestamp', etc.)
            for col in df.columns:
                col_clean = str(col).strip().lower().replace('#', '').replace(' ', '')
                # Check for 't' (single letter) or 'timestamp'
                if col_clean == 't' or 'timestamp' in col_clean:
                    timestamp = row[col]
                    break
            
            # Check for index column ('num')
            if 'num' in col_map:
                pose_index = int(row[col_map['num']])
            
            # Get pose values by column name
            if timestamp is not None:
                # We have a timestamp column, get pose values by name
                x = row.get(col_map.get('x'), None) if 'x' in col_map else None
                y = row.get(col_map.get('y'), None) if 'y' in col_map else None
                z = row.get(col_map.get('z'), None) if 'z' in col_map else None
                qx = row.get(col_map.get('qx'), None) if 'qx' in col_map else None
                qy = row.get(col_map.get('qy'), None) if 'qy' in col_map else None
                qz = row.get(col_map.get('qz'), None) if 'qz' in col_map else None
                qw = row.get(col_map.get('qw'), None) if 'qw' in col_map else None
            else:
                # No timestamp column found, use positional indexing
                # Format: [num, t, x, y, z, qx, qy, qz, qw] or [t, x, y, z, qx, qy, qz, qw]
                if len(row) >= 9:
                    # Has num column: first column is index
                    pose_index = int(row.iloc[0])
                    timestamp = row.iloc[1]
                    x = row.iloc[2]
                    y = row.iloc[3]
                    z = row.iloc[4]
                    qx = row.iloc[5]
                    qy = row.iloc[6]
                    qz = row.iloc[7]
                    qw = row.iloc[8]
                elif len(row) >= 8:
                    # No num column
                    timestamp = row.iloc[0]
                    x = row.iloc[1]
                    y = row.iloc[2]
                    z = row.iloc[3]
                    qx = row.iloc[4]
                    qy = row.iloc[5]
                    qz = row.iloc[6]
                    qw = row.iloc[7]
                else:
                    if idx < 5:
                        print(f"  Row {idx} has only {len(row)} columns, need at least 8, skipping")
                    continue
            
            # Validate all values are present
            if None in [timestamp, x, y, z, qx, qy, qz, qw]:
                if idx < 5:
                    print(f"  Row {idx} missing values, skipping")
                continue
            
            # Store pose with index: [index, x, y, z, qx, qy, qz, qw]
            pose = [pose_index, float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)]
            poses[float(timestamp)] = pose
            
            # Store index to timestamp mapping
            if pose_index is not None:
                index_to_timestamp[pose_index] = float(timestamp)
        except Exception as e:
            if idx < 5:  # Only print first few errors
                print(f"  Error processing row {idx}: {e}")
                print(f"  Row data: {row.tolist()[:9] if len(row) >= 9 else row.tolist()}")
            continue
    
    print(f"\nSuccessfully loaded {len(poses)} poses")
    if len(poses) > 0:
        sample_ts = list(poses.keys())[0]
        sample_pose = poses[sample_ts]
        print(f"  Sample pose (timestamp {sample_ts}, index {sample_pose[0]}): position=[{sample_pose[1]:.2f}, {sample_pose[2]:.2f}, {sample_pose[3]:.2f}], quat=[{sample_pose[4]:.3f}, {sample_pose[5]:.3f}, {sample_pose[6]:.3f}, {sample_pose[7]:.3f}]")
    return poses, index_to_timestamp


def find_closest_pose(timestamp, poses_dict, exact_match_threshold=0.001):
    """
    Find the closest pose timestamp to the given timestamp.
    
    Args:
        timestamp: Target timestamp
        poses_dict: Dictionary mapping timestamp to pose
        exact_match_threshold: Time difference threshold in seconds to consider an exact match (default: 0.001s = 1ms)
    
    Returns:
        Tuple of (closest timestamp key, time_difference), or (None, None) if poses_dict is empty
        Returns None for timestamp if no exact match found (time_diff > threshold)
    """
    if not poses_dict:
        return None, None
    
    pose_timestamps = np.array(list(poses_dict.keys()))
    time_diffs = np.abs(pose_timestamps - timestamp)
    closest_idx = np.argmin(time_diffs)
    closest_ts = pose_timestamps[closest_idx]
    time_diff = time_diffs[closest_idx]
    
    # Check if it's an exact match
    if time_diff > exact_match_threshold:
        # Warn but return None to indicate no exact match
        print(f"  WARNING: No exact pose match for timestamp {timestamp:.6f}. "
              f"Closest pose at {closest_ts:.6f} (difference: {time_diff:.6f}s). Skipping scan.")
        return None, time_diff
    
    return closest_ts, time_diff


def transform_points_to_world(points_xyz, position, quaternion, body_to_lidar_tf=None):
    """
    Transform points from lidar frame to world frame using pose.
    
    Args:
        points_xyz: (N, 3) array of points in lidar frame
        position: [x, y, z] translation (body frame position in world)
        quaternion: [qx, qy, qz, qw] rotation quaternion (body frame orientation in world)
        body_to_lidar_tf: Optional 4x4 transformation matrix from body to lidar frame
    
    Returns:
        world_points: (N, 3) array of points in world frame
    """
    # Create rotation matrix from quaternion (body frame orientation)
    body_rotation_matrix = R.from_quat(quaternion).as_matrix()
    
    # Create 4x4 transformation matrix for body frame in world
    body_to_world = np.eye(4)
    body_to_world[:3, :3] = body_rotation_matrix
    body_to_world[:3, 3] = position
    
    # If body_to_lidar transform is provided, compose the transformations
    # world_to_lidar = world_to_body * body_to_lidar
    # So: lidar_to_world = (body_to_lidar)^-1 * body_to_world
    if body_to_lidar_tf is not None:
        # Transform from body to lidar, then from body to world
        # T_lidar_to_world = T_body_to_world * T_lidar_to_body
        # T_lidar_to_body = inv(T_body_to_lidar)
        lidar_to_body = np.linalg.inv(body_to_lidar_tf)
        transform_matrix = body_to_world @ lidar_to_body
    else:
        transform_matrix = body_to_world
    
    # Transform points to world coordinates
    points_homogeneous = np.hstack(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)]
    )
    world_points = (transform_matrix @ points_homogeneous.T).T
    world_points_xyz = world_points[:, :3]
    
    return world_points_xyz


def compare_gt_inferred_map(dataset_path, seq_name, label_config,
                            max_scans=None, downsample_factor=1, voxel_size=0.1, max_distance=None,
                            view_mode="all", view_variance=False, use_second_above_median=False):
    """
    Build GT map (world points + gt labels) and inferred map (world points + dominant class).
    For each inferred point, find nearest GT point via KD-tree and compare labels.
    view_mode: "all" | "correct" | "incorrect" — show all inferred points, or only correct/incorrect.
    view_variance: If True, color by variance of class probabilities (Viridis: yellow=uncertain, dark=confident).
    """
    root_path = os.path.join(dataset_path, seq_name)
    data_dir = os.path.join(root_path, "lidar_bin/data")
    timestamps_file = os.path.join(root_path, "lidar_bin/timestamps.txt")
    poses_file = os.path.join(root_path, "pose_inW.csv")
    gt_labels_dir = os.path.join(root_path, "gt_labels")
    multiclass_dir = os.path.join(root_path, "inferred_labels", "cenet_mcd", "multiclass_confidence_scores")

    learning_map_inv = label_config["learning_map_inv"]
    label_id_to_color = label_config["color_map_rgb"]

    if not os.path.exists(data_dir) or not os.path.exists(poses_file):
        print("ERROR: Data directory or poses file not found.")
        return
    if not os.path.exists(gt_labels_dir):
        print(f"ERROR: GT labels directory not found: {gt_labels_dir}")
        return
    if not os.path.exists(multiclass_dir):
        print(f"ERROR: Multiclass directory not found: {multiclass_dir}")
        return
    if os.path.exists(timestamps_file):
        timestamps = np.loadtxt(timestamps_file)
    else:
        timestamps = np.array([])

    poses_dict, index_to_timestamp = load_poses(poses_file)
    if not poses_dict or not index_to_timestamp:
        print("ERROR: No poses loaded or no index column.")
        return

    sorted_pose_indices = sorted(index_to_timestamp.keys())
    if downsample_factor > 1:
        sorted_pose_indices = sorted_pose_indices[::downsample_factor]
    if max_scans:
        sorted_pose_indices = sorted_pose_indices[:max_scans]

    all_gt_points = []
    all_gt_labels = []
    all_inf_points = []
    all_inf_labels = []
    all_inf_labels_second = []  # second-highest class (for --use-second-above-median)
    all_inf_variance = []  # variance of class probs (always computed, for reporting)
    all_inf_uncertainty = []  # 0=confident, 1=uncertain (for variance coloring when --variance)

    for pose_num in tqdm(sorted_pose_indices, desc="Loading scans", unit="scan"):
        pose_timestamp = index_to_timestamp[pose_num]
        pose_data = poses_dict[pose_timestamp]
        position = pose_data[1:4]
        quaternion = pose_data[4:8]
        bin_file = f"{pose_num:010d}.bin"
        bin_path = os.path.join(data_dir, bin_file)
        gt_path = os.path.join(gt_labels_dir, bin_file)
        multiclass_path = os.path.join(multiclass_dir, bin_file)
        if not os.path.exists(bin_path) or not os.path.exists(gt_path) or not os.path.exists(multiclass_path):
            continue
        if len(timestamps) > 0 and pose_num - 1 < len(timestamps) and abs(timestamps[pose_num - 1] - pose_timestamp) > 0.1:
            continue
        try:
            points = read_bin_file(bin_path, dtype=np.float32, shape=(-1, 4))
            points_xyz = points[:, :3]
            gt_lbl = read_bin_file(gt_path, dtype=np.int32, shape=(-1))
            raw_probs = read_bin_file(multiclass_path, dtype=np.float16)
            n_points = len(points_xyz)
            n_classes = len(raw_probs) // n_points
            if len(gt_lbl) != n_points or len(raw_probs) != n_points * n_classes:
                continue
            multiclass_probs = raw_probs.reshape(n_points, n_classes)
            class_indices = np.argmax(multiclass_probs, axis=1)
            inf_lbl = map_class_indices_to_labels(class_indices, learning_map_inv)
            # Second-highest class (for "total accuracy after switching" report and --use-second-above-median)
            second_class_indices = np.argsort(multiclass_probs, axis=1)[:, -2]
            inf_lbl_second = map_class_indices_to_labels(second_class_indices, learning_map_inv)
            # Variance of class probs (always store for average-variance report)
            variances = np.var(multiclass_probs.astype(np.float32), axis=1)
            # Uncertainty for optional variance coloring (yellow=uncertain, dark=confident)
            if view_variance:
                max_var = (n_classes - 1) / (n_classes ** 2) if n_classes > 1 else 1.0
                scaled = np.clip(variances / max_var, 0.0, 1.0)
                uncertainty = 1.0 - scaled
            else:
                uncertainty = None
        except Exception as e:
            tqdm.write(f"  Error {bin_file}: {e}")
            continue

        world = transform_points_to_world(points_xyz, position, quaternion, BODY_TO_LIDAR_TF)
        if max_distance is not None and max_distance > 0:
            dist_mask = np.linalg.norm(world - position, axis=1) <= max_distance
            world = world[dist_mask]
            gt_lbl = gt_lbl[dist_mask]
            inf_lbl = inf_lbl[dist_mask]
            inf_lbl_second = inf_lbl_second[dist_mask]
            variances = variances[dist_mask]
            if uncertainty is not None:
                uncertainty = uncertainty[dist_mask]
        if voxel_size is not None and voxel_size > 0 and len(world) > 0:
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(world.astype(np.float64))
            scan_pcd = scan_pcd.voxel_down_sample(voxel_size=voxel_size)
            world_ds = np.asarray(scan_pcd.points)
            if len(world_ds) == 0:
                continue
            tree_scan = cKDTree(world)
            _, idx = tree_scan.query(world_ds, k=1)
            gt_lbl = gt_lbl[idx]
            inf_lbl = inf_lbl[idx]
            inf_lbl_second = inf_lbl_second[idx]
            variances = variances[idx]
            if uncertainty is not None:
                uncertainty = uncertainty[idx]
            world = world_ds
        all_gt_points.append(world.astype(np.float64))
        all_gt_labels.append(gt_lbl)
        all_inf_points.append(world.astype(np.float64))
        all_inf_labels.append(inf_lbl)
        all_inf_labels_second.append(inf_lbl_second)
        all_inf_variance.append(variances)
        if view_variance:
            all_inf_uncertainty.append(uncertainty)

    if not all_gt_points:
        print("ERROR: No points accumulated.")
        return
    gt_points = np.vstack(all_gt_points)
    gt_labels = np.concatenate(all_gt_labels)
    inf_points = np.vstack(all_inf_points)
    inf_labels = np.concatenate(all_inf_labels)
    inf_labels_second = np.concatenate(all_inf_labels_second)
    inf_variance = np.concatenate(all_inf_variance)
    inf_uncertainty = np.concatenate(all_inf_uncertainty) if all_inf_uncertainty else None
    assert len(gt_points) == len(gt_labels) == len(inf_points) == len(inf_labels) == len(inf_variance)
    assert len(inf_labels_second) == len(inf_labels)

    print("Building KD-tree on GT map...")
    tree = cKDTree(gt_points)
    print("Querying nearest GT neighbor for each inferred point...")
    _, nn_idx = tree.query(inf_points, k=1)
    gt_label_matched = gt_labels[nn_idx]
    correct = (inf_labels == gt_label_matched)
    n_correct = int(np.sum(correct))
    n_total = len(correct)
    print(f"Accuracy: {n_correct}/{n_total} ({100.0 * n_correct / n_total:.2f}%)")

    # Uncertainty from variance (0=confident, 1=uncertain) for all points
    n_classes = len(learning_map_inv)
    max_var = (n_classes - 1) / (n_classes ** 2) if n_classes > 1 else 1.0
    scaled_var = np.clip(inf_variance / max_var, 0.0, 1.0)
    uncertainty_all = 1.0 - scaled_var
    median_uncertainty = float(np.median(uncertainty_all))
    print(f"Median uncertainty: {median_uncertainty:.6f}")

    # Total accuracy of entire inferred map after switching above-median points to second-best label
    modified_labels = np.where(uncertainty_all > median_uncertainty, inf_labels_second, inf_labels)
    correct_modified = (modified_labels == gt_label_matched)
    n_correct_modified = int(np.sum(correct_modified))
    accuracy_modified = 100.0 * n_correct_modified / n_total
    print(f"Total accuracy of entire inferred map (after switching to second-best for above-median): {n_correct_modified}/{n_total} ({accuracy_modified:.2f}%)")

    if use_second_above_median:
        n_above_median = int(np.sum(uncertainty_all > median_uncertainty))
        print(f"Using second-highest class for points above median uncertainty: n={n_above_median} points switched")
    else:
        # Remove points with uncertainty > median; compute accuracy on remaining (low-uncertainty) points
        low_uncertainty_mask = (uncertainty_all <= median_uncertainty)
        n_kept = int(np.sum(low_uncertainty_mask))
        n_removed = n_total - n_kept
        correct_kept = correct[low_uncertainty_mask]
        n_correct_kept = int(np.sum(correct_kept))
        accuracy_kept = 100.0 * n_correct_kept / n_kept if n_kept > 0 else 0.0
        print(f"Keeping points with uncertainty <= median: n={n_kept} (removed {n_removed} high-uncertainty points)")
        print(f"Accuracy (inferred map after removing high-uncertainty points): {n_correct_kept}/{n_kept} ({accuracy_kept:.2f}%)")

    # --- Uncertainty calibration: is the model's uncertainty predictive of errors? ---
    incorrect = np.asarray(~correct, dtype=np.float64)  # 1 = wrong, 0 = correct
    rho, p_val = spearmanr(uncertainty_all, incorrect)
    print(f"\n--- Uncertainty calibration ---")
    print(f"Spearman(uncertainty, incorrect): ρ = {rho:.4f} (p = {p_val:.2e})")
    print(f"  → Positive ρ means higher uncertainty correlates with more errors (good).")

    # AUROC: can uncertainty discriminate correct vs incorrect? (1 = perfect, 0.5 = random)
    n_incorrect = int(np.sum(incorrect))
    n_correct_count = n_total - n_incorrect
    if n_incorrect > 0 and n_correct_count > 0:
        ranks = rankdata(uncertainty_all)  # rank 1 = lowest uncertainty
        S = np.sum(ranks[~correct])
        auroc = (S - n_incorrect * (n_incorrect + 1) / 2) / (n_incorrect * n_correct_count)
        print(f"AUROC(uncertainty → predicts error): {auroc:.4f} (1 = perfect, 0.5 = no better than random)")
    else:
        auroc = None

    # Accuracy by uncertainty bin (well-calibrated = accuracy decreases as uncertainty increases)
    n_bins = 10
    bin_edges = np.percentile(uncertainty_all, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-9  # include max
    print(f"\nAccuracy by uncertainty bin (lower bin = more confident):")
    bin_accs = []
    bin_centers = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        b = (uncertainty_all >= lo) & (uncertainty_all < hi)
        n_b = int(np.sum(b))
        if n_b > 0:
            acc_b = 100.0 * np.sum(correct[b]) / n_b
            bin_accs.append(acc_b)
            bin_centers.append((lo + hi) / 2)
            print(f"  bin {i+1:2d} [unc {lo:.3f}-{hi:.3f}]: acc = {acc_b:.1f}%  (n={n_b})")
    if bin_accs:
        # Plot: accuracy vs mean uncertainty per bin (well-calibrated = decreasing)
        fig_cal, ax_cal = plt.subplots(figsize=(7, 4))
        ax_cal.plot(bin_centers, bin_accs, "o-", color="steelblue", linewidth=2, markersize=8)
        ax_cal.set_xlabel("Mean uncertainty (bin)")
        ax_cal.set_ylabel("Accuracy (%)")
        ax_cal.set_title("Accuracy by uncertainty bin (well-calibrated model: curve decreases)")
        ax_cal.set_ylim(0, 105)
        ax_cal.grid(True, alpha=0.3)
        fig_cal.tight_layout()
        plt.show(block=True)
        plt.close(fig_cal)

    if view_mode == "correct":
        mask = correct
    elif view_mode == "incorrect":
        mask = ~correct
    else:
        mask = np.ones(len(correct), dtype=bool)
    n_show = int(np.sum(mask))
    print(f"Showing {n_show} points (view={view_mode})")
    avg_var = float(np.mean(inf_variance[mask]))
    print(f"Average variance ({view_mode}): {avg_var:.6f}")

    # Plot variance distribution for the selected subset
    var_subset = inf_variance[mask]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(var_subset, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(avg_var, color="coral", linewidth=2, label=f"Mean = {avg_var:.4f}")
    ax.set_xlabel("Variance of class probabilities")
    ax.set_ylabel("Count")
    ax.set_title(f"Variance distribution (view={view_mode}, n={n_show})")
    ax.legend()
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    pts = inf_points[mask]
    lbl = inf_labels[mask]
    if view_variance and inf_uncertainty is not None:
        unc = inf_uncertainty[mask]
        colors = scalar_to_viridis_rgb(unc, normalize_range=False)
    else:
        colors = labels_to_colors(lbl, label_id_to_color, confidences=None)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare GT labels vs inferred (dominant class): build maps, match via KD-tree, show accuracy. "
                    "Option to view all / correct / incorrect inferred points."
    )
    parser.add_argument("--dataset-path", type=str, default="/media/donceykong/doncey_ssd_02/datasets/MCD")
    parser.add_argument("--seq", type=str, nargs="+", default=["kth_day_09"], help="Sequence name(s)")
    parser.add_argument("--view", type=str, choices=["all", "correct", "incorrect"], default="all",
                        help="Show all inferred points, only correct, or only incorrect (default: all)")
    parser.add_argument("--max-scans", type=int, default=5000, help="Max scans to process (0 or None for all)")
    parser.add_argument("--downsample-factor", type=int, default=20, help="Process every Nth scan")
    parser.add_argument("--voxel-size", type=float, default=2.0, help="Voxel size in meters (per-scan downsampling)")
    parser.add_argument("--max-distance", type=float, default=100.0, help="Max distance from pose to keep points (m)")
    parser.add_argument("--config", type=str, default=None, help="Path to MCD label config YAML (default: ce_net/config/data_cfg_mcd.yaml)")
    parser.add_argument("--variance", action="store_true", help="Color by variance of class probabilities (Viridis: yellow=uncertain, dark=confident)")
    parser.add_argument("--use-second-above-median", action="store_true",
                        help="For points with uncertainty above median, use second-highest class instead of removing; report accuracy on all points")
    args = parser.parse_args()

    config_path = args.config or DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)
    label_config = load_label_config(config_path)

    max_scans = args.max_scans if args.max_scans else None
    for seq_name in args.seq:
        compare_gt_inferred_map(
            args.dataset_path,
            seq_name,
            label_config,
            max_scans=max_scans,
            downsample_factor=args.downsample_factor,
            voxel_size=args.voxel_size,
            max_distance=args.max_distance,
            view_mode=args.view,
            view_variance=args.variance,
            use_second_above_median=args.use_second_above_median,
        )