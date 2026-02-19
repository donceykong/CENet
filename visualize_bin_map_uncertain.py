#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import yaml
from tqdm import tqdm

# Import dataset_binarize package to set up sys.path for lidar2osm imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ce_net.utils.file_io import read_bin_file

# MCD label config path (same as view_multiclass_semantics)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "ce_net", "config", "data_cfg_mcd.yaml")

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
        rng = 1.0
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


def plot_map(dataset_path, seq_name, label_config, single_label_id=None,
             max_scans=None, downsample_factor=1, voxel_size=0.1, max_distance=None,
             single_label_normalize_range=True, value_floor=0.12, gamma=0.65,
             single_label_grayscale=False):
    """
    Load lidar bin files with multiclass confidence scores, transform using poses, and visualize with Open3D.
    Uses only multiclass_confidence_scores (float16 [N, C]); no single-channel label files.
    
    Args:
        dataset_path: Path to dataset directory
        seq_name: Name of the sequence
        label_config: Dict from load_label_config() (labels, color_map_rgb, learning_map_inv)
        single_label_id: If set, visualize one label's confidence. Else argmax colors.
        single_label_normalize_range: If True, map single-label confidence range to [0,1] for visibility.
        value_floor: Minimum brightness (0..1). Avoids pure black.
        gamma: Gamma for mid-tone contrast (<1 brightens mid-tones).
    """
    root_path = os.path.join(dataset_path, seq_name)
    data_dir = os.path.join(root_path, "lidar_bin/data")
    timestamps_file = os.path.join(root_path, "lidar_bin/timestamps.txt")
    poses_file = os.path.join(root_path, "pose_inW.csv")
    multiclass_dir = os.path.join(root_path, "inferred_labels", "cenet_mcd", "multiclass_confidence_scores")

    learning_map_inv = label_config["learning_map_inv"]
    label_id_to_color = label_config["color_map_rgb"]

    # Check if files exist
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        return
    
    if not os.path.exists(timestamps_file):
        print(f"ERROR: Timestamps file not found: {timestamps_file}")
        return
    
    if not os.path.exists(poses_file):
        print(f"ERROR: Poses file not found: {poses_file}")
        return
    
    if not os.path.exists(multiclass_dir):
        print(f"ERROR: Multiclass labels directory not found: {multiclass_dir}")
        return
    
    # Load timestamps
    print(f"\nLoading timestamps from {timestamps_file}")
    timestamps = np.loadtxt(timestamps_file)
    print(f"Loaded {len(timestamps)} timestamps")
    
    # Load poses
    poses_dict, index_to_timestamp = load_poses(poses_file)
    if not poses_dict:
        print("ERROR: No poses loaded")
        return
    
    if not index_to_timestamp:
        print("ERROR: No index column found in poses CSV. Cannot validate index matching.")
        return
    
    # Process poses in order by their index (num value)
    sorted_pose_indices = sorted(index_to_timestamp.keys())
    
    # Apply downsampling to pose indices (process every Nth pose)
    if downsample_factor > 1:
        sorted_pose_indices = sorted_pose_indices[::downsample_factor]
        print(f"Downsampling: Processing every {downsample_factor} pose (total: {len(sorted_pose_indices)} poses)")
    
    # Limit number of poses if specified
    if max_scans:
        sorted_pose_indices = sorted_pose_indices[:max_scans]
        print(f"Limiting to {max_scans} poses")
    
    print(f"Processing {len(sorted_pose_indices)} scans...")
    
    if max_distance is not None:
        print(f"Filtering points beyond {max_distance}m from each pose position")
    
    # Accumulate all points and colors
    all_world_points = []
    all_colors = []
    
    # Process poses with progress bar
    for pose_num in tqdm(sorted_pose_indices, desc="Processing scans", unit="scan"):
        # Get pose data
        pose_timestamp = index_to_timestamp[pose_num]
        pose_data = poses_dict[pose_timestamp]
        position = pose_data[1:4]  # [x, y, z]
        quaternion = pose_data[4:8]  # [qx, qy, qz, qw]
        
        # Construct bin file name directly from pose_num (1-indexed)
        # Bin files are named 0000000001.bin, 0000000002.bin, etc.
        bin_file = f"{pose_num:010d}.bin"
        bin_path = os.path.join(data_dir, bin_file)
        
        # Check if bin file exists
        if not os.path.exists(bin_path):
            tqdm.write(f"  WARNING: Bin file not found for pose {pose_num}: {bin_file}")
            continue
        
        # Get corresponding timestamp (pose_num is 1-indexed, timestamps array is 0-indexed)
        scan_timestamp = timestamps[pose_num - 1]
        
        # Validate timestamps match (within threshold)
        time_diff = abs(scan_timestamp - pose_timestamp)
        if time_diff > 0.1:  # 100ms threshold
            raise ValueError(
                f"Timestamp mismatch at timestamp index {pose_num - 1} (pose index {pose_num}): "
                f"Scan timestamp is {scan_timestamp:.6f} but pose timestamp is {pose_timestamp:.6f} "
                f"(difference: {time_diff:.6f}s)"
            )
        
        # Load point cloud (bin_path already constructed above)
        try:
            points = read_bin_file(bin_path, dtype=np.float32, shape=(-1, 4))
            points_xyz = points[:, :3]
        except Exception as e:
            tqdm.write(f"  Error loading {bin_file}: {e}")
            continue
        
        # Load multiclass confidence scores (float16, shape [n_points * n_classes])
        multiclass_path = os.path.join(multiclass_dir, bin_file)
        if not os.path.exists(multiclass_path):
            tqdm.write(f"  WARNING: Multiclass file not found: {bin_file}, skipping scan")
            continue
        
        try:
            raw_probs = read_bin_file(multiclass_path, dtype=np.float16)
            n_points = len(points_xyz)
            n_classes = len(raw_probs) // n_points
            if len(raw_probs) != n_points * n_classes:
                tqdm.write(f"  WARNING: Multiclass size {len(raw_probs)} != {n_points} * n_classes for {bin_file}, skipping")
                continue
            multiclass_probs = raw_probs.reshape(n_points, n_classes)
            
            if single_label_id is not None:
                label_id_to_class_idx = get_label_id_to_class_index(learning_map_inv)
                if single_label_id not in label_id_to_class_idx:
                    tqdm.write(f"  WARNING: No class index for label id {single_label_id}, skipping scan")
                    continue
                class_idx = label_id_to_class_idx[single_label_id]
                confidences = np.asarray(multiclass_probs[:, class_idx], dtype=np.float32)
                colors = single_label_confidence_to_colors(
                    confidences, single_label_id, label_id_to_color,
                    normalize_range=single_label_normalize_range,
                    value_floor=value_floor, gamma=gamma,
                    grayscale=single_label_grayscale,
                )
            else:
                class_indices = np.argmax(multiclass_probs, axis=1)
                confidences = np.max(multiclass_probs, axis=1)
                semantic_label_ids = map_class_indices_to_labels(class_indices, learning_map_inv)
                colors = labels_to_colors(
                    semantic_label_ids, label_id_to_color, confidences=confidences,
                    value_floor=value_floor, gamma=gamma,
                )
        except Exception as e:
            tqdm.write(f"  Error loading multiclass from {bin_file}: {e}")
            continue
        
        # Transform to world coordinates (apply body-to-lidar transformation)
        world_points = transform_points_to_world(points_xyz, position, quaternion, BODY_TO_LIDAR_TF)
        
        # Filter points by distance from pose position if max_distance is specified
        if max_distance is not None and max_distance > 0:
            # Calculate distance from pose position for each point
            distances = np.linalg.norm(world_points - position, axis=1)
            # Keep points within max_distance
            mask = distances <= max_distance
            world_points = world_points[mask]
            colors = colors[mask]
            
            if len(world_points) == 0:
                tqdm.write(f"  WARNING: All points filtered out for {bin_file} (max_distance={max_distance}m)")
                continue
        
        # Voxel downsample this scan before accumulating
        if voxel_size is not None and voxel_size > 0:
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(world_points)
            scan_pcd.colors = o3d.utility.Vector3dVector(colors)
            scan_pcd = scan_pcd.voxel_down_sample(voxel_size=voxel_size)
            
            # Extract downsampled points and colors
            world_points = np.asarray(scan_pcd.points)
            colors = np.asarray(scan_pcd.colors)
        
        all_world_points.append(world_points)
        all_colors.append(colors)
    
    if not all_world_points:
        print("ERROR: No points accumulated")
        return
    
    # Concatenate all points and colors
    print(f"\nAccumulating {len(all_world_points)} scans...")
    accumulated_points = np.vstack(all_world_points)
    accumulated_colors = np.vstack(all_colors)
    
    print(f"Total points: {len(accumulated_points)}")
    
    # Create Open3D point cloud with semantic colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(accumulated_points)
    pcd.colors = o3d.utility.Vector3dVector(accumulated_colors)
    
    # Visualize
    print("\nVisualizing point cloud...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Shift + Mouse: Pan view")
    print("  - Mouse wheel: Zoom")
    print("  - Q or ESC: Quit")
    
    if single_label_grayscale:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        vis.run()
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize accumulated map from lidar + multiclass confidence scores. "
                    "Optionally show single-label confidence (--single-label)."
    )
    parser.add_argument("--dataset-path", type=str, default="/media/donceykong/doncey_ssd_02/datasets/MCD")
    parser.add_argument("--seq", type=str, nargs="+", default=["kth_day_09"], help="Sequence name(s)")
    parser.add_argument(
        "--single-label",
        type=str,
        default=None,
        metavar="NAME_OR_ID",
        help="View confidence for one label only (e.g. vehicle-static or 28). Full color = high, black = zero.",
    )
    parser.add_argument("--max-scans", type=int, default=5000, help="Max scans to process (0 or None for all)")
    parser.add_argument("--downsample-factor", type=int, default=200, help="Process every Nth scan")
    parser.add_argument("--voxel-size", type=float, default=2.0, help="Voxel size in meters")
    parser.add_argument("--max-distance", type=float, default=100.0, help="Max distance from pose to keep points (m)")
    parser.add_argument("--config", type=str, default=None, help="Path to MCD label config YAML (default: ce_net/config/data_cfg_mcd.yaml)")
    parser.add_argument("--no-normalize", action="store_true", help="Single-label: use raw probability (0=black, 1=full) instead of normalizing range")
    parser.add_argument("--value-floor", type=float, default=0.12, metavar="F", help="Minimum brightness 0..1 (default 0.12); with --grayscale points use 0 (white to black)")
    parser.add_argument("--gamma", type=float, default=0.65, metavar="G", help="Gamma for mid-tone contrast (default 0.65; <1 brightens mid-tones)")
    parser.add_argument("--grayscale", action="store_true", help="Single-label: points white (high) to black (low); scene background gray")
    args = parser.parse_args()

    config_path = args.config or DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)
    label_config = load_label_config(config_path)

    single_label_id = None
    if args.single_label is not None:
        single_label_id, single_label_name = _parse_single_label(args.single_label, label_config)
        if single_label_id is None:
            print(f"ERROR: Unknown label '{args.single_label}'. Use a label name or id from the config.")
            sys.exit(1)
        print(f"Single-label mode: showing confidence for '{single_label_name}' (id={single_label_id})")

    # With --grayscale, points are white→black (value_floor=0); scene background is set to gray in the viewer
    value_floor = args.value_floor
    if args.grayscale and single_label_id is not None:
        value_floor = 0.0

    max_scans = args.max_scans if args.max_scans else None
    seq_names = args.seq

    for seq_name in seq_names:
        plot_map(
            args.dataset_path,
            seq_name,
            label_config,
            single_label_id=single_label_id,
            max_scans=max_scans,
            downsample_factor=args.downsample_factor,
            voxel_size=args.voxel_size,
            max_distance=args.max_distance,
            single_label_normalize_range=not args.no_normalize,
            value_floor=value_floor,
            gamma=args.gamma,
            single_label_grayscale=args.grayscale,
        )