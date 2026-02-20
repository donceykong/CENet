#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from tqdm import tqdm
from collections import namedtuple

# Project root (repo root containing ce_net): two levels up from test_scripts/mcd/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
from ce_net.utils.file_io import read_bin_file

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


Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        "id",  # An integer ID that is associated with this label.
        "color",  # The color of this label
    ],
)

sem_kitti_labels = [
    # name, id, color
    Label("unlabeled", 0, (0, 0, 0)),
    Label("outlier", 1, (0, 0, 0)),
    Label("car", 10, (0, 0, 142)),
    Label("bicycle", 11, (119, 11, 32)),
    Label("bus", 13, (250, 80, 100)),
    Label("motorcycle", 15, (0, 0, 230)),
    Label("on-rails", 16, (255, 0, 0)),
    Label("truck", 18, (0, 0, 70)),
    Label("other-vehicle", 20, (51, 0, 51)),
    Label("person", 30, (220, 20, 60)),
    Label("bicyclist", 31, (200, 40, 255)),
    Label("motorcyclist", 32, (90, 30, 150)),
    Label("road", 40, (128, 64, 128)),
    Label("parking", 44, (250, 170, 160)),
    Label("OSM BUILDING", 45, (0, 0, 255)),   # OSM
    Label("OSM ROAD", 46, (255, 0, 0)),       # OSM
    Label("sidewalk", 48, (244, 35, 232)),
    Label("other-ground", 49, (81, 0, 81)),
    Label("building", 50, (0, 100, 0)),
    Label("fence", 51, (190, 153, 153)),
    Label("other-structure", 52, (0, 150, 255)),
    Label("lane-marking", 60, (170, 255, 150)),
    Label("vegetation", 70, (107, 142, 35)),
    Label("trunk", 71, (0, 60, 135)),
    Label("terrain", 72, (152, 251, 152)),
    Label("pole", 80, (153, 153, 153)),
    Label("traffic-sign", 81, (0, 0, 255)),
    Label("other-object", 99, (255, 255, 50)),
]

# Create a mapping from sem_kitti label names to colors for reuse
_sem_kitti_color_map = {label.name: label.color for label in sem_kitti_labels}

# Semantic labels (converted from dictionary to namedtuple list)
# Colors are reused from sem_kitti_labels where names match
semantic_labels = [
    Label("barrier", 0, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("bike", 1, _sem_kitti_color_map["bicycle"]),  # Matches sem_kitti "bicycle"
    Label("building", 2, _sem_kitti_color_map["building"]),  # Matches sem_kitti "building"
    Label("chair", 3, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("cliff", 4, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("container", 5, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("curb", 6, _sem_kitti_color_map["sidewalk"]),  # Placeholder color, needs assignment
    Label("fence", 7, _sem_kitti_color_map["fence"]),  # Matches sem_kitti "fence"
    Label("hydrant", 8, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("infosign", 9, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("lanemarking", 10, _sem_kitti_color_map["lane-marking"]),  # Matches sem_kitti "lane-marking"
    Label("noise", 11, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("other", 12, _sem_kitti_color_map["other-object"]),  # Matches sem_kitti "other-object"
    Label("parkinglot", 13, _sem_kitti_color_map["parking"]),  # Matches sem_kitti "parking"
    Label("pedestrian", 14, _sem_kitti_color_map["person"]),  # Matches sem_kitti "person"
    Label("pole", 15, _sem_kitti_color_map["pole"]),  # Matches sem_kitti "pole"
    Label("road", 16, _sem_kitti_color_map["road"]),  # Matches sem_kitti "road"
    Label("shelter", 17, _sem_kitti_color_map["building"]),  # Placeholder color, needs assignment
    Label("sidewalk", 18, _sem_kitti_color_map["sidewalk"]),  # Matches sem_kitti "sidewalk"
    Label("stairs", 19, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("structure-other", 20, _sem_kitti_color_map["other-structure"]),  # Matches sem_kitti "other-structure"
    Label("traffic-cone", 21, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("traffic-sign", 22, _sem_kitti_color_map["traffic-sign"]),  # Matches sem_kitti "traffic-sign"
    Label("trashbin", 23, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("treetrunk", 24, _sem_kitti_color_map["trunk"]),  # Matches sem_kitti "trunk"
    Label("vegetation", 25, _sem_kitti_color_map["vegetation"]),  # Matches sem_kitti "vegetation"
    Label("vehicle-dynamic", 26, _sem_kitti_color_map["car"]),  # Matches sem_kitti "car"
    Label("vehicle-other", 27, _sem_kitti_color_map["car"]),  # Matches sem_kitti "other-vehicle"
    Label("vehicle-static", 28, _sem_kitti_color_map["car"]),  # Matches sem_kitti "car"
]

def labels_to_colors(labels):
    """
    Convert semantic label IDs to RGB colors using semantic_labels.
    
    Args:
        labels: (N,) array of semantic label IDs
    
    Returns:
        colors: (N, 3) array of RGB colors in [0, 1] range
    """
    # Create a mapping from label ID to color (RGB tuple in 0-255 range)
    label_id_to_color = {label.id: label.color for label in semantic_labels}
    
    # Map labels to colors
    colors = np.zeros((len(labels), 3), dtype=np.float32)
    for i, label_id in enumerate(labels):
        label_id_int = int(label_id)
        if label_id_int in label_id_to_color:
            # Convert from (0-255) range to (0-1) range
            color_255 = label_id_to_color[label_id_int]
            colors[i] = np.array(color_255, dtype=np.float32) / 255.0
        else:
            # Unknown label, use gray
            colors[i] = [0.5, 0.5, 0.5]
    
    return colors

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


def plot_map(dataset_path, seq_name, max_scans=None, downsample_factor=1, voxel_size=0.1, max_distance=None):
    """
    Load lidar bin files with semantic labels, transform using poses, and visualize with Open3D.
    
    Args:
        dataset_path: Path to dataset directory
        seq_name: Name of the sequence
        max_scans: Maximum number of scans to process (None for all)
        downsample_factor: Process every Nth scan (1 = all scans)
        voxel_size: Voxel size in meters for downsampling (default: 0.1m, set to None to disable)
        max_distance: Maximum distance in meters from origin to keep points (None to keep all points)
    """
    root_path = os.path.join(dataset_path, seq_name)
    data_dir = os.path.join(root_path, "lidar_bin/data")
    timestamps_file = os.path.join(root_path, "lidar_bin/timestamps.txt")
    poses_file = os.path.join(root_path, "pose_inW.csv")
    labels_dir = os.path.join(root_path, "inferred_labels", "cenet_mcd")
    # labels_dir = os.path.join(root_path, "gt_labels")

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
    
    if not os.path.exists(labels_dir):
        print(f"ERROR: Labels directory not found: {labels_dir}")
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
        
        # Load corresponding label file
        label_file = bin_file  # Label file should have same name as bin file
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(label_path):
            tqdm.write(f"  WARNING: Label file not found: {label_file}, skipping scan")
            continue
        
        try:
            # Load labels (assuming int32 format)
            labels = read_bin_file(label_path, dtype=np.int32, shape=(-1))
            
            # Validate that labels match points
            if len(labels) != len(points_xyz):
                tqdm.write(f"  WARNING: Label count ({len(labels)}) != point count ({len(points_xyz)}) for {bin_file}, skipping")
                continue
            
            # Convert labels to colors
            colors = labels_to_colors(labels)
        except Exception as e:
            tqdm.write(f"  Error loading labels from {label_file}: {e}")
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
    
    o3d.visualization.draw_geometries([pcd])
    
    # Save to PLY file
    # Extract sequence name from root_path (e.g., "kth_day_06" from "/path/to/MCD/kth_day_06")
    ply_dir = os.path.join(root_path, "ply")
    ply_filename = f"inferred_labels_{seq_name}.ply"
    ply_path = os.path.join(ply_dir, ply_filename)
    
    # Create ply directory if it doesn't exist
    if not os.path.exists(ply_dir):
        os.makedirs(ply_dir)
        print(f"Created directory: {ply_dir}")
    
    # Save point cloud
    print(f"\nSaving point cloud to: {ply_path}")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Successfully saved {len(accumulated_points)} points to {ply_path}")


if __name__ == '__main__':
    # Update these paths to match your setup
    dataset_path = "/media/donceykong/doncey_ssd_02/datasets/MCD"
    seq_names = ["kth_day_09"]

    # Optional: limit number of scans for faster visualization
    # max_scans = 100  # Set to None to process all scans
    max_scans = 5000
    
    # Optional: downsample scans (process every Nth scan)
    downsample_factor = 200  # Set to 1 to process all scans, 2 for every other scan, etc.
    
    # Voxel size for downsampling (in meters)
    voxel_size = 2.0  # 1m voxels
    
    # Maximum distance from origin to keep points (in meters)
    max_distance = 200.0  # e.g., 100.0 to keep points within 100m from origin

    for seq_name in seq_names:
        plot_map(dataset_path, seq_name, max_scans=max_scans, downsample_factor=downsample_factor, 
                 voxel_size=voxel_size, max_distance=max_distance)