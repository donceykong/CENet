#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from tqdm import tqdm
from pypcd import pypcd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import namedtuple

# Import dataset_binarize package to set up sys.path for lidar2osm imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Body to LiDAR transformation matrix
BODY_TO_LIDAR_TF = np.array([
    [0.9999135040741837, -0.011166365511073898, -0.006949579221822984, -0.04894521120494695],
    [-0.011356389542502144, -0.9995453006865824, -0.02793249526856565, -0.03126929060348084],
    [-0.006634514801117132, 0.02800900135032654, -0.999585653686922, -0.01755515794222565],
    [0.0, 0.0, 0.0, 1.0]
])

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
    Label("noise", 11, (255, 0, 0)),  # Placeholder color, needs assignment
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

# Create a dictionary mapping label ID to Label name for backward compatibility
SEMANTIC_LABELS = {label.id: label.name for label in semantic_labels}

def labels_to_colors(labels):
    """
    Convert semantic label IDs to RGB colors.
    
    Args:
        labels: (N,) array of semantic label IDs (0-28)
    
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


def process_sequence(root_path, max_scans=None, downsample_factor=1, voxel_size=0.1):
    """
    Load PCD files with semantics, transform using poses, and accumulate points.
    
    Args:
        root_path: Path to root directory (should contain gt_lidar_labels/ with pose_inW.csv and inL_labelled/)
        max_scans: Maximum number of scans to process (None for all)
        downsample_factor: Process every Nth scan (1 = all scans)
        voxel_size: Voxel size in meters for downsampling (default: 0.1m, set to None to disable)
    
    Returns:
        accumulated_points: (N, 3) numpy array of points in world coordinates
        accumulated_labels: (N,) numpy array of semantic label IDs (int32)
    """
    pcd_dir = os.path.join(root_path, "gt_lidar_labels", "inL_labelled")
    poses_file = os.path.join(root_path, "gt_lidar_labels", "pose_inW.csv")
    
    # Check if files exist
    if not os.path.exists(pcd_dir):
        print(f"ERROR: PCD directory not found: {pcd_dir}")
        return
    
    if not os.path.exists(poses_file):
        print(f"ERROR: Poses file not found: {poses_file}")
        return
    
    # Load poses
    poses_dict, index_to_timestamp = load_poses(poses_file)
    if not poses_dict:
        print("ERROR: No poses loaded")
        return
    
    if not index_to_timestamp:
        print("ERROR: No index column found in poses CSV. Cannot validate index matching.")
        return
    
    # Get all PCD files (named like cloud_<index>.pcd)
    all_pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd') and f.startswith('cloud_')])
    print(f"\nFound {len(all_pcd_files)} PCD files")
    
    # Extract indices from filenames (cloud_<index>.pcd)
    pcd_indices = []
    pcd_files_dict = {}
    for pcd_file in all_pcd_files:
        try:
            # Extract index from filename: cloud_<index>.pcd
            index_str = pcd_file.replace('cloud_', '').replace('.pcd', '')
            index = int(index_str)
            pcd_indices.append(index)
            pcd_files_dict[index] = pcd_file
        except ValueError:
            tqdm.write(f"  WARNING: Could not parse index from filename: {pcd_file}")
            continue
    
    pcd_indices = sorted(pcd_indices)
    print(f"Found {len(pcd_indices)} valid PCD files with indices")
    
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
    
    # Accumulate all points and labels
    all_world_points = []
    all_labels = []
    
    # Process poses with progress bar
    for pose_num in tqdm(sorted_pose_indices, desc="Processing scans", unit="scan"):
        # Check if PCD file exists for this pose index
        if pose_num not in pcd_files_dict:
            tqdm.write(f"  WARNING: Pose index {pose_num} has no corresponding PCD file")
            continue
        
        # Get pose data
        pose_timestamp = index_to_timestamp[pose_num]
        pose_data = poses_dict[pose_timestamp]
        position = pose_data[1:4]  # [x, y, z]
        quaternion = pose_data[4:8]  # [qx, qy, qz, qw]
        
        # Load PCD file using pypcd
        pcd_file = pcd_files_dict[pose_num]
        pcd_path = os.path.join(pcd_dir, pcd_file)
        try:
            # Read PCD file using pypcd
            pc = pypcd.PointCloud.from_path(pcd_path)
            
            # Extract points as numpy array
            points_xyz = np.column_stack([pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z']]).astype(np.float32)
            
            if len(points_xyz) == 0:
                tqdm.write(f"  WARNING: Empty point cloud in {pcd_file}")
                continue
            
            # Check available fields in PCD
            available_fields = pc.pc_data.dtype.names
            if pose_num == sorted_pose_indices[0]:  # Print info for first file only
                tqdm.write(f"  PCD fields: {available_fields}")
            
            # Try to extract labels/class information
            labels = None
            label_field_name = None
            
            # Check for common label field names
            for field_name in ['label', 'class', 'semantic', 'labels', 'classes', 'semantic_id']:
                if field_name in available_fields:
                    labels = pc.pc_data[field_name].astype(np.int32)
                    label_field_name = field_name
                    if pose_num == sorted_pose_indices[0]:  # Print info for first file only
                        unique_labels = np.unique(labels)
                        tqdm.write(f"  Found label field '{field_name}' with {len(unique_labels)} unique labels: {unique_labels[:10]}")
                    break
            
            # Extract labels if available
            if labels is None:
                # Try to extract labels from color fields if no label field exists
                # This is a fallback - we prefer explicit label fields
                if pose_num == sorted_pose_indices[0]:  # Print warning for first file only
                    tqdm.write(f"  WARNING: No label field found in {pcd_file}, skipping scan")
                continue
            
            # Ensure labels are int32
            labels = labels.astype(np.int32)
        except Exception as e:
            tqdm.write(f"  Error loading {pcd_file}: {e}")
            import traceback
            tqdm.write(traceback.format_exc())
            continue
        
        # Transform to world coordinates (apply body-to-lidar transformation)
        world_points = transform_points_to_world(points_xyz, position, quaternion, BODY_TO_LIDAR_TF)
        
        # Voxel downsample this scan before accumulating
        if voxel_size is not None and voxel_size > 0:
            # Downsample points and labels together by grouping into voxels
            # Round points to voxel grid
            voxel_coords = np.round(world_points / voxel_size).astype(np.int32)
            
            # Group points by voxel and get mode label for each voxel
            from collections import defaultdict
            voxel_to_points = defaultdict(list)
            voxel_to_labels = defaultdict(list)
            
            for i, voxel_coord in enumerate(voxel_coords):
                voxel_key = tuple(voxel_coord)
                voxel_to_points[voxel_key].append(world_points[i])
                voxel_to_labels[voxel_key].append(labels[i])
            
            # Get unique voxel centers and their mode labels
            unique_voxels = []
            voxel_labels = []
            for voxel_key in voxel_to_points.keys():
                # Use voxel center as point location
                voxel_center = np.array(voxel_key, dtype=np.float32) * voxel_size
                unique_voxels.append(voxel_center)
                
                # Get mode (most common) label for this voxel
                label_list = voxel_to_labels[voxel_key]
                unique, counts = np.unique(label_list, return_counts=True)
                mode_label = unique[np.argmax(counts)]
                voxel_labels.append(mode_label)
            
            world_points = np.array(unique_voxels, dtype=np.float32)
            labels = np.array(voxel_labels, dtype=np.int32)
        
        all_world_points.append(world_points)
        all_labels.append(labels)
    
    if not all_world_points:
        print("ERROR: No points accumulated")
        return None, None
    
    # Concatenate all points and labels
    print(f"\nAccumulating {len(all_world_points)} scans...")
    accumulated_points = np.vstack(all_world_points)
    accumulated_labels = np.concatenate(all_labels)
    
    print(f"Total points: {len(accumulated_points)}")
    
    return accumulated_points, accumulated_labels


def merge_sequences(dataset_path, seq_names, output_npy_path, max_scans=None, downsample_factor=1, voxel_size=0.1):
    """
    Process multiple sequences and merge them into a single numpy file with x, y, z, label.
    
    Args:
        dataset_path: Path to dataset directory containing sequence folders
        seq_names: List of sequence names to process
        output_npy_path: Path to output .npy file (will be created) with shape (N, 4) where columns are [x, y, z, label]
        max_scans: Maximum number of scans to process per sequence (None for all)
        downsample_factor: Process every Nth scan (1 = all scans)
        voxel_size: Voxel size in meters for downsampling (default: 0.1m, set to None to disable)
    """
    print(f"\n{'='*80}")
    print(f"Merging {len(seq_names)} sequences into single numpy file")
    print(f"{'='*80}")
    print(f"Sequences: {seq_names}")
    print(f"Output file: {output_npy_path}")
    print(f"{'='*80}\n")
    
    all_merged_points = []
    all_merged_labels = []
    
    # Process each sequence
    for seq_idx, seq_name in enumerate(seq_names):
        print(f"\n{'='*80}")
        print(f"Processing sequence {seq_idx + 1}/{len(seq_names)}: {seq_name}")
        print(f"{'='*80}")
        
        root_path = os.path.join(dataset_path, seq_name)
        
        # Process this sequence
        points, labels = process_sequence(
            root_path,
            max_scans=max_scans,
            downsample_factor=downsample_factor,
            voxel_size=voxel_size
        )
        
        if points is not None and labels is not None:
            all_merged_points.append(points)
            all_merged_labels.append(labels)
            print(f"  Added {len(points)} points from {seq_name}")
        else:
            print(f"  WARNING: No points from {seq_name}, skipping")
    
    if not all_merged_points:
        print("\nERROR: No points accumulated from any sequence!")
        return
    
    # Merge all sequences
    print(f"\n{'='*80}")
    print(f"Merging all sequences...")
    print(f"{'='*80}")
    merged_points = np.vstack(all_merged_points)
    merged_labels = np.concatenate(all_merged_labels)
    
    print(f"Total merged points: {len(merged_points)}")
    print(f"  X range: [{merged_points[:, 0].min():.2f}, {merged_points[:, 0].max():.2f}]")
    print(f"  Y range: [{merged_points[:, 1].min():.2f}, {merged_points[:, 1].max():.2f}]")
    print(f"  Z range: [{merged_points[:, 2].min():.2f}, {merged_points[:, 2].max():.2f}]")
    
    # Get label statistics
    unique_labels, counts = np.unique(merged_labels, return_counts=True)
    print(f"\nLabel statistics:")
    label_names = {label.id: label.name for label in semantic_labels}
    for label_id, count in zip(unique_labels, counts):
        label_name = label_names.get(int(label_id), f"unknown({label_id})")
        percentage = (count / len(merged_labels)) * 100.0
        print(f"  Label {label_id:2d} ({label_name:20s}): {count:10d} points ({percentage:5.2f}%)")
    
    # Combine points and labels into single array: [x, y, z, label]
    merged_data = np.hstack([merged_points, merged_labels.reshape(-1, 1)])
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_npy_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated directory: {output_dir}")
    
    # Save merged point cloud as numpy file
    print(f"\nSaving merged point cloud to: {output_npy_path}")
    print(f"  Shape: {merged_data.shape}, dtype: {merged_data.dtype}")
    print(f"  Columns: [x, y, z, label]")
    np.save(output_npy_path, merged_data)
    print(f"Successfully saved {len(merged_points)} points to {output_npy_path}")
    
    print(f"\n{'='*80}")
    print(f"DONE! Merged {len(seq_names)} sequences into single numpy file")
    print(f"{'='*80}")


if __name__ == '__main__':
    # Configuration
    dataset_path = "/media/donceykong/doncey_ssd_02/datasets/MCD"
    
    # List of sequences to merge
    seq_names = [
        "kth_day_06",
        "kth_day_09",
        "kth_night_05",
    ]
    
    # Output numpy file path
    # Will be saved in dataset_path/ply/merged_gt_labels_<seq_names>.npy
    seq_name_str = "_".join(seq_names)
    output_npy_path = os.path.join(dataset_path, "ply", f"merged_gt_labels_{seq_name_str}.npy")
    
    # Processing parameters
    max_scans = 10000  # Maximum number of scans per sequence (None for all)
    downsample_factor = 1  # Process every Nth scan (1 = all scans)
    voxel_size = 1.0  # Voxel size for downsampling (in meters)
    
    # Merge all sequences into single numpy file
    merge_sequences(
        dataset_path=dataset_path,
        seq_names=seq_names,
        output_npy_path=output_npy_path,
        max_scans=max_scans,
        downsample_factor=downsample_factor,
        voxel_size=voxel_size
    )

