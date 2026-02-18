#!/usr/bin/env python3
"""
Visualize semantic labels on a .bin scan file.

Loads a .bin scan and its corresponding semantic labels from gt_labels/ directory,
then visualizes the scan with colors based on the semantic labels.
"""

import os
import sys
import numpy as np
import open3d as o3d

# Import dataset_binarize package to set up sys.path for lidar2osm imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lidar2osm.utils.file_io import read_bin_file

# Import semantic labels for color mapping
from create_seq_gt_map_npy import semantic_labels


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


def visualize_bin_labels(root_path, scan_idx=0):
    """
    Load a .bin scan and its semantic labels, then visualize with Open3D.
    
    Args:
        root_path: Path to root directory (should contain lidar_bin/data/ and gt_labels/)
        scan_idx: Index of the scan to visualize (default: 0, first scan)
    """
    # Extract sequence name from root_path
    seq_name = os.path.basename(os.path.normpath(root_path))
    
    # Paths
    bin_data_dir = os.path.join(root_path, "lidar_bin", "data")
    labels_dir = os.path.join(root_path, "gt_labels")
    
    # Check if directories exist
    if not os.path.exists(bin_data_dir):
        print(f"ERROR: Bin data directory not found: {bin_data_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"ERROR: Labels directory not found: {labels_dir}")
        return
    
    # Get all .bin files
    bin_files = sorted([f for f in os.listdir(bin_data_dir) if f.endswith('.bin')])
    
    if len(bin_files) == 0:
        print(f"ERROR: No .bin files found in {bin_data_dir}")
        return
    
    if scan_idx >= len(bin_files):
        print(f"WARNING: scan_idx {scan_idx} >= number of scans {len(bin_files)}, using last scan")
        scan_idx = len(bin_files) - 1
    
    # Get the selected bin file
    bin_file = bin_files[scan_idx]
    bin_path = os.path.join(bin_data_dir, bin_file)
    
    # Get corresponding label file
    label_filename = f"{scan_idx:010d}.bin"
    label_path = os.path.join(labels_dir, label_filename)
    
    print(f"\n{'='*80}")
    print(f"Visualizing semantic labels on bin scan")
    print(f"{'='*80}")
    print(f"Sequence: {seq_name}")
    print(f"Bin scan file: {bin_file} (index {scan_idx})")
    print(f"Label file: {label_filename}")
    print(f"{'='*80}\n")
    
    # Load .bin scan file
    print(f"Loading .bin scan file: {bin_path}")
    try:
        points = read_bin_file(bin_path, dtype=np.float32, shape=(-1, 4))
        points_xyz = points[:, :3]  # Extract xyz coordinates
        intensities = points[:, 3]  # Extract intensity
    except Exception as e:
        print(f"ERROR: Failed to load .bin file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"  Loaded {len(points_xyz)} points from .bin file")
    
    # Load semantic labels
    if not os.path.exists(label_path):
        print(f"ERROR: Label file not found: {label_path}")
        print(f"  Make sure you have run plot_bin_on_pcd.py to generate labels first")
        return
    
    print(f"\nLoading semantic labels: {label_path}")
    try:
        labels = read_bin_file(label_path, dtype=np.int32, shape=(-1))
    except Exception as e:
        print(f"ERROR: Failed to load label file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"  Loaded {len(labels)} labels")
    
    # Validate that labels match points
    if len(labels) != len(points_xyz):
        print(f"ERROR: Label count ({len(labels)}) != point count ({len(points_xyz)})")
        return
    
    # Convert labels to colors
    print(f"\nConverting labels to colors...")
    colors = labels_to_colors(labels)
    
    # Get label statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"  Found {len(unique_labels)} unique labels:")
    label_names = {label.id: label.name for label in semantic_labels}
    for label_id, count in zip(unique_labels, counts):
        label_name = label_names.get(int(label_id), f"unknown({label_id})")
        percentage = (count / len(labels)) * 100.0
        print(f"    Label {label_id:2d} ({label_name:20s}): {count:8d} points ({percentage:5.2f}%)")
    
    # Create Open3D point cloud
    print(f"\nCreating point cloud for visualization...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"\n{'='*80}")
    print(f"Point cloud statistics:")
    print(f"  Total points: {len(points_xyz)}")
    print(f"  X range: [{points_xyz[:, 0].min():.2f}, {points_xyz[:, 0].max():.2f}]")
    print(f"  Y range: [{points_xyz[:, 1].min():.2f}, {points_xyz[:, 1].max():.2f}]")
    print(f"  Z range: [{points_xyz[:, 2].min():.2f}, {points_xyz[:, 2].max():.2f}]")
    print(f"{'='*80}")
    
    # Visualize
    print(f"\nVisualizing point cloud with semantic labels...")
    print(f"  Colors represent semantic classes")
    print(f"\nControls:")
    print(f"  - Mouse: Rotate view")
    print(f"  - Shift + Mouse: Pan view")
    print(f"  - Mouse wheel: Zoom")
    print(f"  - Q or ESC: Quit")
    print(f"{'='*80}\n")
    
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    # Configuration
    # root_path should be the path to a sequence directory (e.g., /path/to/dataset/seq_name)
    # The script will look for:
    #   - root_path/lidar_bin/data/*.bin (bin scan files)
    #   - root_path/gt_labels/<scan_idx:010d>.bin (semantic label files)
    
    root_path = "/media/donceykong/doncey_ssd_02/datasets/MCD/kth_day_06"
    scan_idx = 10  # Index of the scan to visualize (0 = first scan)
    
    visualize_bin_labels(root_path, scan_idx=scan_idx)

