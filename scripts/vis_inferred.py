#!/usr/bin/env python3
"""
Script to visualize semantic point clouds from inferred data.
Supports CU-MULTI dataset with semantic segmentation results.
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Internal imports
from lidar2osm.datasets.cu_multi_dataset import labels as sem_kitti_labels
from lidar2osm.core.pointcloud import labels2RGB
from lidar2osm.utils.file_io import read_bin_file
from lidar2osm.core.projection import *


class SemanticPointCloudVisualizer:
    """Visualizer for semantic point clouds with inferred labels."""
    
    def __init__(self, dataset_path: str, dataset_name: str = "CU-MULTI"):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.labels_list = sem_kitti_labels  # This is a list of Label namedtuples
        self.labels_dict = self._convert_labels_to_dict()  # Convert to dict for labels2RGB
        self.current_frame = 0
        self.current_robot = "robot1"
        self.current_environment = "main_campus"
        
        # Color mapping for semantic classes
        self.semantic_colors = self._get_semantic_colors()
        
        # Visualization settings
        self.point_size = 1.0
        self.show_coordinate_frame = True
        self.voxel_size = 0.1  # For downsampling
        
    def _convert_labels_to_dict(self) -> Dict[int, Tuple[int, int, int]]:
        """Convert labels list to dictionary format expected by labels2RGB."""
        labels_dict = {}
        for label in self.labels_list:
            labels_dict[label.id] = label.color
        return labels_dict
        
    def _get_semantic_colors(self) -> Dict[int, Tuple[float, float, float]]:
        """Get RGB colors for semantic classes."""
        colors = {}
        for label in self.labels_list:
            # Convert from 0-255 to 0-1 range
            colors[label.id] = tuple(c/255.0 for c in label.color)
        return colors
    
    def load_point_cloud_data(self, robot: str, environment: str, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load point cloud and semantic labels for a specific frame.
        
        Args:
            robot: Robot name (e.g., "robot1")
            environment: Environment name (e.g., "main_campus")
            frame: Frame number (0-based index)
            
        Returns:
            Tuple of (points, labels) where points is Nx3 and labels is Nx1
        """
        # Construct paths
        velodyne_path = self.dataset_path / environment / robot / "lidar_bin/data"
        labels_path = self.dataset_path / environment / robot / "lidar_labels"
        
        # Find all files and sort them
        velodyne_files = sorted([f for f in velodyne_path.glob("*.bin")])
        label_files = sorted([f for f in labels_path.glob("*.bin")])
        
        if frame >= len(velodyne_files) or frame >= len(label_files):
            raise ValueError(f"Frame {frame} not found. Available frames: 0-{len(velodyne_files)-1}")
        
        # Get the specific files for this frame
        velodyne_file = velodyne_files[frame]
        label_file = label_files[frame]
        
        print(f"Loading velodyne file: {velodyne_file.name}")
        print(f"Loading label file: {label_file.name}")
        
        # Load point cloud (assuming KITTI format: x, y, z, intensity)
        points = read_bin_file(velodyne_file, dtype=np.float32, shape=(-1, 4))
        points_xyz = points[:, :3]  # Extract xyz coordinates
        
        # Load semantic labels
        labels = read_bin_file(label_file, dtype=np.int32)
        
        return points_xyz, labels
    
    def load_inferred_labels(self, robot: str, environment: str, frame: int, 
                           predictions_dir: str = None) -> np.ndarray:
        """
        Load inferred semantic labels from predictions directory.
        
        Args:
            robot: Robot name
            environment: Environment name  
            frame: Frame number (0-based index)
            predictions_dir: Path to predictions directory
            
        Returns:
            Array of inferred labels
        """
        if predictions_dir is None:
            # Look for predictions in the dataset directory
            predictions_dir = self.dataset_path / environment / robot / "predictions"
        else:
            predictions_dir = Path(predictions_dir)
            
        if not predictions_dir.exists():
            raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
        
        # Find prediction file for this frame
        pred_files = sorted([f for f in predictions_dir.glob("*.bin")])
        
        if frame >= len(pred_files):
            raise ValueError(f"Prediction for frame {frame} not found. Available: 0-{len(pred_files)-1}")
        
        # Get the specific prediction file for this frame
        pred_file = pred_files[frame]
        print(f"Loading prediction file: {pred_file.name}")
        
        # Load predictions
        predictions = read_bin_file(pred_file, dtype=np.int32)
        return predictions
    
    def create_semantic_point_cloud(self, points: np.ndarray, labels: np.ndarray, 
                                  use_inferred: bool = False) -> o3d.geometry.PointCloud:
        """
        Create Open3D point cloud with semantic colors.
        
        Args:
            points: Nx3 array of 3D points
            labels: Nx1 array of semantic labels
            use_inferred: Whether to use inferred labels
            
        Returns:
            Open3D point cloud with semantic colors
        """
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Convert labels to RGB colors
        rgb_colors = labels2RGB(labels, self.labels_dict)
        pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
        
        return pcd
    
    def downsample_point_cloud(self, pcd: o3d.geometry.PointCloud, 
                             voxel_size: float = None) -> o3d.geometry.PointCloud:
        """Downsample point cloud using voxel grid."""
        if voxel_size is None:
            voxel_size = self.voxel_size
        return pcd.voxel_down_sample(voxel_size=voxel_size)
    
    def visualize_interactive(self, robot: str, environment: str, start_frame: int = 0,
                            use_inferred: bool = False, predictions_dir: str = None,
                            downsample: bool = True):
        """
        Interactive visualization with keyboard navigation.
        Use left/right arrow keys to navigate through frames.
        
        Args:
            robot: Robot name
            environment: Environment name
            start_frame: Starting frame number
            use_inferred: Whether to use inferred labels
            predictions_dir: Path to predictions directory
            downsample: Whether to downsample point clouds
        """
        print(f"Starting interactive visualization from frame {start_frame}")
        print("Controls:")
        print("  Left Arrow: Previous frame")
        print("  Right Arrow: Next frame")
        print("  ESC or Q: Quit")
        print("  R: Reset view")
        
        # Get total number of frames
        velodyne_path = self.dataset_path / environment / robot / "lidar_bin/data"
        velodyne_files = sorted([f for f in velodyne_path.glob("*.bin")])
        total_frames = len(velodyne_files)
        
        current_frame = start_frame
        
        # Create visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"Interactive Semantic Point Cloud - {robot} - {environment}", 
                        width=1200, height=800)
        
        # Add key callbacks
        def next_frame(vis):
            nonlocal current_frame
            if current_frame < total_frames - 1:
                current_frame += 1
                self._update_frame(vis, robot, environment, current_frame, 
                                 use_inferred, predictions_dir, downsample)
                vis.update_renderer()
            return False
        
        def prev_frame(vis):
            nonlocal current_frame
            if current_frame > 0:
                current_frame -= 1
                self._update_frame(vis, robot, environment, current_frame,
                                 use_inferred, predictions_dir, downsample)
                vis.update_renderer()
            return False
        
        def reset_view(vis):
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            return False
        
        def quit_visualizer(vis):
            return True
        
        # Register key callbacks
        vis.register_key_callback(262, next_frame)  # Right arrow
        vis.register_key_callback(263, prev_frame)  # Left arrow
        vis.register_key_callback(82, reset_view)   # R key
        vis.register_key_callback(27, quit_visualizer)  # ESC
        vis.register_key_callback(81, quit_visualizer)  # Q key
        
        # Load initial frame
        self._update_frame(vis, robot, environment, current_frame,
                         use_inferred, predictions_dir, downsample)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.point_size = self.point_size
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        # Set initial view
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        
        print(f"Frame {current_frame}/{total_frames-1}. Use arrow keys to navigate.")
        vis.run()
        vis.destroy_window()
    
    def _update_frame(self, vis, robot: str, environment: str, frame: int,
                     use_inferred: bool, predictions_dir: str, downsample: bool):
        """Update the visualization with a new frame."""
        try:
            # Clear existing geometries
            vis.clear_geometries()
            
            # Load new frame data
            points, gt_labels = self.load_point_cloud_data(robot, environment, frame)
            
            if use_inferred:
                labels = self.load_inferred_labels(robot, environment, frame, predictions_dir)
            else:
                labels = gt_labels
            
            # Create semantic point cloud
            pcd = self.create_semantic_point_cloud(points, labels, use_inferred)
            
            # Downsample if requested
            if downsample:
                pcd = self.downsample_point_cloud(pcd)
            
            # Add geometries
            vis.add_geometry(pcd)
            if self.show_coordinate_frame:
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
                vis.add_geometry(coordinate_frame)
            
            # Print frame info
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"\nFrame {frame}: {len(pcd.points)} points, {len(unique_labels)} semantic classes")
            
        except Exception as e:
            print(f"Error loading frame {frame}: {e}")
    
    def visualize_single_frame(self, robot: str, environment: str, frame: int,
                             use_inferred: bool = False, predictions_dir: str = None,
                             downsample: bool = True, save_image: bool = False):
        """
        Visualize a single frame with semantic point cloud.
        
        Args:
            robot: Robot name
            environment: Environment name
            frame: Frame number
            use_inferred: Whether to use inferred labels
            predictions_dir: Path to predictions directory
            downsample: Whether to downsample the point cloud
            save_image: Whether to save visualization as image
        """
        print(f"Loading frame {frame} from {robot} in {environment}...")
        
        try:
            # Load point cloud data
            points, gt_labels = self.load_point_cloud_data(robot, environment, frame)
            
            # Use inferred labels if requested
            if use_inferred:
                labels = self.load_inferred_labels(robot, environment, frame, predictions_dir)
                print(f"Using inferred labels for frame {frame}")
            else:
                labels = gt_labels
                print(f"Using ground truth labels for frame {frame}")
            
            # Create semantic point cloud
            pcd = self.create_semantic_point_cloud(points, labels, use_inferred)
            
            # Downsample if requested
            if downsample:
                pcd = self.downsample_point_cloud(pcd)
                print(f"Downsampled to {len(pcd.points)} points")
            
            # Print statistics
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"Semantic classes found: {len(unique_labels)}")
            for label_id, count in zip(unique_labels, counts):
                # Find the label name from the labels list
                label_name = "unknown"
                for label in self.labels_list:
                    if label.id == label_id:
                        label_name = label.name
                        break
                print(f"  {label_name} (ID: {label_id}): {count} points")
            
            # Visualize
            self._visualize_point_cloud(pcd, title=f"Frame {frame} - {robot} - {environment}")
            
            if save_image:
                self._save_visualization(pcd, f"frame_{frame}_{robot}_{environment}")
                
        except Exception as e:
            print(f"Error loading frame {frame}: {e}")
    
    def visualize_multiple_frames(self, robot: str, environment: str, 
                                start_frame: int, end_frame: int, step: int = 1,
                                use_inferred: bool = False, predictions_dir: str = None,
                                downsample: bool = True):
        """
        Visualize multiple frames sequentially.
        
        Args:
            robot: Robot name
            environment: Environment name
            start_frame: Starting frame number
            end_frame: Ending frame number
            step: Step size between frames
            use_inferred: Whether to use inferred labels
            predictions_dir: Path to predictions directory
            downsample: Whether to downsample point clouds
        """
        print(f"Visualizing frames {start_frame} to {end_frame} (step {step})...")
        
        for frame in range(start_frame, end_frame + 1, step):
            try:
                self.visualize_single_frame(robot, environment, frame, 
                                          use_inferred, predictions_dir, downsample)
                
                # Wait for user input to continue
                input("Press Enter to continue to next frame (or Ctrl+C to exit)...")
                
            except KeyboardInterrupt:
                print("Visualization interrupted by user.")
                break
            except Exception as e:
                print(f"Skipping frame {frame} due to error: {e}")
                continue
    
    def visualize_accumulated_point_cloud(self, robot: str, environment: str,
                                       start_frame: int, end_frame: int, step: int = 10,
                                       use_inferred: bool = False, predictions_dir: str = None,
                                       downsample: bool = True):
        """
        Create an accumulated point cloud from multiple frames.
        
        Args:
            robot: Robot name
            environment: Environment name
            start_frame: Starting frame number
            end_frame: Ending frame number
            step: Step size between frames
            use_inferred: Whether to use inferred labels
            predictions_dir: Path to predictions directory
            downsample: Whether to downsample the final point cloud
        """
        print(f"Creating accumulated point cloud from frames {start_frame} to {end_frame}...")
        
        all_points = []
        all_labels = []
        
        for frame in tqdm(range(start_frame, end_frame + 1, step), desc="Loading frames"):
            try:
                points, gt_labels = self.load_point_cloud_data(robot, environment, frame)
                
                if use_inferred:
                    labels = self.load_inferred_labels(robot, environment, frame, predictions_dir)
                else:
                    labels = gt_labels
                
                all_points.append(points)
                all_labels.append(labels)
                
            except Exception as e:
                print(f"Skipping frame {frame} due to error: {e}")
                continue
        
        if not all_points:
            print("No frames loaded successfully.")
            return
        
        # Combine all points and labels
        combined_points = np.vstack(all_points)
        combined_labels = np.hstack(all_labels)
        
        print(f"Combined {len(all_points)} frames into {len(combined_points)} points")
        
        # Create accumulated point cloud
        pcd = self.create_semantic_point_cloud(combined_points, combined_labels, use_inferred)
        
        # Downsample if requested
        if downsample:
            pcd = self.downsample_point_cloud(pcd)
            print(f"Downsampled to {len(pcd.points)} points")
        
        # Visualize
        self._visualize_point_cloud(pcd, title=f"Accumulated Point Cloud - {robot} - {environment}")
    
    def _visualize_point_cloud(self, pcd: o3d.geometry.PointCloud, title: str = "Semantic Point Cloud"):
        """Visualize point cloud with Open3D."""
        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1200, height=800)
        
        # Add geometries
        vis.add_geometry(pcd)
        if self.show_coordinate_frame:
            vis.add_geometry(coordinate_frame)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.point_size = self.point_size
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        # Set view point
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        
        print(f"Visualizing {len(pcd.points)} points. Close the window to continue.")
        vis.run()
        vis.destroy_window()
    
    def _save_visualization(self, pcd: o3d.geometry.PointCloud, filename: str):
        """Save point cloud visualization as image."""
        # Create visualizer for saving
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        
        # Save image
        image_path = f"{filename}.png"
        vis.capture_screen_image(image_path)
        vis.destroy_window()
        
        print(f"Saved visualization to {image_path}")
    
    def print_dataset_info(self):
        """Print information about available data in the dataset."""
        print(f"Dataset: {self.dataset_name}")
        print(f"Dataset path: {self.dataset_path}")
        
        if not self.dataset_path.exists():
            print("Dataset path does not exist!")
            return
        
        # List available environments
        environments = [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
        print(f"Available environments: {environments}")
        
        # List available robots for each environment
        for env in environments:
            env_path = self.dataset_path / env
            robots = [d.name for d in env_path.iterdir() if d.is_dir()]
            print(f"  {env}: {robots}")
            
            # Count frames for each robot
            for robot in robots:
                velodyne_path = env_path / robot / "lidar_bin/data"
                if velodyne_path.exists():
                    frame_count = len(list(velodyne_path.glob("*.bin")))
                    print(f"    {robot}: {frame_count} frames")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Visualize semantic point clouds from inferred data")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", "-d", type=str, required=True,
                       help="Path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, default="CU-MULTI",
                       help="Name of the dataset (default: CU-MULTI)")
    
    # Robot and environment arguments
    parser.add_argument("--robot", "-r", type=str, default="robot1",
                       help="Robot name (default: robot1)")
    parser.add_argument("--environment", "-e", type=str, default="main_campus",
                       help="Environment name (default: main_campus)")
    
    # Frame arguments
    parser.add_argument("--frame", "-f", type=int, default=0,
                       help="Frame number to visualize (default: 0)")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="Starting frame for multi-frame visualization")
    parser.add_argument("--end_frame", type=int, default=10,
                       help="Ending frame for multi-frame visualization")
    parser.add_argument("--step", type=int, default=1,
                       help="Step size between frames (default: 1)")
    
    # Visualization options
    parser.add_argument("--use_inferred", action="store_true",
                       help="Use inferred labels instead of ground truth")
    parser.add_argument("--predictions_dir", type=str,
                       help="Path to predictions directory")
    parser.add_argument("--no_downsample", action="store_true",
                       help="Disable point cloud downsampling")
    parser.add_argument("--voxel_size", type=float, default=0.1,
                       help="Voxel size for downsampling (default: 0.1)")
    parser.add_argument("--point_size", type=float, default=1.0,
                       help="Point size for visualization (default: 1.0)")
    
    # Visualization mode
    parser.add_argument("--mode", type=str, choices=["single", "multiple", "accumulated", "interactive", "info"],
                       default="accumulated", help="Visualization mode (default: single)")
    parser.add_argument("--save_image", action="store_true",
                       help="Save visualization as image")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = SemanticPointCloudVisualizer(args.dataset_path, args.dataset_name)
    visualizer.voxel_size = args.voxel_size
    visualizer.point_size = args.point_size
    visualizer.show_coordinate_frame = True
    
    # Handle different modes
    if args.mode == "info":
        visualizer.print_dataset_info()
        return
    
    elif args.mode == "single":
        visualizer.visualize_single_frame(
            args.robot, args.environment, args.frame,
            args.use_inferred, args.predictions_dir,
            not args.no_downsample, args.save_image
        )
    
    elif args.mode == "multiple":
        visualizer.visualize_multiple_frames(
            args.robot, args.environment,
            args.start_frame, args.end_frame, args.step,
            args.use_inferred, args.predictions_dir,
            not args.no_downsample
        )
    
    elif args.mode == "accumulated":
        visualizer.visualize_accumulated_point_cloud(
            args.robot, args.environment,
            args.start_frame, args.end_frame, args.step,
            args.use_inferred, args.predictions_dir,
            not args.no_downsample
        )
    
    elif args.mode == "interactive":
        visualizer.visualize_interactive(
            args.robot, args.environment, args.frame,
            args.use_inferred, args.predictions_dir,
            not args.no_downsample
        )


if __name__ == "__main__":
    main()
