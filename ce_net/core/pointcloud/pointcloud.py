#!/usr/bin/env python3

# External
import numpy as np
import open3d as o3d

# Internal
from lidar2osm.core.projection import (
    convert_pointcloud_to_latlon,
    post_process_points,
)

# def labels2RGB(label_ids, labels_dict):
#     """
#     Get the color values for a set of semantic labels using their label IDs and the labels dictionary.

#     Args:
#         label_ids (np array of int): The semantic labels.
#         labels_dict (dict): The dictionary containing the semantic label IDs and their corresponding RGB values.

#     Returns:
#         rgb_array (np array of float): The RGB values corresponding to the semantic labels.
#     """
#     # Convert the labels_dict to a lookup array for faster access
#     max_label = max(labels_dict.keys()) if labels_dict else 0
#     lookup = np.zeros((max_label + 1, 3), dtype=float)

#     for label_id, color in labels_dict.items():
#         lookup[label_id] = np.array(color) / 255.0

#     # Ensure label_ids is an integer array
#     label_ids = label_ids.astype(int)

#     # Use the lookup table to get RGB values
#     rgb_array = lookup[label_ids] if len(label_ids) > 0 else np.empty((0, 3), dtype=float)
#     return rgb_array


# def labels2RGB(label_ids, labels_dict):
#     """
#     Get the color values for a set of semantic labels using their label IDs and the labels dictionary.

#     Args:
#         label_ids (np array of int): The semantic labels.
#         labels_dict (dict): The dictionary containing the semantic label IDs and their corresponding RGB values.

#     Returns:
#         rgb_array (np array of float): The RGB values corresponding to the semantic labels.
#     """
#     # Convert the labels_dict to a lookup array for faster access
#     max_label = max(labels_dict.keys()) if labels_dict else 0
#     lookup = np.zeros((max_label + 1, 3), dtype=float)

#     for label_id, color in labels_dict.items():
#         lookup[label_id] = np.array(color) / 255.0

#     # Use the lookup table to get RGB values
#     rgb_array = lookup[label_ids] if len(label_ids) > 0 else np.empty((0, 3), dtype=float)
#     return rgb_array

def labels2RGB(label_ids, labels_dict):
    """
        Get the color values for a set of semantic labels using their label IDs and the labels dictionary.

        Args:
            label_ids (np array of int): The semantic labels.
            labels_dict (dict): The dictionary containing the semantic label IDs and their corresponding RGB values.

        Returns:
            rgb_array (np array of float): The RGB values corresponding to the semantic labels.
    """
    # Prepare the output array
    # rgb_array = np.zeros((label_ids.shape[0], 3), dtype=float)
    rgb_array = np.zeros((len(label_ids), 3), dtype=float)
    for idx, label_id in enumerate(label_ids):
        if label_id in labels_dict:
            color = labels_dict.get(label_id, (0, 0, 0))  # Default color is black
            rgb_array[idx] = np.array(color) / 255.0
    return rgb_array

def labels2RGB_tqdm(label_ids, labels_dict):
    """
        Get the color values for a set of semantic labels using their label IDs and the labels dictionary.

        Args:
            label_ids (np array of int): The semantic labels.
            labels_dict (dict): The dictionary containing the semantic label IDs and their corresponding RGB values.

        Returns:
            rgb_array (np array of float): The RGB values corresponding to the semantic labels.
    """
    # Prepare the output array
    # rgb_array = np.zeros((label_ids.shape[0], 3), dtype=float)
    from tqdm import tqdm

    rgb_array = np.zeros((len(label_ids), 3), dtype=float)

    # Show progress bar
    with tqdm(total=len(label_ids), desc="Converting labels to RGB") as pbar:
        for idx, label_id in enumerate(label_ids):
            if label_id in labels_dict:
                color = labels_dict.get(label_id, (0, 0, 0))  # Default color is black
                rgb_array[idx] = np.array(color) / 255.0
            pbar.update(1)
    return rgb_array

def show_colored_pc(points_semantic, labels_dict):
    points_semantic_xyz = points_semantic[:, :3]
    points_semantic_intensity = points_semantic[:, 3]
    points_semantic_label = points_semantic[:, 4]

    rgb_np = labels2RGB(points_semantic_label, labels_dict)

    monochrome_colors = np.stack(
        [points_semantic_intensity] * 3, axis=-1
    ) * np.array([0, 1, 0])

    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points_semantic_xyz)
    colored_pcd.colors = o3d.utility.Vector3dVector(rgb_np)
    o3d.visualization.draw_geometries([colored_pcd])


def get_transformed_point_cloud(pc, transformation_matrices, frame_number):
    """Transform a point cloud from origin to where the lidar sensor is, given a scan number.

    Args:
        pc (np array of points): The lidar pointcloud to be transformed.
        transformation_matrices (dict of poses): Python dictionary containing the 3x3 pose of lidar sensor for a given scan number.
        frame_number (int): The nth lidar scan (scan number) within the specific sequence.

    Returns:
        transformed_xyz: The pointcloud transformed to where the lidar sensor is.
    """
    # Get the transformation matrix for the current frame
    transformation_matrix = transformation_matrices.get(frame_number)

    # Separate the XYZ coordinates and intensity values
    xyz = pc[:, :3]
    if pc.shape[1] > 3:
        intensity = pc[:, 3].reshape(-1, 1)

    # Convert the XYZ coordinates to homogeneous coordinates
    xyz_homogeneous = np.hstack([xyz, np.ones((xyz.shape[0], 1))])

    # Apply the transformation to each XYZ coordinate
    transformed_xyz = np.dot(xyz_homogeneous, transformation_matrix.T)[:, :3]

    return transformed_xyz


def get_inverse_transformed_point_cloud(
    transformed_pc, transformation_matrices, frame_number
):
    """Transform a point cloud from the lidar sensor's frame back to the original frame, given a scan number.

    Args:
        transformed_pc (np array of points): The transformed lidar pointcloud.
        transformation_matrices (dict of poses): Python dictionary containing the 3x3 pose of lidar sensor for a given scan number.
        frame_number (int): The nth lidar scan (scan number) within the specific sequence.

    Returns:
        original_xyz: The pointcloud transformed back to the original frame.
    """
    # Get the transformation matrix for the current frame
    transformation_matrix = transformation_matrices.get(frame_number)

    # Compute the inverse of the transformation matrix
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Convert the XYZ coordinates to homogeneous coordinates
    xyz_homogeneous = np.hstack([transformed_pc, np.ones((transformed_pc.shape[0], 1))])

    # Apply the inverse transformation to each XYZ coordinate
    original_xyz = np.dot(xyz_homogeneous, inverse_transformation_matrix.T)[:, :3]

    return original_xyz


def transform_points_lat_lon(semantic_3d_points, frame_num, origin_latlon, lidar_poses):
    '''
    Transform points and either keep in lidar frame of reference or transform to lat-long.
    '''
    # Separate points, point intensities, and point labels
    semantic_3d_xyz = np.copy(semantic_3d_points)[:, :3]
    semantic_3d_intensities = np.copy(semantic_3d_points)[:, 3]
    semantic_3d_labels = np.copy(semantic_3d_points)[:, 4]

    # Transform cloud in egocentric frame
    pc_trans_lidar_frame = get_transformed_point_cloud(semantic_3d_xyz, lidar_poses, frame_num)

    # # Convert pointcloud to latlon frame
    # pc_reshaped = np.array([np.eye(4) for _ in range(pc_trans_lidar_frame.shape[0])])
    # pc_reshaped[:, 0:3, 3] = pc_trans_lidar_frame[:, :3]
    # pc_lla = np.asarray(post_process_points(pc_reshaped))
    # pc_lla = np.asarray(convert_pointcloud_to_latlon(pc_lla, origin_latlon=origin_latlon))[:, :3]
    # points_trans = pc_lla
    points_trans = convert_pointcloud_to_latlon(pc_trans_lidar_frame, origin_latlon=origin_latlon)

    # Add point intensities and semantics back into pointcloud
    semantic_3d_points_trans = np.concatenate(
        (
            points_trans,
            semantic_3d_intensities.reshape(-1, 1),
            semantic_3d_labels.reshape(-1, 1),
        ),
        axis=1,
    )
    
    return semantic_3d_points_trans


def get_semantic_points(downsampled_pcd, semantic_ids, labels_dict):
    pc = downsampled_pcd[:, :3]
    intensities_np = downsampled_pcd[:, 3]
    labels_np = downsampled_pcd[:, 4]

    label_mask = np.isin(labels_np, semantic_ids)
    pc = pc[label_mask]
    intensities_np = intensities_np[label_mask]
    labels_np = labels_np[label_mask]

    filtered_points = np.concatenate(
        (pc, intensities_np.reshape(-1, 1), labels_np.reshape(-1, 1)), axis=1
    )

    # filtered_points = pc
    filtered_colors = labels2RGB(labels_np, labels_dict)

    return filtered_points, filtered_colors


def downsample_pointcloud(points_np, labels_np, labels_dict):
    points_xyz_np = points_np[:, :3]
    intensities = points_np[:, 3]

    # Covert labels to RGB stack for downsampling
    labels_3x3 = np.stack([labels_np] * 3, axis=-1)

    # Convert intensities to monochrome RGB colors
    monochrome_colors = np.stack([intensities] * 3, axis=-1)

    # Convert class labels to RGB colors
    rgb_np = labels2RGB(labels_np, labels_dict)

    if len(points_xyz_np) == 0:
        return None
    else:
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points_xyz_np)
        pointcloud.colors = o3d.utility.Vector3dVector(rgb_np)
        downsampled_pointcloud = pointcloud

        pointcloud2 = o3d.geometry.PointCloud()
        pointcloud2.points = o3d.utility.Vector3dVector(points_xyz_np)
        pointcloud2.colors = o3d.utility.Vector3dVector(labels_3x3)
        downsampled_pointcloud2 = pointcloud2
        downsampled_labels = np.asarray(downsampled_pointcloud2.colors)[
            :, 0
        ].reshape(-1, 1)

        pointcloud3 = o3d.geometry.PointCloud()
        pointcloud3.points = o3d.utility.Vector3dVector(points_xyz_np)
        pointcloud3.colors = o3d.utility.Vector3dVector(monochrome_colors)
        downsampled_pointcloud3 = pointcloud3

        # Extract the first channel as the downsampled intensities
        downsampled_intensities = np.asarray(downsampled_pointcloud3.colors)[
            :, 0
        ].reshape(-1, 1)

        return downsampled_pointcloud, downsampled_intensities, downsampled_labels