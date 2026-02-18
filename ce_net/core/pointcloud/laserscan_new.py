#!/usr/bin/env python3

# External Imports
import math
import random
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt
import open3d as o3d

# Internal Imports
from Lidar2OSM.lidar2osm.core.pointcloud.pointcloud import labels2RGB, get_transformed_point_cloud
from datasets.cu_multi_dataset import labels
from lidar2osm.utils.file_io import read_bin_file

def scale_to_255(a, min, max, dtype=np.float32):
    """Scales an array of values from specified min, max range to 0-255
    Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
# Default values
res = 4.0  # 0.1
side_range = (-40.0, 40.0)  # left-most to right-most
fwd_range = (-40.0, 40.0)
height_range = (-4.0, 4.0)


def point_cloud_2_birdseye(
    points,
    res=res,
    side_range=side_range,  # left-most to right-most
    fwd_range=fwd_range,  # back-most to forward-most
    height_range=height_range,  # bottom-most to upper-most
):
    """Creates a 2D bird's-eye view representation of the point cloud data with intensity and class ID channels.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z.
                    Each point may have up to two extra channels: intensity and semantic labels.
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent a square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the bird's-eye view with 5 channels.
    """
    num_points, num_channels = points.shape

    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    if num_channels > 3:
        intensities = points[:, 3]
    if num_channels > 4:
        labels = points[:, 4]

    # FILTER - To return only indices of points within desired cube
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    if num_channels > 3:
        intensities = intensities[indices]
    if num_channels > 4:
        labels = labels[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    z_points = np.clip(a=z_points, a_min=height_range[0], a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(z_points, min=height_range[0], max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)

    birdseye_view = np.zeros((y_max, x_max, num_channels - 2), dtype=np.float32)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    birdseye_view[y_img, x_img, 0] = pixel_values
    if num_channels > 3:
        birdseye_view[y_img, x_img, 1] = intensities
    if num_channels > 4:
        birdseye_view[y_img, x_img, 2] = labels

    return birdseye_view


def birdseye_to_point_cloud(
    bev_image,
    res=res,
    side_range=side_range,  # left-most to right-most
    fwd_range=fwd_range,  # back-most to forward-most
    height_range=height_range,  # bottom-most to upper-most
):  # bottom-most to upper-most
    """
    Converts a 2D bird's eye view image back to a 3D point cloud.

    Args:
        bev_image: 2D numpy array representing the bird's eye view image.
        res:       (float) Desired resolution in meters.
        side_range: (tuple of two floats) (-left, right) in meters.
        fwd_range: (tuple of two floats) (-behind, front) in meters.
        height_range: (tuple of two floats) (min, max) heights in meters relative to the origin.

    Returns:
        numpy array of shape (N, 3) representing the 3D point cloud.
    """

    # Get dimensions of the image
    y_max, x_max, num_channels = bev_image.shape

    # Extract point coordinates, intensities, and semantics
    bev_image_xyz = bev_image[:, :, 0]
    if num_channels > 1:
        bev_image_intensity = bev_image[:, :, 1].reshape(y_max * x_max, 1)
    if num_channels > 2:
        bev_image_semantic = bev_image[:, :, 2].reshape(y_max * x_max, 1)

    # Generate grid of pixel indices
    x_img, y_img = np.meshgrid(np.arange(x_max), np.arange(y_max))

    # Convert pixel indices to actual positions in meters
    x_points = -(y_img * res + side_range[0])
    y_points = -(x_img * res - fwd_range[1])

    # Get height values from pixel values
    z_points = (
        bev_image_xyz / 255.0 * (height_range[1] - height_range[0]) + height_range[0]
    )

    # Flatten arrays and filter out zero values (no data)
    x_points = x_points.flatten()
    y_points = y_points.flatten()
    z_points = z_points.flatten()

    valid_indices = np.where(bev_image_xyz.flatten() > 0)

    x_points = x_points[valid_indices]
    y_points = y_points[valid_indices]
    z_points = z_points[valid_indices]

    if num_channels > 1:
        points_intensity = bev_image_intensity[valid_indices]
    if num_channels > 2:
        points_semantic = bev_image_semantic[valid_indices]

    # Combine x, y, z into a single point cloud array
    points_xyz = np.vstack((x_points, y_points, z_points)).T

    if num_channels == 1:
        return points_xyz
    if num_channels > 1:
        return points_xyz, points_intensity
    if num_channels > 2:
        return points_xyz, points_intensity, points_semantic


class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""

    EXTENSIONS_SCAN = [".bin"]

    def __init__(
        self,
        project=True,
        H=64,
        W=1024,
        fov_up=2.0,
        fov_down=-24.8,
        DA=False,
        flip_sign=False,
        rot=False,
        drop_points=False,
        bev_res=4.0,
        bev_side_range=(-40.0, 40.0),
        bev_fwd_range=(-40.0, 40.0),
        bev_height_range=(-4.0, 4.0),
    ):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.DA = DA
        self.flip_sign = flip_sign
        self.rot = rot
        self.drop_points = drop_points

        self.bev_res = bev_res
        self.bev_side_range = bev_side_range
        self.bev_fwd_range = bev_fwd_range
        self.bev_height_range = bev_height_range

        self.reset()

    def reset(self):
        """Reset scan members."""
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros(
            (self.proj_H, self.proj_W), dtype=np.int32
        )  # [H,W] mask

        # BEV attributes
        self.bev_image = None

    def size(self):
        """Return the size of the point cloud."""
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """Open raw scan and fill in attributes"""
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError(
                "Filename should be string type, "
                "but was {type}".format(type=str(type(filename)))
            )

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        if self.drop_points is not False:
            self.points_to_drop = np.random.randint(
                0, len(points) - 1, int(len(points) * self.drop_points)
            )
            points = np.delete(points, self.points_to_drop, axis=0)
            remissions = np.delete(remissions, self.points_to_drop)

        self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        """Set scan attributes (instead of opening from file)"""
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points  # get
        if self.flip_sign:
            self.points[:, 1] = -self.points[:, 1]
        if self.DA:
            jitter_x = random.uniform(-5, 5)
            jitter_y = random.uniform(-3, 3)
            jitter_z = random.uniform(-1, 0)
            self.points[:, 0] += jitter_x
            self.points[:, 1] += jitter_y
            self.points[:, 2] += jitter_z
        if self.rot:
            euler_angle = np.random.normal(0, 90, 1)[0]  # 40
            r = np.array(
                R.from_euler("zyx", [[euler_angle, 0, 0]], degrees=True).as_matrix()
            )
            r_t = r.transpose()
            self.points = self.points.dot(r_t)
            self.points = np.squeeze(self.points)
        if remissions is not None:
            self.remissions = remissions  # get remission
            # if self.DA:
            #    self.remissions = self.remissions[::-1].copy()
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()
            self.do_bev_projection()

    def do_bev_projection(self):
        """Project a pointcloud into a BEV projection image."""
        self.bev_image = point_cloud_2_birdseye(
            self.points,
            res=self.bev_res,
            side_range=self.bev_side_range,
            fwd_range=self.bev_fwd_range,
            height_range=self.bev_height_range,
        )

    def do_range_projection(self):
        """Project a pointcloud into a spherical projection image."""
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)

    def fill_spherical(self, range_image):
        # fill in spherical image for calculating normal vector
        height, width = np.shape(range_image)[:2]
        value_mask = np.asarray(1.0 - np.squeeze(range_image) > 0.1).astype(np.uint8)
        dt, lbl = cv2.distanceTransformWithLabels(
            value_mask, cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL
        )

        with_value = np.squeeze(range_image) > 0.1

        depth_list = np.squeeze(range_image)[with_value]

        label_list = np.reshape(lbl, [1, height * width])
        depth_list_all = depth_list[label_list - 1]

        depth_map = np.reshape(depth_list_all, (height, width))

        depth_map = cv2.GaussianBlur(depth_map, (7, 7), 0)
        depth_map = range_image * with_value + depth_map * (1 - with_value)
        return depth_map

    def calculate_normal(self, range_image):

        one_matrix = np.ones((self.proj_H, self.proj_W))
        # img_gaussian =cv2.GaussianBlur(range_image,(3,3),0)
        img_gaussian = range_image
        # prewitt
        kernelx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        self.partial_r_theta = img_prewitty / (np.pi * 2.0 / self.proj_W) / 6
        self.partial_r_phi = (
            img_prewittx
            / (((self.fov_up - self.fov_down) / 180.0 * np.pi) / self.proj_H)
            / 6
        )

        partial_vector = [
            1.0 * one_matrix,
            self.partial_r_theta / (range_image * np.cos(self.phi_channel)),
            self.partial_r_phi / range_image,
        ]
        partial_vector = np.asarray(partial_vector)
        partial_vector = np.transpose(partial_vector, (1, 2, 0))
        partial_vector = np.reshape(partial_vector, [self.proj_H, self.proj_W, 3, 1])
        normal_vector = np.matmul(self.R_theta_phi, partial_vector)
        normal_vector = np.squeeze(normal_vector)
        normal_vector = normal_vector / np.reshape(
            np.max(np.abs(normal_vector), axis=2), (self.proj_H, self.proj_W, 1)
        )
        normal_vector_camera = np.zeros((self.proj_H, self.proj_W, 3))
        normal_vector_camera[:, :, 0] = normal_vector[:, :, 1]
        normal_vector_camera[:, :, 1] = -normal_vector[:, :, 2]
        normal_vector_camera[:, :, 2] = normal_vector[:, :, 0]
        return normal_vector_camera


class SemLaserScan(LaserScan):
    """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""

    EXTENSIONS_LABEL = [".bin"]

    def __init__(
        self,
        sem_color_dict=None,
        project=False,
        H=64,
        W=1024,
        fov_up=5.0,
        fov_down=-25,
        max_classes=65536,
        DA=False,
        flip_sign=False,
        rot=False,
        drop_points=False,
        bev_res=0.1,
        bev_side_range=(-40.0, 40.0),
        bev_fwd_range=(-40.0, 40.0),
        bev_height_range=(-4.0, 4.0),
    ):
        super(SemLaserScan, self).__init__(
            project,
            H,
            W,
            fov_up,
            fov_down,
            DA=DA,
            flip_sign=flip_sign,
            rot=rot,
            drop_points=drop_points,
            bev_res=bev_res,
            bev_side_range=bev_side_range,
            bev_fwd_range=bev_fwd_range,
            bev_height_range=bev_height_range,
        )
        self.reset()

        self.sem_color_dict = sem_color_dict
        # make semantic colors
        if sem_color_dict:
            # if I have a dict, make it
            max_sem_key = 0
            for key, data in sem_color_dict.items():
                if key + 1 > max_sem_key:
                    max_sem_key = key + 1
            self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
            for key, value in sem_color_dict.items():
                self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
        else:
            # otherwise make random
            max_sem_key = max_classes
            self.sem_color_lut = np.random.uniform(
                low=0.0, high=1.0, size=(max_sem_key, 3)
            )
            # force zero to a gray-ish color
            self.sem_color_lut[0] = np.full((3), 0.1)

        # make instance colors
        max_inst_id = 100000
        self.inst_color_lut = np.random.uniform(
            low=0.0, high=1.0, size=(max_inst_id, 3)
        )
        # force zero to a gray-ish color
        self.inst_color_lut[0] = np.full((3), 0.1)

    def reset(self):
        """Reset scan members."""
        super(SemLaserScan, self).reset()

        # semantic labels
        self.sem_label = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: label
        self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # instance labels
        self.inst_label = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: label
        self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # projection color with semantic labels
        self.proj_sem_label = np.zeros(
            (self.proj_H, self.proj_W), dtype=np.int32
        )  # [H,W]  label
        self.proj_sem_color = np.zeros(
            (self.proj_H, self.proj_W, 3), dtype=np.float32
        )  # [H,W,3] color

        # projection color with instance labels
        self.proj_inst_label = np.zeros(
            (self.proj_H, self.proj_W), dtype=np.int32
        )  # [H,W]  label
        self.proj_inst_color = np.zeros(
            (self.proj_H, self.proj_W, 3), dtype=np.float32
        )  # [H,W,3] color

    def open_label(self, filename):
        """Open raw scan and fill in attributes"""
        # check filename is string
        if not isinstance(filename, str):
            raise TypeError(
                "Filename should be string type, "
                "but was {type}".format(type=str(type(filename)))
            )

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
            raise RuntimeError("Filename extension is not valid label file.")

        # if all goes well, open label
        label = np.fromfile(filename, dtype=np.int32)
        label = label.reshape((-1))

        if self.drop_points is not False:
            label = np.delete(label, self.points_to_drop)
        # set it
        self.set_label(label)

    def set_label(self, label):
        """Set points for label not from file but from np"""
        # check label makes sense
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")

        # only fill in attribute if the right size
        if label.shape[0] == self.points.shape[0]:
            self.sem_label = label  # semantic label in lower half
        else:
            print("Points shape: ", self.points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        if self.project:
            self.do_label_projection()
            self.do_bev_label_projection()

    def colorize(self):
        """Colorize pointcloud with the color of each semantic label"""
        self.sem_label_color = self.sem_color_lut[self.sem_label]
        self.sem_label_color = self.sem_label_color.reshape((-1, 3))

    def do_label_projection(self):
        # only map colors to labels that exist
        mask = self.proj_idx >= 0

        # semantics
        self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
        self.proj_sem_color[mask] = self.sem_color_lut[
            self.sem_label[self.proj_idx[mask]]
        ]

    def do_bev_label_projection(self):
        """Project a pointcloud into a BEV projection image with semantic labels."""
        points_with_labels = np.hstack(
            (self.points, self.remissions.reshape(-1, 1), self.sem_label.reshape(-1, 1))
        )
        self.bev_image = point_cloud_2_birdseye(
            points_with_labels,
            res=self.bev_res,
            side_range=self.bev_side_range,
            fwd_range=self.bev_fwd_range,
            height_range=self.bev_height_range,
        )

        self.proj_bev_color = labels2RGB(
            self.bev_image[:, :, 2].flatten(), self.sem_color_dict
        ).reshape(self.bev_image.shape[0], self.bev_image.shape[1], 3)

        # Count the number of points with specific labels
        num_osm_road_points = np.sum(self.sem_label == 46)
        num_osm_building_points = np.sum(self.sem_label == 45)
        num_unlabeled_points = np.sum(self.sem_label == 0)

        # Output the counts for verification
        print(f"Number of OSM road points: {num_osm_road_points}")
        print(f"Number of OSM building points: {num_osm_building_points}")
        print(f"Number of unlabeled points: {num_unlabeled_points}")

        percent_osm_road_points = num_osm_road_points / len(self.sem_label)
        percent_osm_building_points = num_osm_building_points / len(self.sem_label)
        percent_unlabeled = num_unlabeled_points / len(self.sem_label)

        print(f"percent_osm_road_points: {percent_osm_road_points}")
        print(f"percent_osm_building_points: {percent_osm_building_points}")
        print(f"percent_unlabeled: {percent_unlabeled}")

if __name__ == "__main__":
    sequence_dir = "2013_05_28_drive_0000_sync"
    label_dir = f"/media/donceykong/doncey_ssd_01/datasets/KITTI-360-OSM/data_3d_semantics/{sequence_dir}/osm_labels"
    # label_dir = f"/media/donceykong/doncey_ssd_01/kitti360_logs/sequences/{sequence_dir}/predictions"
    # label_dir = "/home/donceykong/Desktop/ARPG/projects/summer_2024/Lidar2OSM_FULL/CENet-OSM/kitti360_logs/sequences/2013_05_28_drive_0002_sync/predictions/"
    labels_dict = {label.id: label.color for label in labels}
    
    velodyne_poses_file = os.path.join("/media/donceykong/doncey_ssd_01/datasets/KITTI-360-OSM/data_poses", sequence_dir, "velodyne_poses.txt")
    velodyne_poses = read_poses(velodyne_poses_file)
    color_map = {
        0: [0, 0, 0],
        1: [0, 0, 255],
        10: [245, 150, 100],
        11: [245, 230, 100],
        13: [250, 80, 100],
        15: [150, 60, 30],
        16: [255, 0, 0],
        18: [180, 30, 80],
        20: [255, 0, 0],
        30: [30, 30, 255],
        31: [200, 40, 255],
        32: [90, 30, 150],
        40: [255, 0, 255],
        44: [255, 150, 255],
        48: [75, 0, 75],
        49: [75, 0, 175],
        50: [0, 200, 255],
        51: [50, 120, 255],
        52: [0, 150, 255],
        60: [170, 255, 150],
        70: [0, 175, 0],
        71: [0, 60, 135],
        72: [80, 240, 150],
        80: [150, 240, 255],
        81: [0, 0, 255],
        99: [255, 255, 50],
        252: [245, 150, 100],
        256: [255, 0, 0],
        253: [200, 40, 255],
        254: [30, 30, 255],
        255: [90, 30, 150],
        257: [250, 80, 100],
        258: [180, 30, 80],
        259: [255, 0, 0],
    }

    scan = SemLaserScan(project=True, sem_color_dict=labels_dict)

    # Initialize a list to store filenames
    file_names = []

    # Iterate over all files in the directory
    for filename in os.listdir(label_dir):
        if filename.endswith(".bin"):
            file_name_without_extension = filename.split(".bin")[0]
            file_names.append(file_name_without_extension)

    # Order list
    file_names.sort()

    pc_list = []
    label_list = []
    for idx, file in enumerate(file_names):
        if idx % 5 == 0:  # Skip every 20th file
            print(f"file: {file}")
            label_file = os.path.join(label_dir, f"{file}.bin")
            scan_file = f"/media/donceykong/doncey_ssd_01/datasets/KITTI-360-OSM/data_3d_raw/{sequence_dir}/velodyne_points/data/{file}.bin"
            
            frame_num = int(file)
            points_np = read_bin_file(scan_file, dtype=np.float32, shape=(-1, 4))
            points_xyz_np = points_np[:, :3]
            intensities = points_np[:, 3]
            labels_np = read_bin_file(label_file, dtype=np.int32, shape=(-1))

            # Create mask for labels 45 and 46
            mask = np.isin(labels_np, [45, 46])

            # Filter points and labels based on the mask
            points_xyz_np = points_xyz_np[mask]
            intensities = intensities[mask]
            labels_np = labels_np[mask]

            pc = get_transformed_point_cloud(points_xyz_np, velodyne_poses, frame_num)
            
            pc[:, 2] = 0
            pc_list.extend(pc)
            label_list.extend(labels_np)
            # open and obtain scan
            scan.open_scan(scan_file)
            scan.open_label(label_file)

            plt.imshow(scan.proj_bev_color)
            plt.show()

            plt.imshow(scan.proj_sem_color)
            plt.show()

    pc_np = np.array(pc_list)
    labels_np = np.array(label_list)

    # Print shapes of points and shape of labels
    print(f"Shape of point cloud: {pc_np.shape}")
    print(f"Shape of labels: {labels_np.shape}")

    # Convert class labels to RGB colors
    print("Converting class labels to RGB colors...")
    rgb_np = labels2RGB(labels_np, labels_dict)
    print("Done.")

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pc_np)
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_np)
    o3d.visualization.draw_geometries([pointcloud])