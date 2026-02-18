import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import accimage
except ImportError:
    accimage = None

import numpy as np

# Internal 
from lidar2osm.core.pointcloud.laserscan import LaserScan, SemLaserScan


EXTENSIONS_SCAN = [".bin"]
EXTENSIONS_LABEL = [".bin"]


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


# def my_collate(batch):
#     data = [item[0] for item in batch]
#     project_mask = [item[1] for item in batch]
#     proj_labels = [item[2] for item in batch]
#     data = torch.stack(data, dim=0)
#     project_mask = torch.stack(project_mask, dim=0)
#     proj_labels = torch.stack(proj_labels, dim=0)

#     to_augment = (proj_labels == 12).nonzero()
#     to_augment_unique_12 = torch.unique(to_augment[:, 0])

#     to_augment = (proj_labels == 5).nonzero()
#     to_augment_unique_5 = torch.unique(to_augment[:, 0])

#     to_augment = (proj_labels == 8).nonzero()
#     to_augment_unique_8 = torch.unique(to_augment[:, 0])

#     to_augment_unique = torch.cat(
#         (to_augment_unique_5, to_augment_unique_8, to_augment_unique_12), dim=0
#     )
#     to_augment_unique = torch.unique(to_augment_unique)

#     for k in to_augment_unique:
#         data = torch.cat((data, torch.flip(data[k.item()], [2]).unsqueeze(0)), dim=0)
#         proj_labels = torch.cat(
#             (proj_labels, torch.flip(proj_labels[k.item()], [1]).unsqueeze(0)), dim=0
#         )
#         project_mask = torch.cat(
#             (project_mask, torch.flip(project_mask[k.item()], [1]).unsqueeze(0)), dim=0
#         )

#     return data, project_mask, proj_labels


class KITTI_360(Dataset):
    def __init__(
        self,
        root,  # directory where data is
        sequences,  # sequences for this data (e.g. [1,3,4,6])
        labels,  # label dict: (e.g 10: "car")
        color_map,  # colors dict bgr (e.g 10: [255, 0, 0])
        learning_map,  # classes to learn (0 to N-1 for xentropy)
        learning_map_inv,  # inverse of previous (recover labels)
        sensor,  # sensor to parse scans from
        max_points=150000,  # max number of points present in dataset
        gt=True,
        transform=False,
    ):  # send ground truth?
        # save deats
        self.root = os.path.join(root)
        self.sequences = sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points
        self.gt = gt
        self.transform = transform

        # get number of classes (can't be len(self.learning_map) because there
        # are multiple repeated entries, so the number that matters is how many
        # there are for the xentropy)
        self.nclasses = len(self.learning_map_inv)

        # sanity checks

        # make sure directory exists
        if os.path.isdir(self.root):
            print("Sequences folder exists! Using sequences from %s" % self.root)
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        # make sure labels is a dict
        assert isinstance(self.labels, dict)

        # make sure color_map is a dict
        assert isinstance(self.color_map, dict)

        # make sure learning_map is a dict
        assert isinstance(self.learning_map, dict)

        # make sure sequences is a list
        assert isinstance(self.sequences, list)

        # placeholder for filenames
        self.scan_files = []
        self.label_files = []

        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:
            # to string
            seq = f"2013_05_28_drive_{seq:04d}_sync"

            print(f"parsing seq {seq}")

            # get paths for each
            scan_path = os.path.join(
                self.root,
                "data_3d_raw",
                seq,
                "velodyne_points/data",
            )
            # label_path = os.path.join(
            #     self.root, "data_3d_semantics", seq, "labels_int32"
            # )

            # orig_label_path = os.path.join(self.root, "data_3d_semantics", seq, "labels")
            # orig_label_files = [
            #     os.path.join(dp, f)
            #     for dp, dn, fn in os.walk(os.path.expanduser(orig_label_path))
            #     for f in fn
            #     if is_label(f)
            # ]

            # orig_label_bases = set(
            #     os.path.splitext(os.path.basename(f))[0] for f in orig_label_files
            # )

            osm_label_path = os.path.join(self.root, "data_3d_semantics", seq, "osm_labels")
            osm_label_files = [
                os.path.join(dp, f)
                for dp, dn, fn in os.walk(os.path.expanduser(osm_label_path))
                for f in fn
                if is_label(f)
            ]
            osm_label_bases = set(
                os.path.splitext(os.path.basename(f))[0] for f in osm_label_files
            )

            # Filter scan files to include only those with a corresponding label file
            scan_files = [
                os.path.join(dp, f)
                for dp, dn, fn in os.walk(os.path.expanduser(scan_path))
                for f in fn
                if is_scan(f)
                and os.path.splitext(os.path.basename(f))[0] in osm_label_bases
            ]

            print(f"\n\nlen(scan_files): {len(scan_files)}, len(label_files): {len(osm_label_files)}")
            # # check all scans have labels
            if self.gt:
                assert len(scan_files) == len(osm_label_files)

            # extend list
            self.scan_files.extend(scan_files)
            self.label_files.extend(osm_label_files)

        # sort for correspondance
        self.scan_files.sort()
        self.label_files.sort()
        
        print(f"Using {len(self.scan_files)} scans from sequences {self.sequences}")

    def __getitem__(self, index):
        # get item in tensor shape
        scan_file = self.scan_files[index]
        if self.gt:
            label_file = self.label_files[index]

        # open a semantic laserscan
        DA = False
        flip_sign = False
        rot = False
        drop_points = False
        if self.transform:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    DA = True
                if random.random() > 0.5:
                    flip_sign = True
                if random.random() > 0.5:
                    rot = True
                drop_points = random.uniform(0, 0.5)

        if self.gt:
            scan = SemLaserScan(
                self.color_map,
                project=True,
                H=self.sensor_img_H,
                W=self.sensor_img_W,
                fov_up=self.sensor_fov_up,
                fov_down=self.sensor_fov_down,
                DA=DA,
                flip_sign=flip_sign,
                rot=rot,
                drop_points=drop_points,
            )
        else:
            scan = LaserScan(
                project=True,
                H=self.sensor_img_H,
                W=self.sensor_img_W,
                fov_up=self.sensor_fov_up,
                fov_down=self.sensor_fov_down,
                DA=DA,
                flip_sign=flip_sign,
                rot=rot,
                drop_points=drop_points,
            )

        # open and obtain scan
        scan.open_scan(scan_file)
        if self.gt:
            scan.open_label(label_file)
            # map unused classes to used classes (also for projection)
            scan.sem_label = self.map(scan.sem_label, self.learning_map)
            scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        if self.gt:
            unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
            unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()

        #     proj_normal = torch.from_numpy(scan.normal_image).clone()

        proj_mask = torch.from_numpy(scan.proj_mask)
        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
        else:
            proj_labels = []
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)

        proj = torch.cat(
            [
                proj_range.unsqueeze(0).clone(),
                proj_xyz.clone().permute(2, 0, 1),
                proj_remission.unsqueeze(0).clone(),
            ]
        )

        #     proj = torch.cat([proj_range.unsqueeze(0).clone(),
        #                       proj_xyz.clone().permute(2, 0, 1),
        #                       proj_remission.unsqueeze(0).clone(),
        #                       proj_normal.unsqueeze(0).clone()])

        # proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]

        img_means = scan.get_img_means()
        img_stds = scan.get_img_stds()

        proj = (proj - img_means) / img_stds
        proj = proj * proj_mask.float()

        # get name and sequence
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-4]
        path_name = path_split[-1].replace(".bin", ".bin")

        # return
        return (
            proj,
            proj_mask,
            proj_labels,
            unproj_labels,
            path_seq,
            path_name,
            proj_x,
            proj_y,
            proj_range,
            unproj_range,
            proj_xyz,
            unproj_xyz,
            proj_remission,
            unproj_remissions,
            unproj_n_points,
        )

    def __len__(self):
        return len(self.scan_files)

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]