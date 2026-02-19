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
from ce_net.core.pointcloud.laserscan import LaserScan, SemLaserScan


EXTENSIONS_SCAN = [".bin"]
EXTENSIONS_LABEL = [".bin"]


def get_mcd_split_from_sequences_and_ratios(root, sequences, split_ratios, seed=1024):
    """
    Collect all scan/label file pairs from the given sequences under root,
    shuffle with seed, and split into train/valid/test by split_ratios.

    Args:
        root: dataset root (e.g. /path/to/MCD)
        sequences: list of sequence names (e.g. ["kth_day_06", "kth_day_10"])
        split_ratios: [train_ratio, valid_ratio, test_ratio], e.g. [0.8, 0.1, 0.1]

    Returns:
        dict with keys "train", "valid", "test". Each value is
        (list of scan paths, list of label paths).
    """
    train_r, valid_r, test_r = split_ratios
    scan_files = []
    label_files = []
    for seq in sequences:
        scan_path = os.path.join(root, seq, "lidar_bin", "data")
        label_path = os.path.join(root, seq, "gt_labels")
        if not os.path.isdir(label_path) or not os.path.isdir(scan_path):
            continue
        label_list = [
            os.path.join(label_path, f)
            for f in os.listdir(label_path)
            if is_label(f)
        ]
        label_bases = {os.path.splitext(os.path.basename(f))[0] for f in label_list}
        for f in os.listdir(scan_path):
            if not is_scan(f):
                continue
            base = os.path.splitext(f)[0]
            if base not in label_bases:
                continue
            scan_files.append(os.path.join(scan_path, f))
            label_files.append(os.path.join(label_path, f))
    # sort so scan/label stay aligned
    pairs = list(zip(scan_files, label_files))
    pairs.sort(key=lambda x: x[0])
    scan_files = [p[0] for p in pairs]
    label_files = [p[1] for p in pairs]
    n = len(scan_files)
    if n == 0:
        return {"train": ([], []), "valid": ([], []), "test": ([], [])}
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    n_train = int(round(n * train_r))
    n_valid = int(round(n * valid_r))
    n_test = n - n_train - n_valid
    if n_test < 0:
        n_test = 0
        n_valid = n - n_train
    i0, i1 = 0, n_train
    i2 = n_train + n_valid
    train_scans = [scan_files[i] for i in indices[i0:i1]]
    train_labels = [label_files[i] for i in indices[i0:i1]]
    valid_scans = [scan_files[i] for i in indices[i1:i2]]
    valid_labels = [label_files[i] for i in indices[i1:i2]]
    test_scans = [scan_files[i] for i in indices[i2:]]
    test_labels = [label_files[i] for i in indices[i2:]]
    return {
        "train": (train_scans, train_labels),
        "valid": (valid_scans, valid_labels),
        "test": (test_scans, test_labels),
    }


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class MCD(Dataset):
    def __init__(
        self,
        root,               # directory where data is
        labels,             # label dict: (e.g 10: "car")
        color_map,          # colors dict bgr (e.g 10: [255, 0, 0])
        learning_map,       # classes to learn (0 to N-1 for xentropy)
        learning_map_inv,   # inverse of previous (recover labels)
        sensor,             # sensor to parse scans from
        max_points=150000,  # max number of points present in dataset
        seq=None,           # single sequence name (used if scan_files/label_files not provided)
        scan_files=None,    # optional: list of scan paths (overrides seq)
        label_files=None,   # optional: list of label paths, same length as scan_files
        gt=True,
        transform=False,
    ):  # send ground truth?
        # save deats
        self.root = os.path.join(root)
        self.seq = seq
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
        print(f"\nMCD: seq={seq}, scan_files provided={scan_files is not None}")

        # make sure labels is a dict
        print("\nAsserting labels")
        assert isinstance(self.labels, dict)

        # make sure color_map is a dict
        print("\nAsserting color map")
        assert isinstance(self.color_map, dict)

        # make sure learning_map is a dict
        print("\nAsserting learning map")
        assert isinstance(self.learning_map, dict)

        # placeholder for filenames
        self.scan_files = []
        self.label_files = []

        if scan_files is not None and label_files is not None:
            # use provided file lists (e.g. from ratio split)
            assert len(scan_files) == len(label_files)
            self.scan_files = list(scan_files)
            self.label_files = list(label_files)
            print(f"Using {len(self.scan_files)} scans from file list")
        else:
            # discover from single sequence
            if not os.path.isdir(self.root):
                raise ValueError("Sequences folder doesn't exist! Exiting...")
            if seq is None:
                raise ValueError("MCD requires either (scan_files, label_files) or seq")
            print(f"\n\nparsing seq {self.seq}\n\n")

            scan_path = os.path.join(self.root, self.seq, "lidar_bin/data")
            print(f"scan_path: {scan_path}")
            label_path = os.path.join(self.root, self.seq, "gt_labels")
            print(f"label_path: {label_path}")

            label_files = [
                os.path.join(dp, f)
                for dp, dn, fn in os.walk(os.path.expanduser(label_path))
                for f in fn
                if is_label(f)
            ]
            print(f"found {len(label_files)} label files")

            label_bases = set(
                os.path.splitext(os.path.basename(f))[0] for f in label_files
            )

            scan_files = [
                os.path.join(dp, f)
                for dp, dn, fn in os.walk(os.path.expanduser(scan_path))
                for f in fn
                if is_scan(f)
                and os.path.splitext(os.path.basename(f))[0] in label_bases
            ]
            print(f"found {len(scan_files)} scan files")

            if self.gt:
                assert len(scan_files) == len(label_files)

            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)
            self.scan_files.sort()
            self.label_files.sort()
            print(f"Using {len(self.scan_files)} scans from seq {self.seq}")

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
    