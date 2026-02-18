import random

import numpy as np
import torch

try:
    import accimage
except ImportError:
    accimage = None

import numpy as np

# Internal 
from lidar2osm.core.parsers.cumulti import CU_MULTI
from lidar2osm.core.parsers.kitti360 import KITTI_360
from lidar2osm.core.parsers.mcd import MCD

class Parser:
    # standard conv, BN, relu
    def __init__(
        self,
        root,                   # directory for data
        dataset_name,           # Identifier for dataset
        train_sequences,        # sequences to train
        valid_sequences,        # sequences to validate.
        test_sequences,         # sequences to test (if none, don't get)
        labels,                 # labels in data
        color_map,              # color for each label
        learning_map,           # mapping for training labels
        learning_map_inv,       # recover labels from xentropy
        sensor,                 # sensor to use
        max_points,             # max points in each scan in entire dataset
        batch_size,             # batch size for train and val
        workers,                # threads to load data
        environment=None,       # environments (not required for KITTI-360)
        seq=None,               # sequence (not required for CU-MULTI)
        train_robots = None,    # robots (not required for KITTI-360)
        val_robots = None,      # robots (not required for KITTI-360)
        test_robots = None,     # robots (not required for KITTI-360)
        gt=True,                # get gt?
        shuffle_train=True,     # shuffle training set?
        TRAIN=False,
    ):
        super(Parser, self).__init__()

        # if I am training, get the dataset
        self.dataset_name = dataset_name
        self.root = root
        self.environment = environment
        self.seq = seq
        self.train_robots = train_robots
        self.val_robots = val_robots
        self.test_robots = test_robots
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.max_points = max_points
        self.batch_size = batch_size
        self.workers = workers
        self.gt = gt
        self.shuffle_train = shuffle_train
        self.TRAIN = TRAIN

        # number of classes that matters is the one for xentropy
        self.nclasses = len(self.learning_map_inv)

        if self.TRAIN:
            if dataset_name == "CU-MULTI":
                print(f"\nUSING CU-MULTI Dataset!!\n")
                self.train_dataset = CU_MULTI(
                    root=self.root,
                    # sequences=self.test_sequences,
                    environment=self.environment,
                    robots=self.train_robots,
                    labels=self.labels,
                    color_map=self.color_map,
                    learning_map=self.learning_map,
                    learning_map_inv=self.learning_map_inv,
                    sensor=self.sensor,
                    max_points=max_points,
                    transform=True,
                    gt=self.gt,
                )
            elif dataset_name == "KITTI-360":
                self.train_dataset = KITTI_360(
                    root=self.root,
                    sequences=self.train_sequences,
                    labels=self.labels,
                    color_map=self.color_map,
                    learning_map=self.learning_map,
                    learning_map_inv=self.learning_map_inv,
                    sensor=self.sensor,
                    max_points=max_points,
                    transform=True,
                    gt=self.gt,
                )
            elif dataset_name == "MCD":
                self.train_dataset = MCD(
                    root=self.root,
                    seq=self.mcd_seq,
                    labels=self.labels,
                    color_map=self.color_map,
                    learning_map=self.learning_map,
                    learning_map_inv=self.learning_map_inv,
                    sensor=self.sensor,
                    max_points=max_points,
                    gt=self.gt,
                )
            #     np.random.seed(0)
            #     dataset_size = len(self.train_dataset)
            #     indices = list(range(dataset_size))
            #     split = int(0.5 * dataset_size)
            #     np.random.shuffle(indices)
            #     train_indices = indices[:split]
            #     train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
            #     print('Subsample:', len(train_indices))

            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            g = torch.Generator()
            g.manual_seed(1024)
            #sampler=train_sampler,

            print(f"Still calling trainloader")
            self.trainloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                num_workers=self.workers,
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=True,
            )

            print(f"len train loader: {len(self.trainloader)}")
            assert len(self.trainloader) > 0
            self.trainiter = iter(self.trainloader)

        if TRAIN and self.val_robots is not None:
            if self.dataset_name == "CU-MULTI":
                self.valid_dataset = CU_MULTI(
                    root=self.root,
                    # sequences=self.valid_sequences,
                    environment=self.environment,
                    robots=self.val_robots,
                    labels=self.labels,
                    color_map=self.color_map,
                    learning_map=self.learning_map,
                    learning_map_inv=self.learning_map_inv,
                    sensor=self.sensor,
                    max_points=max_points,
                    gt=self.gt,
                )
            elif self.dataset_name == "KITTI-360":
                self.valid_dataset = KITTI_360(
                    root=self.root,
                    sequences=self.valid_sequences,
                    labels=self.labels,
                    color_map=self.color_map,
                    learning_map=self.learning_map,
                    learning_map_inv=self.learning_map_inv,
                    sensor=self.sensor,
                    max_points=max_points,
                    gt=self.gt,
                )
            
            self.validloader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                drop_last=True,
            )
            print(f"\nlen valid loader: {len(self.validloader)}")
            assert len(self.validloader) > 0
            self.validiter = iter(self.validloader)

        if not self.TRAIN: 
            if self.dataset_name == "CU-MULTI":
                if self.test_robots is not None:
                    print(f"\n WE ARE USING TEST BOTS: {self.test_robots}")   
                    # if self.test_sequences:
                    self.test_dataset = CU_MULTI(
                        root=self.root,
                        environment=self.environment,
                        robots=self.test_robots,
                        labels=self.labels,
                        color_map=self.color_map,
                        learning_map=self.learning_map,
                        learning_map_inv=self.learning_map_inv,
                        sensor=self.sensor,
                        max_points=max_points,
                        gt=False,
                    )

            elif self.dataset_name == "MCD":
                self.test_dataset = MCD(
                    root=self.root,
                    seq=self.seq,
                    labels=self.labels,
                    color_map=self.color_map,
                    learning_map=self.learning_map,
                    learning_map_inv=self.learning_map_inv,
                    sensor=self.sensor,
                    max_points=max_points,
                    gt=False,
                )

            self.testloader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                drop_last=True,
            )
            # assert len(self.testloader) > 0
            print(f"\nlen test loader: {len(self.testloader)}")
            assert len(self.testloader) > 0
            self.testiter = iter(self.testloader)

    def get_train_batch(self):
        scans = self.trainiter.next()
        return scans

    def get_train_set(self):
        return self.trainloader

    def get_valid_batch(self):
        scans = self.validiter.next()
        return scans

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        scans = self.testiter.next()
        return scans

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)

    def get_n_classes(self):
        return self.nclasses

    def get_original_class_string(self, idx):
        return self.labels[idx]

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def to_original(self, label):
        # put label in original values
        if self.dataset_name == "CU-MULTI":
            return CU_MULTI.map(label, self.learning_map_inv)
        elif self.dataset_name == "KITTI-360":
            return KITTI_360.map(label, self.learning_map_inv)
        elif self.dataset_name == "MCD":
            return MCD.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        # put label in xentropy values
        if self.dataset_name == "CU-MULTI":
            return CU_MULTI.map(label, self.learning_map)
        elif self.dataset_name == "KITTI-360":
            return KITTI_360.map(label, self.learning_map)
        elif self.dataset_name == "MCD":
            return MCD.map(label, self.learning_map)

    def to_color(self, label):
        if self.dataset_name == "CU-MULTI":
            label = CU_MULTI.map(label, self.learning_map_inv)  # put label in original values
            return CU_MULTI.map(label, self.color_map)          # put label in color
        elif self.dataset_name == "KITTI-360":
            label = KITTI_360.map(label, self.learning_map_inv) # put label in original values
            return KITTI_360.map(label, self.color_map)         # put label in color
        elif self.dataset_name == "MCD":
            label = MCD.map(label, self.learning_map_inv)       # put label in original values
            return MCD.map(label, self.color_map)                # put label in color
