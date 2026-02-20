#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torch.nn.functional as F

# Internal
from ce_net.models.postproc.KNN import KNN
from ce_net.core.parsers.parser import Parser


class User:
    def __init__(self, ARCH, DATA, dataset_name, dataset_path, modeldir, split):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.modeldir = modeldir
        self.split = split

        if self.dataset_name == "CU-MULTI":
            self.environment = DATA["environment"]
            self.robot = DATA["test_robots"][0]
            split_cfg = self.DATA.get("split") if isinstance(self.DATA.get("split"), dict) else {}
            self.parser = Parser(
                root=self.dataset_path,
                dataset_name = dataset_name,
                train_sequences=split_cfg.get("train", []),
                valid_sequences=split_cfg.get("valid", []),
                test_sequences=split_cfg.get("test", []),
                labels=self.DATA["labels"],
                color_map=self.DATA["color_map"],
                learning_map=self.DATA["learning_map"],
                learning_map_inv=self.DATA["learning_map_inv"],
                sensor=self.ARCH["dataset"]["sensor"],
                max_points=self.ARCH["dataset"]["max_points"],
                batch_size=1,
                workers=self.ARCH["train"]["workers"],
                environment = self.DATA["environment"],
                train_robots = self.DATA["train_robots"],
                val_robots = self.DATA["val_robots"],
                test_robots = self.DATA["test_robots"],
                gt=True,
                shuffle_train=False,
                TRAIN=False,
            )

        elif self.dataset_name == "MCD":
            self.sequences = self.DATA.get("sequences", [self.DATA.get("seq")] if self.DATA.get("seq") else [])
            self.parser = Parser(
                root=self.dataset_path,
                dataset_name=dataset_name,
                train_sequences=[],
                valid_sequences=[],
                test_sequences=self.sequences,
                labels=self.DATA["labels"],
                color_map=self.DATA["color_map"],
                learning_map=self.DATA["learning_map"],
                learning_map_inv=self.DATA["learning_map_inv"],
                sensor=self.ARCH["dataset"]["sensor"],
                max_points=self.ARCH["dataset"]["max_points"],
                batch_size=1,
                workers=self.ARCH["train"]["workers"],
                environment=self.DATA.get("environment"),
                seq=self.sequences[0] if self.sequences else None,
                gt=False,
                shuffle_train=False,
                TRAIN=False,
            )

        print("\n\nDONE PARSING DATA\n\n")

        # concatenate the encoder and the head
        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from ce_net.models.network.HarDNet import HarDNet

                self.model = HarDNet(
                    self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"]
                )

            if self.ARCH["train"]["pipeline"] == "res":
                from ce_net.models.network.ResNet import ResNet_34

                self.model = ResNet_34(
                    self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"]
                )

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.model, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.model, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from ce_net.models.network.Fid import ResNet_34

                self.model = ResNet_34(
                    self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"]
                )

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.model, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.model, nn.SiLU())

        #     print(self.model)
        w_dict = torch.load(
            # modeldir + "/SalsaNext_valid_best",
            modeldir + "/SENet_valid_best",
            # modeldir + "/SENet_train_best",
            map_location=lambda storage, loc: storage,
        )
        self.model.load_state_dict(w_dict["state_dict"], strict=True)

        # Evidential uncertainty: when model was trained with evidential_loss, output logits and compute vacuity (K/S)
        self.use_evidential = self.ARCH["train"].get("evidential_loss", False)
        if self.use_evidential:
            self.model.return_logits = True
            from ce_net.models.losses.evidential_loss import EvidentialLossCal
            unc_args = self.ARCH["train"].get("evidential", {"unc_act": "exp", "unc_type": "log", "kl_strength": 0.5, "ohem": None})
            self.evidential_loss_cal = EvidentialLossCal(unc_args=unc_args, void_index=0, max_epoch=1, writer=None)
            print("Evidential inference: uncertainty (vacuity) will be saved in confidence_scores (higher = more uncertain).")
        else:
            self.evidential_loss_cal = None

        # use knn post processing?
        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(
                self.ARCH["post"]["KNN"]["params"], self.parser.get_n_classes()
            )
        print(self.parser.get_n_classes())

        # GPU?
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

    def infer(self):
        cnn = []
        knn = []

        # If no split is provided, infer on all splits
        if self.split == None:

            self.infer_subset(
                loader=self.parser.get_train_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )

            # do valid set
            self.infer_subset(
                loader=self.parser.get_valid_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
            # do test set
            self.infer_subset(
                loader=self.parser.get_test_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )

        elif self.split == "valid":
            self.infer_subset(
                loader=self.parser.get_valid_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
        elif self.split == "train":
            self.infer_subset(
                loader=self.parser.get_train_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
        else:
            self.infer_subset(
                loader=self.parser.get_test_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
        print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
        print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
        print("Total Frames:{}".format(len(cnn)))
        print("Finished Infering")

        return

    def infer_subset(self, loader, to_orig_fn, cnn, knn):
        # switch to evaluate mode
        self.model.eval()
        total_time = 0
        total_frames = 0
        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            for i, (
                proj_in,
                proj_mask,
                _,
                _,
                path_seq,
                path_name,
                p_x,
                p_y,
                proj_range,
                unproj_range,
                _,
                _,
                _,
                _,
                npoints,
            ) in enumerate(loader):
                # first cut to rela size (batch size one allows it)
                print("\n\nReading recieved values:")
                print(f"npoints: {npoints}")
                print(f"p_x.shape: {p_x.shape}")
                print(f"p_y.shape: {p_y.shape}")
                print("\n\n")

                p_x = p_x[0, :npoints]
                p_y = p_y[0, :npoints]
                proj_range = proj_range[0, :npoints]
                unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]

                print(f"path_seq: {path_seq}")
                print(f"path_name: {path_name}")

                if self.gpu:
                    proj_in = proj_in.cuda()
                    p_x = p_x.cuda()
                    p_y = p_y.cuda()
                    if self.post:
                        proj_range = proj_range.cuda()
                        unproj_range = unproj_range.cuda()
                end = time.time()

                if self.ARCH["train"]["aux_loss"]:
                    with torch.cuda.amp.autocast(enabled=True):
                        [proj_output, x_2, x_3, x_4] = self.model(proj_in)
                else:
                    with torch.cuda.amp.autocast(enabled=True):
                        proj_output = self.model(proj_in)

                print(f"\nproj_output.shape: {proj_output.shape}\n\n")
                if self.use_evidential:
                    # Evidential: proj_output is logits -> alpha -> vacuity (K/S), predictive probs = alpha/S
                    alpha = self.evidential_loss_cal.logit_to_alpha(proj_output)
                    n_classes = alpha.shape[1]
                    S = alpha.sum(dim=1, keepdim=True)
                    vacuity = (n_classes / S).squeeze(1)
                    probs = alpha / S
                    proj_argmax = alpha.argmax(dim=1)[0]
                    conf = vacuity[0]
                    proj_output_for_multiclass = probs
                else:
                    conf, proj_argmax = torch.max(proj_output[0], dim=0)
                    proj_output_for_multiclass = proj_output
                print(f"\nproj_argmax.shape: {proj_argmax.shape}\n\n")

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - end
                print("Network seq", path_seq, "scan", path_name, "in", res, "sec")
                end = time.time()
                cnn.append(res)

                if self.post:
                    # knn postproc
                    unproj_argmax = self.post(
                        proj_range, unproj_range, proj_argmax, p_x, p_y
                    )
                #             # nla postproc
                #             proj_unfold_range, proj_unfold_pre = NN_filter(proj_range, proj_argmax)
                #             proj_unfold_range=proj_unfold_range.cpu().numpy()
                #             proj_unfold_pre=proj_unfold_pre.cpu().numpy()
                #             unproj_range = unproj_range.cpu().numpy()
                #             #  Check this part. Maybe not correct (Low speed caused by for loop)
                #             #  Just simply change from
                #             #  https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI/blob/7f90b45a765b8bba042b25f642cf12d8fccb5bc2/semantic_inference.py#L177-L202
                #             for jj in range(len(p_x)):
                #                 py, px = p_y[jj].cpu().numpy(), p_x[jj].cpu().numpy()
                #                 if unproj_range[jj] == proj_range[py, px]:
                #                     unproj_argmax = proj_argmax[py, px]
                #                 else:
                #                     potential_label = proj_unfold_pre[0, :, py, px]
                #                     potential_range = proj_unfold_range[0, :, py, px]
                #                     min_arg = np.argmin(abs(potential_range - unproj_range[jj]))
                #                     unproj_argmax = potential_label[min_arg]

                else:
                    # put in original pointcloud using indexes
                    unproj_argmax = proj_argmax[p_y, p_x]

                # measure elapsed time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - end
                print("KNN Infered seq", path_seq, "scan", path_name, "in", res, "sec")
                knn.append(res)
                end = time.time()

                # save scan
                # get the first scan in batch and project scan
                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)

                # map to original label
                pred_np = to_orig_fn(pred_np)

                print(f"pred_np.shape: {pred_np.shape}")
                # save scan
                # path = os.path.join (
                #     self.logdir, path_seq, "inferred", path_name
                # )

                # save scan
                if self.dataset_name == "CU-MULTI":
                    relative_infer_dir = self.DATA.get("relative_infer_dir", "inferred_labels/cenet_mcd")
                    label_dir = os.path.join(self.dataset_path, self.environment, self.robot, relative_infer_dir)
                    print(f"Saving scan to {label_dir}/{path_name}")    
                    path = os.path.join(label_dir, path_name)
                elif self.dataset_name == "KITTI-360":
                    # seq = f"2013_05_28_drive_{path_seq:04d}_sync"
                    path = os.path.join(self.dataset_path, "data_3d_semantics", path_seq, "inferred", path_name)
                elif self.dataset_name == "MCD":
                    # path_seq from batch is sequence name (e.g. kth_day_06); save under relative_infer_dir
                    relative_infer_dir = self.DATA.get("relative_infer_dir", "inferred_labels/cenet_mcd")
                    label_dir = os.path.join(self.dataset_path, path_seq, relative_infer_dir)
                    path = os.path.join(label_dir, path_name)
                    print(f"Saving scan to {path}")

                # SAVE PREDICTIONS
                pred_np.tofile(path)


                # # *************** BELOW ADDED BY DONCEY ***************
                # label_dir = os.path.join(self.dataset_path, self.seq, "inferred_labels")

                # *************** Save confidence (or evidential vacuity when use_evidential) ***************
                # Standard: conf = max class probability. EDL: conf = vacuity K/S (higher = more uncertain)
                conf_unproj = conf[p_y, p_x]
                conf_np = conf_unproj.cpu().numpy()
                conf_np = conf_np.reshape((-1)).astype(np.float16)

                conf_dir = os.path.join(label_dir, "confidence_scores")
                conf_path = os.path.join(conf_dir, path_name)
                if not os.path.exists(conf_dir):
                    os.makedirs(conf_dir)

                if self.use_evidential:
                    if not getattr(self, "_evidential_save_logged", False):
                        vmin, vmax = float(conf_np.min()), float(conf_np.max())
                        print(f"[Evidential] Saving vacuity (K/S) to confidence_scores (range [{vmin:.4f}, {vmax:.4f}])")
                        self._evidential_save_logged = True
                print(f"Saving confidence scores to {conf_path}\n")
                conf_np.tofile(conf_path)

                # *************** Save multiclass: Dirichlet predictive probs (alpha/S) when EDL, else softmax probs ***************
                multiclass_conf = proj_output_for_multiclass[0]  # [C, H, W]
                multiclass_conf_unproj = multiclass_conf[:, p_y, p_x].permute(1, 0)  # [Points, Classes]
                multiclass_probs_np = multiclass_conf_unproj.detach().cpu().numpy().astype(np.float16)

                multiclass_conf_dir = os.path.join(label_dir, "multiclass_confidence_scores")
                multiclass_conf_path = os.path.join(multiclass_conf_dir, path_name)
                if not os.path.exists(multiclass_conf_dir):
                    os.makedirs(multiclass_conf_dir)

                if self.use_evidential and not getattr(self, "_evidential_multiclass_logged", False):
                    row_sum = multiclass_probs_np[0].sum()
                    print(f"[Evidential] Saving Dirichlet predictive probs (alpha/S) to multiclass_confidence_scores (row sum ~ {row_sum:.4f})")
                    self._evidential_multiclass_logged = True
                multiclass_probs_np.tofile(multiclass_conf_path)
                print(f"Saved point probabilities to {multiclass_conf_path} with shape {multiclass_probs_np.shape}")