#!/usr/bin/env python3
"""Entry point for CENet training.

Loads the top-level training yaml, materializes per-sensor shards from the
data_cfg's `sensor_groups`, and hands fully-resolved DATA + ARCH to Trainer.
"""

import argparse
import copy
import os
import shutil
from shutil import copyfile
import sys

import yaml

# Internal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ce_net import CONFIG_DIR
from ce_net.core.parsers.mcd import split_mcd_sensor_groups
from ce_net.models.trainer_wb import Trainer
from ce_net.utils.sensor import load_sensor, materialize_sensor_groups


def load_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_yaml(base_config, model_config):
    parser = argparse.ArgumentParser("./train.py")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=base_config["dataset_path"],
        help="Dataset root path. Default: from training yaml.",
    )
    parser.add_argument(
        "--arch_cfg", "-ac",
        type=str,
        default=model_config["arch_config"],
        help="Architecture yaml. Default: from training yaml.",
    )
    parser.add_argument(
        "--data_cfg", "-dc",
        type=str,
        default=base_config["data_config"],
        help="Data yaml. Default: from training yaml.",
    )
    parser.add_argument(
        "--img_width", "-iw",
        type=int,
        default=model_config.get("img_width"),
        help="Range-image width. Default: from training yaml.",
    )
    parser.add_argument(
        "--img_height", "-ih",
        type=int,
        default=model_config.get("img_height"),
        help="Range-image height. Default: from training yaml.",
    )
    parser.add_argument(
        "--log", "-l",
        type=str,
        default=base_config["training"]["model_path"],
        help="Log directory root.",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=model_config.get("model_name", ""),
        help="Sub-directory name appended to --log.",
    )
    parser.add_argument(
        "--pretrained", "-p",
        type=str,
        default=model_config.get("pretrained_model_path", None),
        help="Pretrained model dir. Omit for from-scratch training.",
    )

    FLAGS, _ = parser.parse_known_args()
    return FLAGS


def _resolve_dataset(FLAGS, base_config, dataset_name):
    """Open ARCH and DATA yamls; for MCD, materialize sensor_groups+split."""
    if FLAGS.img_width is None or FLAGS.img_height is None:
        raise ValueError(
            "img_width and img_height must be set in model_configs.<model> "
            "(or via --img_width/--img_height)."
        )

    print(f"Opening arch config: {FLAGS.arch_cfg}")
    ARCH = load_yaml(FLAGS.arch_cfg)
    print(f"Opening data config: {FLAGS.data_cfg}")
    DATA = load_yaml(FLAGS.data_cfg)

    if dataset_name == "MCD":
        if "sensor_groups" not in DATA:
            raise ValueError(
                "data_cfg is missing 'sensor_groups' (required for MCD)."
            )
        groups = materialize_sensor_groups(
            DATA["sensor_groups"],
            img_width=FLAGS.img_width,
            img_height=FLAGS.img_height,
        )
        split_ratios = DATA.get("split", [0.8, 0.1, 0.1])
        if not isinstance(split_ratios, list):
            raise ValueError("data_cfg.split must be a [train, valid, test] ratio list.")
        DATA["split"] = split_mcd_sensor_groups(
            FLAGS.dataset, groups, split_ratios, seed=1024
        )
        for split_name in ("train", "valid", "test"):
            n = sum(len(s["scan_files"]) for s in DATA["split"][split_name])
            per_sensor = ", ".join(
                f"{s['sensor_name']}={len(s['scan_files'])}"
                for s in DATA["split"][split_name]
            )
            print(f"MCD {split_name}: {n} scans ({per_sensor})")
        ARCH.setdefault("dataset", {})["sensor"] = None
    else:
        # Non-MCD datasets expect a single sensor block. Pull from
        # training.sensor_config (preferred) or fall back to whatever the
        # arch yaml already had.
        sensor_ref = base_config["training"].get("sensor_config")
        if sensor_ref is not None:
            ARCH.setdefault("dataset", {})["sensor"] = load_sensor(
                sensor_ref, FLAGS.img_width, FLAGS.img_height
            )

    return ARCH, DATA


def _persist_run_artifacts(FLAGS, ARCH, DATA, model_config):
    """Wipe + recreate the run dir and snapshot configs into it."""
    if os.path.isdir(FLAGS.log):
        shutil.rmtree(FLAGS.log)
    os.makedirs(FLAGS.log)

    # Save merged ARCH so inference can reload it as-is.
    with open(os.path.join(FLAGS.log, "arch_cfg.yaml"), "w") as f:
        yaml.safe_dump(ARCH, f, sort_keys=False)
    copyfile(FLAGS.data_cfg, os.path.join(FLAGS.log, "data_cfg.yaml"))

    # Snapshot the resolved model entry (img_width/img_height + paths) so
    # inference picks up the same projection shape that was trained on.
    with open(os.path.join(FLAGS.log, "model_config.yaml"), "w") as f:
        yaml.safe_dump(model_config, f, sort_keys=False)


def load_config_and_train(FLAGS, base_config, dataset_name, model_config):
    if FLAGS.name:
        FLAGS.log = os.path.join(FLAGS.log, FLAGS.name)

    print("\n----------")
    print("Train Configuration:")
    print("  dataset:    ", FLAGS.dataset)
    print("  arch_cfg:   ", FLAGS.arch_cfg)
    print("  data_cfg:   ", FLAGS.data_cfg)
    print("  img_width:  ", FLAGS.img_width)
    print("  img_height: ", FLAGS.img_height)
    print("  log:        ", FLAGS.log)
    print("  pretrained: ", FLAGS.pretrained)
    print("----------\n")

    ARCH, DATA = _resolve_dataset(FLAGS, base_config, dataset_name)

    if FLAGS.pretrained is not None:
        if os.path.isdir(FLAGS.pretrained):
            print(f"Pretrained dir found: {FLAGS.pretrained}")
        else:
            print(f"Pretrained dir not found at {FLAGS.pretrained}; starting from scratch.")
    else:
        print("No pretrained dir provided; starting from scratch.")

    _persist_run_artifacts(FLAGS, ARCH, DATA, model_config)
    return ARCH, DATA


if __name__ == "__main__":
    def seed_torch(seed=1024):
        import random as _random
        import numpy as _np
        import torch as _torch
        _random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)
        _torch.cuda.manual_seed(seed)
        print(f"Seeded RNGs with {seed}")

    seed_torch()

    base_config = load_yaml(CONFIG_DIR / "training_mcd_EDL.yaml")
    dataset_name = base_config["dataset_name"]
    use_progressive = base_config["training"]["use_progressive_growing"]
    model_configs = base_config["training"]["model_configs"]

    if not use_progressive:
        single_name = base_config["training"]["model_name"]
        if single_name not in model_configs:
            raise KeyError(
                f"training.model_name='{single_name}' is not in training.model_configs."
            )
        model_configs = {single_name: model_configs[single_name]}

    for model_cfg_name, current_model_config in model_configs.items():
        # Use a deep copy so the per-model FLAGS don't accumulate state across
        # progressive stages.
        cfg_copy = copy.deepcopy(current_model_config)
        print(f"\n=== Training model: {model_cfg_name} ===")
        if cfg_copy.get("pretrained_model_path"):
            print(f"  pretrained: {cfg_copy['pretrained_model_path']}")
        FLAGS = parse_yaml(base_config, cfg_copy)
        ARCH, DATA = load_config_and_train(FLAGS, base_config, dataset_name, cfg_copy)
        trainer = Trainer(
            ARCH, DATA, dataset_name, FLAGS.dataset, FLAGS.log, FLAGS.pretrained
        )
        trainer.train()
