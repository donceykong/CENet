#!/usr/bin/env python3
"""Entry point for CENet inference.

Loads the trained model dir's saved configs (arch_cfg, data_cfg,
model_config), materializes sensor_groups for the right img_width /
img_height, and builds per-sequence inference shards. Each sequence is
projected with the sensor it was recorded with.
"""

import argparse
import os
import sys

import yaml

# Internal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ce_net import CONFIG_DIR
from ce_net.core.parsers.mcd import build_mcd_inference_shards
from ce_net.models.user import User
from ce_net.utils.sensor import load_sensor, materialize_sensor_groups


def load_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _make_inference_dirs(dataset_path, dataset_name, DATA, shards):
    """Create the per-sequence/robot inferred-labels directories."""
    rel_dir = DATA.get("relative_infer_dir", "inferred_labels/UNSPEC")
    if dataset_name == "MCD":
        for s in shards:
            base = os.path.join(dataset_path, s["seq"], rel_dir)
            _ensure_dirs(
                base,
                os.path.join(base, "confidence_scores"),
                os.path.join(base, "multiclass_confidence_scores"),
            )
    elif dataset_name == "CU-MULTI":
        env = DATA["environment"]
        for robot in DATA["test_robots"]:
            base = os.path.join(dataset_path, env, robot, rel_dir)
            _ensure_dirs(
                base,
                os.path.join(base, "confidence_scores"),
                os.path.join(base, "multiclass_confidence_scores"),
            )
    elif dataset_name == "KITTI-360":
        seq_list = DATA.get("sequences")
        if not seq_list and isinstance(DATA.get("split"), dict):
            seq_list = [
                f"2013_05_28_drive_{s:04d}_sync"
                for s in DATA["split"].get("test", [])
            ]
        for seq in seq_list or []:
            base = os.path.join(dataset_path, seq, rel_dir)
            _ensure_dirs(
                base,
                os.path.join(base, "confidence_scores"),
                os.path.join(base, "multiclass_confidence_scores"),
            )


def main():
    config = load_yaml(CONFIG_DIR / "inference_mcd.yaml")
    splits = ["train", "valid", "test"]

    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
        "--dataset_path", "-dataset_path",
        type=str,
        default=config.get("dataset_path"),
        help="Dataset root. Default: from inference yaml.",
    )
    parser.add_argument(
        "--dataset_name", "-d_name",
        type=str,
        default=config.get("dataset_name"),
        help="Dataset name. Default: from inference yaml.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=config.get("inference", {}).get("model_path"),
        help="Trained model dir.",
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        choices=splits,
        default=config.get("inference", {}).get("split", "valid"),
        help=f"Split to evaluate on. One of {splits}.",
    )
    parser.add_argument(
        "--data_config", "-data_config",
        type=str,
        default=config.get("data_config"),
        help="data_cfg yaml. Default: from inference yaml.",
    )
    FLAGS, _ = parser.parse_known_args()

    print("----------")
    print("INFERENCE:")
    print("  dataset_name:", FLAGS.dataset_name)
    print("  dataset_path:", FLAGS.dataset_path)
    print("  data_config: ", FLAGS.data_config)
    print("  model:       ", FLAGS.model)
    print("  split:       ", FLAGS.split)
    print("----------\n")

    if not os.path.isdir(FLAGS.model):
        raise FileNotFoundError(f"Model dir not found: {FLAGS.model}")

    # Load configs from the model run dir (saved at train time).
    arch_cfg_path = os.path.join(FLAGS.model, "arch_cfg.yaml")
    model_cfg_path = os.path.join(FLAGS.model, "model_config.yaml")
    print(f"Opening arch config: {arch_cfg_path}")
    ARCH = load_yaml(arch_cfg_path)
    print(f"Opening data config: {FLAGS.data_config}")
    DATA = load_yaml(FLAGS.data_config)

    if not os.path.isfile(model_cfg_path):
        raise FileNotFoundError(
            f"Missing {model_cfg_path}. Re-run training with the updated "
            "scripts/train.py so img_width/img_height are persisted."
        )
    model_cfg = load_yaml(model_cfg_path)
    img_width = model_cfg["img_width"]
    img_height = model_cfg["img_height"]
    print(f"Range image: {img_height} x {img_width}")

    DATA["relative_infer_dir"] = config.get(
        "relative_infer_dir", "inferred_labels/cenet_mcd"
    )

    if FLAGS.dataset_name == "MCD":
        if "sensor_groups" not in DATA:
            raise ValueError("data_cfg is missing 'sensor_groups' (required for MCD).")
        groups = materialize_sensor_groups(
            DATA["sensor_groups"], img_width=img_width, img_height=img_height
        )
        infer_seqs = DATA.get("infer_sequences")
        shards = build_mcd_inference_shards(
            FLAGS.dataset_path, groups, sequences=infer_seqs
        )
        if not shards:
            raise RuntimeError(
                "No MCD inference sequences resolved. Check sensor_groups / "
                "infer_sequences / dataset_path."
            )
        for s in shards:
            print(f"  {s['seq']:35s} sensor={s['sensor_name']:8s} "
                  f"scans={len(s['scan_files'])}")
        DATA["split"] = {"train": [], "valid": [], "test": shards}
        ARCH.setdefault("dataset", {})["sensor"] = None
        _make_inference_dirs(FLAGS.dataset_path, FLAGS.dataset_name, DATA, shards)
    else:
        # Single-sensor datasets: ensure ARCH carries a sensor. If the saved
        # arch_cfg already has one, use it; otherwise pull from data_cfg.
        if not ARCH.get("dataset", {}).get("sensor"):
            sensor_ref = DATA.get("sensor_config")
            if sensor_ref is None:
                raise ValueError(
                    "ARCH has no sensor and data_cfg has no 'sensor_config'."
                )
            ARCH.setdefault("dataset", {})["sensor"] = load_sensor(
                sensor_ref, img_width, img_height
            )
        DATA.setdefault(
            "sequences", [DATA.get("seq")] if DATA.get("seq") else []
        )
        _make_inference_dirs(FLAGS.dataset_path, FLAGS.dataset_name, DATA, shards=[])

    user = User(
        ARCH, DATA, FLAGS.dataset_name, FLAGS.dataset_path, FLAGS.model, FLAGS.split
    )
    user.infer()


if __name__ == "__main__":
    main()
