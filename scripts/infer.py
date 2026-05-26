# infer.py is a script to infer on a dataset using a trained model.
#!/usr/bin/env python3

import argparse
import datetime
import os
import shutil
import subprocess
from shutil import copyfile
import yaml
import sys

# Internal modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ce_net.models.user import User
from ce_net import CONFIG_DIR
from ce_net.core.parsers.mcd import build_mcd_inference_shards
from ce_net.utils.sensor import materialize_sensor_groups

def load_yaml(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    # Default configuration path
    config = load_yaml(CONFIG_DIR / "inference_mcd_EDL.yaml")

    # Setup command line arguments
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
        "--dataset_path",
        "-dataset_path",
        type=str,
        default=config.get('dataset_path', None),
        required=False,
        help="Dataset to train with. Default: (from config.yaml)"
    )
    parser.add_argument(
        "--dataset_name",
        "-d_name",
        type=str,
        default=config.get('dataset_name', None),
        required=False,
        help="Dataset's name. Default: (from config.yaml)"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=config.get('inference', {}).get('model_path', None),
        required=False,
        help="Directory to get the trained model. Default: (from config.yaml)"
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        choices=splits,
        default=config.get('inference', {}).get('split', 'valid'),
        required=False,
        help=f"Split to evaluate on. One of {splits}. Defaults to %(default)s"
    )
    parser.add_argument(
        "--data_config",
        "-data_config",
        type=str,
        default=config.get('data_config', None),
        required=False,
        help="Yaml file with configuration params for inference. Default: None"
    )

    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset_name", FLAGS.dataset_name)
    print("dataset_path", FLAGS.dataset_path)
    print("data_config", FLAGS.data_config)
    print("model", FLAGS.model)
    print("infering", FLAGS.split)
    print("----------\n")

    # open arch config file
    model_name = os.path.basename(FLAGS.model)
    try:
        print("Opening arch config file for %s" % FLAGS.model)
        # ARCH = yaml.safe_load(open(f"{FLAGS.model}", "r"))
        ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    evidential = ARCH.get("train", {}).get("evidential_loss", False)
    print("Evidential inference (vacuity + Dirichlet probs): %s" % ("ON" if evidential else "OFF (arch has evidential_loss: False or missing)"))

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        # DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
        DATA = yaml.safe_load(open(f"{FLAGS.data_config}", "r"))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # Use relative_infer_dir from inference config (e.g. "inferred_labels/cenet_mcd")
    relative_infer_dir = config.get("relative_infer_dir", "inferred_labels/cenet_mcd")
    if FLAGS.dataset_name == "MCD":
        DATA["relative_infer_dir"] = relative_infer_dir
        # Resolve per-sensor inference shards from sensor_groups (each sequence
        # is projected with the LiDAR it was recorded with). img_width/height
        # come from the model run dir's model_config.yaml (persisted at train).
        model_cfg_path = os.path.join(FLAGS.model, "model_config.yaml")
        if not os.path.isfile(model_cfg_path):
            print(f"Missing {model_cfg_path}; re-train so img dims are persisted.")
            quit()
        model_cfg = yaml.safe_load(open(model_cfg_path, "r"))
        if "sensor_groups" not in DATA:
            print("data_cfg is missing 'sensor_groups' (required for MCD).")
            quit()
        groups = materialize_sensor_groups(
            DATA["sensor_groups"],
            img_width=model_cfg["img_width"],
            img_height=model_cfg["img_height"],
        )
        shards = build_mcd_inference_shards(
            FLAGS.dataset_path, groups, sequences=DATA.get("infer_sequences")
        )
        if not shards:
            print("No MCD inference sequences resolved. Check sensor_groups / "
                  "infer_sequences / dataset_path.")
            quit()
        for s in shards:
            print(f"  {s['seq']:35s} sensor={s['sensor_name']:8s} scans={len(s['scan_files'])}")
        DATA["split"] = {"train": [], "valid": [], "test": shards}
        ARCH.setdefault("dataset", {})["sensor"] = None
    elif FLAGS.dataset_name == "CU-MULTI":
        DATA["relative_infer_dir"] = relative_infer_dir
    elif FLAGS.dataset_name == "KITTI-360":
        DATA["relative_infer_dir"] = relative_infer_dir

    # create log folder for each sequence
    try:
        if FLAGS.dataset_name == "CU-MULTI":
            env = DATA["environment"]
            rel_dir = DATA.get("relative_infer_dir", "inferred_labels/cenet_mcd")
            for robot in DATA["test_robots"]:
                inference_dir = os.path.join(FLAGS.dataset_path, env, robot, rel_dir)
                conf_dir = os.path.join(inference_dir, "confidence_scores")
                multiclass_conf_dir = os.path.join(inference_dir, "multiclass_confidence_scores")
                print(f"inference_dir: {inference_dir}")
                if not os.path.isdir(inference_dir):
                    os.makedirs(inference_dir)
                    os.makedirs(conf_dir)
                    os.makedirs(multiclass_conf_dir)
        elif FLAGS.dataset_name == "KITTI-360":
            for seq in DATA["sequences"]:
                inference_dir = os.path.join(FLAGS.dataset_path, seq, relative_infer_dir)
                conf_dir = os.path.join(inference_dir, "confidence_scores")
                multiclass_conf_dir = os.path.join(inference_dir, "multiclass_confidence_scores")
                print(f"inference_dir: {inference_dir}")
                if not os.path.isdir(inference_dir):
                    os.makedirs(inference_dir)
                if not os.path.isdir(conf_dir):
                    os.makedirs(conf_dir)
                if not os.path.isdir(multiclass_conf_dir):
                    os.makedirs(multiclass_conf_dir)
        elif FLAGS.dataset_name == "MCD":
            relative_infer_dir = DATA.get("relative_infer_dir", "inferred_labels/cenet_mcd")
            for s in DATA["split"]["test"]:
                inference_dir = os.path.join(FLAGS.dataset_path, s["seq"], relative_infer_dir)
                conf_dir = os.path.join(inference_dir, "confidence_scores")
                multiclass_conf_dir = os.path.join(inference_dir, "multiclass_confidence_scores")
                print(f"inference_dir: {inference_dir}")
                if not os.path.isdir(inference_dir):
                    os.makedirs(inference_dir)
                if not os.path.isdir(conf_dir):
                    os.makedirs(conf_dir)
                if not os.path.isdir(multiclass_conf_dir):
                    os.makedirs(multiclass_conf_dir)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    # # create user and infer dataset
    user = User(ARCH, DATA, FLAGS.dataset_name, FLAGS.dataset_path, FLAGS.model, FLAGS.split)
    user.infer()
