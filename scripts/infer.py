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
from lidar2osm.models.user import User
from lidar2osm import CONFIG_DIR

def load_yaml(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    # Default configuration path
    config = load_yaml(CONFIG_DIR / "inference_MCD.yaml")

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

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        # DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
        DATA = yaml.safe_load(open(f"{FLAGS.data_config}", "r"))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # create log folder for each sequence
    try:
        if FLAGS.dataset_name == "CU-MULTI":
            env = DATA["environment"]
            for robot in DATA["test_robots"]:
                inference_dir = os.path.join(FLAGS.dataset_path, env, robot, f"{robot}_{env}_lidar_labels_confidence")
                conf_dir = os.path.join(inference_dir, "confidence_scores")
                multiclass_conf_dir = os.path.join(inference_dir, "multiclass_confidence_scores")
                print(f"inference_dir: {inference_dir}")
                if not os.path.isdir(inference_dir):
                    os.makedirs(inference_dir)
                    os.makedirs(conf_dir)
                    os.makedirs(multiclass_conf_dir)
        elif FLAGS.dataset_name == "KITTI-360":
            for seq in DATA["split"]["test"]:
                seq = f"2013_05_28_drive_{seq:04d}_sync"
                inference_dir = os.path.join(FLAGS.dataset_path, "data_3d_semantics", seq, "inferred")
                print(f"inference_dir: {inference_dir}")
                if not os.path.isdir(inference_dir):
                    os.makedirs(inference_dir)
        elif FLAGS.dataset_name == "MCD":
            seq = DATA["seq"]
            env_dir = os.path.join(FLAGS.dataset_path, f"{seq}")
            inference_dir = os.path.join(env_dir, "inferred_labels_SKITTI_CENET")
            print(f"inference_dir: {inference_dir}")
            if not os.path.isdir(inference_dir):
                os.makedirs(inference_dir)
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
