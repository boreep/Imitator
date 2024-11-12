"""
A collection of utility functions for working with files, such as reading metadata from
demonstration datasets, loading model checkpoints, or downloading dataset files.
"""

import os
import re
from omegaconf import OmegaConf
import json
import shutil
import sys
import subprocess
from imitator.utils import PROJECT_ROOT, PACKAGE_ROOT, CACHE_DIR


def install_octo():
    os.makedirs(CACHE_DIR, exist_ok=True)
    octo_dir = os.path.join(CACHE_DIR, "octo")
    if os.path.exists(octo_dir):
        shutil.rmtree(octo_dir)
    subprocess.run(["git", "clone", "https://github.com/ojh6404/octo.git", octo_dir])
    subprocess.run(["pip", "install", "-e", octo_dir])
    path = os.path.join(CACHE_DIR, "octo")
    sys.path.append(path)
    print("octo installed successfully")


def extract_number(name):
    match = re.search(r"\d+", name)
    if match:
        return int(match.group())
    else:
        return 0


def sort_names_by_number(names):
    sorted_names = sorted(names, key=extract_number)
    return sorted_names


# function that create project dir in the root directory
def create_project_dir(project_name):
    project_dir = os.path.join(PROJECT_ROOT, project_name)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    return project_dir


# function that get project dir in the root directory
def get_project_dir(project_name):
    project_dir = os.path.join(PROJECT_ROOT, project_name)
    return project_dir


def get_models_dir(project_name):
    project_dir = get_project_dir(project_name)
    model_dir = os.path.join(project_dir, "models")
    return model_dir


def get_config_dir(project_name):
    project_dir = get_project_dir(project_name)
    config_dir = os.path.join(project_dir, "config")
    return config_dir


def get_data_dir(project_name):
    project_dir = get_project_dir(project_name)
    data_dir = os.path.join(project_dir, "data")
    return data_dir


def get_config_file(project_name):
    config_dir = get_config_dir(project_name)
    config_file = os.path.join(config_dir, "config.yaml")
    return config_file


def get_config_from_project_name(project_name):
    config_file = get_config_file(project_name)
    config = OmegaConf.load(config_file)
    return config


def pprint_config(config):
    dict_config = OmegaConf.to_container(config, resolve=True)
    print("==================== CONFIG ====================")
    print(json.dumps(dict_config, indent=4))
    print("================================================")


def main():
    import sys

    command = sys.argv[1]
    assert command in ["init", "run", "dataset"], f"Unknown command {command}"

    def init_project(project_name):
        try:
            project_path = create_project_dir(project_name)
            config_dir = get_config_dir(project_name)
            os.makedirs(config_dir, exist_ok=True)

            # create config dir and copy default config
            default_config = os.path.join(PACKAGE_ROOT, "config", "default.yaml")
            config_file = get_config_file(project_name)
            shutil.copy(default_config, config_file)

            # copy default train config
            default_train_config = os.path.join(
                PACKAGE_ROOT, "config", "train_config.py"
            )
            train_config_file = os.path.join(config_dir, "train_config.py")
            shutil.copy(default_train_config, train_config_file)

            # create data dir
            data_dir = get_data_dir(project_name)
            os.makedirs(data_dir, exist_ok=True)
            print(f"Project '{project_name}' created at {project_path}")
        except OSError as e:
            print(f"Error: {e}")
            sys.exit(1)

    if command == "init":
        project_name = sys.argv[2]
        init_project(project_name)
    elif command == "run":
        pass  # TODO: implement run command
    elif command == "dataset":
        args = sys.argv[2:]
        assert args[0] in ["download", "build"], f"Unknown dataset command {args[0]}"
    else:
        raise NotImplementedError(f"Command {command} not implemented")
