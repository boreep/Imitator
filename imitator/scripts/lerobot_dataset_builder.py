#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import flatten_dict

from imitator.data.hdf5_format_to_lerobot import from_raw_to_lerobot_format
from imitator.utils.file_utils import get_config_from_project_name, get_data_dir

def save_meta_data(
    info: dict[str, Any], stats: dict, episode_data_index: dict[str, list], meta_data_dir: Path
):
    meta_data_dir.mkdir(parents=True, exist_ok=True)

    # save info
    info_path = meta_data_dir / "info.json"
    with open(str(info_path), "w") as f:
        json.dump(info, f, indent=4)

    # save stats
    stats_path = meta_data_dir / "stats.safetensors"
    save_file(flatten_dict(stats), stats_path)

    # save episode_data_index
    episode_data_index = {key: torch.tensor(episode_data_index[key]) for key in episode_data_index}
    ep_data_idx_path = meta_data_dir / "episode_data_index.safetensors"
    save_file(episode_data_index, ep_data_idx_path)

def build_dataset(
    project_name: str,
    video: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
):

    raw_dir = Path(get_data_dir(project_name))
    if not raw_dir.exists():
        raise NotADirectoryError(
            f"{raw_dir} does not exists. Check your paths or run this command to download an existing raw dataset on the hub:"
            f"python lerobot/common/datasets/push_dataset_to_hub/_download_raw.py --raw-dir your/raw/dir --repo-id your/repo/id_raw"
        )
    local_dir = raw_dir / "lerobot_dataset"
    local_dir.mkdir(parents=True, exist_ok=True)
    config = get_config_from_project_name(project_name)

    # Robustify when `local_dir` is str instead of Path
    meta_data_dir = local_dir / "meta_data"
    videos_dir = local_dir / "videos"

    # convert dataset from original raw format to LeRobot format
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(
        config, raw_dir, videos_dir, video
    )

    lerobot_dataset = LeRobotDataset.from_preloaded(
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    stats = compute_stats(lerobot_dataset, batch_size, num_workers)
    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))
    save_meta_data(info, stats, episode_data_index, meta_data_dir)
    return lerobot_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pn",
        "--project-name",
        type=Path,
        required=True,
        help="Name of the project to build the dataset from.",
    )
    parser.add_argument(
        "--video",
        type=int,
        default=1,
        help="Convert each episode of the raw dataset to an mp4 video. This option allows 60 times lower disk space consumption and 25 faster loading time during training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader for computing the dataset statistics.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of processes of Dataloader for computing the dataset statistics.",
    )
    args = parser.parse_args()
    build_dataset(**vars(args))


if __name__ == "__main__":
    main()
