#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains utilities to process raw data format of HDF5 files like in: https://github.com/tonyzhaozh/act
"""

import gc
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from omegaconf import DictConfig
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
from imitator.utils.file_utils import sort_names_by_number


def check_format(raw_path: Path, config: DictConfig) -> None:
    with h5py.File(raw_path, "r") as data:
        for demo in tqdm.tqdm(data["data"].keys()):
            ep = data[f"data/{demo}"]

            # check actions and observations exist
            assert "actions" in ep
            assert "obs" in ep

            # check that all obs keys have correct dimensions
            assert ep["actions"].ndim == 2
            for obs_key in config.obs.keys():
                assert obs_key in ep["obs"]
                assert ep["obs"][obs_key].ndim == 2 if "image" not in obs_key else 4

            # check that all obs keys have the same number of frames
            num_frames = ep["actions"].shape[0]
            for obs_key in config.obs.keys():
                assert num_frames == ep["obs"][obs_key].shape[0]


def load_from_raw(
    config: DictConfig, raw_path: Path, videos_dir: Path, fps: int, video: bool
):
    ep_dicts = []
    state_keys = [
        state_key for state_key in config.obs.keys() if "image" not in state_key
    ]
    image_keys = [state_key for state_key in config.obs.keys() if "image" in state_key]
    with h5py.File(raw_path, "r") as data:
        demos = sort_names_by_number(list(data["data"].keys()))
        for demo in tqdm.tqdm(demos):
            ep = data[f"data/{demo}"]
            ep_idx = int(demo[5:])
            num_frames = ep["actions"].shape[0]

            # last step of demonstration is considered done
            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True

            # concat state with state keys and convert to float64
            state = torch.cat(
                [torch.from_numpy(ep["obs"][state_key][:]) for state_key in state_keys],
                dim=1,
            )
            action = torch.from_numpy(ep["actions"][:])  # [num_frames, action_dim]

            ep_dict = {}

            for image_key in image_keys:
                new_image_key = f"observation.images.{image_key}"

                imgs_array = ep[f"obs/{image_key}"][:]  # [num_frames, h, w, c]
                imgs_array = np.array(
                    [
                        np.array(
                            PILImage.fromarray(x).resize(config.obs[image_key].dim[:2])
                        )
                        for x in imgs_array
                    ]
                )

                if video:
                    # save png images in temporary directory
                    tmp_imgs_dir = videos_dir / "tmp_images"
                    save_images_concurrently(imgs_array, tmp_imgs_dir)

                    # encode images to a mp4 video
                    fname = f"{new_image_key}_episode_{ep_idx:06d}.mp4"
                    video_path = videos_dir / fname
                    encode_video_frames(tmp_imgs_dir, video_path, fps)

                    # clean temporary images directory
                    shutil.rmtree(tmp_imgs_dir)

                    # store the reference to the video frame
                    ep_dict[new_image_key] = [
                        {"path": f"videos/{fname}", "timestamp": i / fps}
                        for i in range(num_frames)
                    ]
                else:
                    ep_dict[new_image_key] = [PILImage.fromarray(x) for x in imgs_array]

            ep_dict["observation.state"] = state
            ep_dict["action"] = action
            ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
            ep_dict["next.done"] = done
            # TODO(rcadene): add reward and success by computing them in sim

            assert isinstance(ep_idx, int)
            ep_dicts.append(ep_dict)

        gc.collect()

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1],
        feature=Value(dtype="float32", id=None),
    )
    if "observation.velocity" in data_dict:
        features["observation.velocity"] = Sequence(
            length=data_dict["observation.velocity"].shape[1],
            feature=Value(dtype="float32", id=None),
        )
    if "observation.effort" in data_dict:
        features["observation.effort"] = Sequence(
            length=data_dict["observation.effort"].shape[1],
            feature=Value(dtype="float32", id=None),
        )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    imitator_config: DictConfig,
    raw_dir: Path,
    videos_dir: Path,
    video: bool = True,
):
    # sanity check
    raw_path = raw_dir / "dataset.hdf5"
    check_format(raw_path, imitator_config)

    fps = imitator_config.get("rate", None)
    if fps is None:
        raise ValueError("fps must be provided in the config")

    data_dict = load_from_raw(imitator_config, raw_path, videos_dir, fps, video=video)
    hf_dataset = to_hf_dataset(data_dict, video=video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
