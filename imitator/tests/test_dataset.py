#!/usr/bin/env python

import os
import numpy as np
from PIL import Image

from imitator.data.dataset import make_single_dataset
from imitator.data.utils.data_utils import NormalizationType

TASK = "image_conditioned"
WINDOW_SIZE = 5
FUTURE_ACTION_WINDOW_SIZE = 10
BATCH_SIZE = 128
DATASET_NAME = "imitator_dataset"
DATA_DIR = os.path.expanduser("~/tensorflow_datasets")
STATE_OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
ACTION_NORMALIZATION_MASK = [True, True, True, True, True, True, True]

if __name__ == "__main__":
    if TASK == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif TASK == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif TASK == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=WINDOW_SIZE,
        action_horizon=FUTURE_ACTION_WINDOW_SIZE,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
    )
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name=DATASET_NAME,
            data_dir=DATA_DIR,
            image_obs_keys={
                "primary": "agentview_image",
                "wrist": "robot0_eye_in_hand_image",
            },
            state_obs_keys=STATE_OBS_KEYS,
            language_key="language_instruction",
            action_state_normalization_type=NormalizationType.BOUNDS,
            action_normalization_mask=ACTION_NORMALIZATION_MASK,
        ),
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=dict(
            resize_size={"primary": (112, 112), "wrist": (112, 112)},
            image_augment_kwargs={
                "primary": workspace_augment_kwargs,
                "wrist": wrist_augment_kwargs,
            },
        ),
        train=True,
    )

    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(10000)
        .batch(BATCH_SIZE)
        .iterator()  # can reduce this if RAM consumption too high
    )

    # get a batch of data
    batch = next(train_data_iter)
    print("Batch keys: ", batch.keys())
    print("Batch observation keys: ", batch["observation"].keys())
    print("State shapes: ", batch["observation"]["state"].shape)
    print("Action shapes: ", batch["action"].shape)

    action_mean = dataset.dataset_statistics["action"]["mean"]
    action_std = dataset.dataset_statistics["action"]["std"]
    obs_mean = dataset.dataset_statistics["state"]["mean"]
    obs_std = dataset.dataset_statistics["state"]["std"]

    first_obs = batch["observation"]["state"][0]
    print("First observation: ", first_obs)
    first_action = batch["action"][0]
    print("First action: ", first_action)

    primary_images = batch["observation"]["image_primary"][0]  # remove batch dimension
    wrist_images = batch["observation"]["image_wrist"][0]  # remove batch dimension

    # visualize images
    primary_images = Image.fromarray(np.concatenate(primary_images, axis=1))
    wrist_images = Image.fromarray(np.concatenate(wrist_images, axis=1))
    primary_images.show()
    wrist_images.show()
