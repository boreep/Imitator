from typing import Optional, Sequence, Tuple, Dict
from collections import deque
import logging

import numpy as np
import cv2
import jax
import gymnasium as gym

from imitator.utils.env.env_utils import listdict2dictlist, space_stack, stack_and_pad


class ProcessObsWrapper(gym.ObservationWrapper):
    """
    Processes the observation dictionary to match the expected format for the model.
    """

    def __init__(
        self,
        env: gym.Env,
        flatten_keys: Sequence[str] = ("proprio",),
        image_keys: Optional[Dict[str, str]] = None,
    ):
        super().__init__(env)
        assert isinstance(
            self.observation_space, gym.spaces.Dict
        ), "Only Dict observation spaces are supported."
        self.flatten_keys = flatten_keys
        self.image_keys = image_keys

        logging.info(f"Flattening keys: {self.flatten_keys}")
        logging.info(f"Image keys: {self.image_keys}")

        spaces = self.observation_space.spaces

        spaces["state"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(sum([spaces[key].shape[-1] for key in flatten_keys]),),
            dtype=np.float32,
        )
        if image_keys is not None:  # rename image keys
            for k, v in image_keys.items():
                if v is not None:
                    spaces["image_" + k] = gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=spaces[v].shape,
                        dtype=np.uint8,
                    )
        self.observation_space = gym.spaces.Dict(spaces)

    def flatten_state(self, obs):
        return np.concatenate([obs[key] for key in self.flatten_keys], axis=-1)

    def observation(self, obs):
        obs["state"] = self.flatten_state(obs)
        if self.image_keys is not None:
            for k, v in self.image_keys.items():
                if v is not None:
                    obs["image_" + k] = obs[v]
        return obs


class HistoryWrapper(gym.Wrapper):
    """
    Accumulates the observation history into `horizon` size chunks. If the length of the history
    is less than the length of the horizon, we pad the history to the full horizon length.
    A `timestep_pad_mask` key is added to the final observation dictionary that denotes which timesteps
    are padding.
    """

    def __init__(self, env: gym.Env, horizon: int):
        super().__init__(env)
        self.horizon = horizon

        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0

        self.observation_space = space_stack(self.env.observation_space, self.horizon)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.num_obs += 1
        self.history.append(obs)
        assert len(self.history) == self.horizon
        full_obs = stack_and_pad(self.history, self.num_obs)

        return full_obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.num_obs = 1
        self.history.extend([obs] * self.horizon)
        full_obs = stack_and_pad(self.history, self.num_obs)

        return full_obs, info


class RHCWrapper(gym.Wrapper):
    """
    Performs receding horizon control. The policy returns `pred_horizon` actions and
    we execute `exec_horizon` of them.
    """

    def __init__(self, env: gym.Env, exec_horizon: int):
        super().__init__(env)
        self.exec_horizon = exec_horizon

    def step(self, actions):
        if self.exec_horizon == 1 and len(actions.shape) == 1:
            actions = actions[None]
        assert len(actions) >= self.exec_horizon
        rewards = []
        observations = []
        infos = []

        for i in range(self.exec_horizon):
            obs, reward, done, trunc, info = self.env.step(actions[i])
            observations.append(obs)
            rewards.append(reward)
            infos.append(info)

            if done or trunc:
                break

        infos = listdict2dictlist(infos)
        infos["rewards"] = rewards
        infos["observations"] = observations

        return obs, np.sum(rewards), done, trunc, infos


class TemporalEnsembleWrapper(gym.Wrapper):
    """
    Performs temporal ensembling from https://arxiv.org/abs/2304.13705
    At every timestep we execute an exponential weighted average of the last
    `pred_horizon` predictions for that timestep.
    """

    def __init__(self, env: gym.Env, pred_horizon: int, exp_weight: int = 0):
        super().__init__(env)
        self.pred_horizon = pred_horizon
        self.exp_weight = exp_weight

        self.act_history = deque(maxlen=self.pred_horizon)

        self.action_space = space_stack(self.env.action_space, self.pred_horizon)

    def step(self, actions):
        assert len(actions) >= self.pred_horizon

        self.act_history.append(actions[: self.pred_horizon])
        num_actions = len(self.act_history)

        # select the predicted action for the current step from the history of action chunk predictions
        curr_act_preds = np.stack(
            [
                pred_actions[i]
                for (i, pred_actions) in zip(
                    range(num_actions - 1, -1, -1), self.act_history
                )
            ]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.exp_weight * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return self.env.step(action)

    def reset(self, **kwargs):
        self.act_history = deque(maxlen=self.pred_horizon)
        return self.env.reset(**kwargs)


# class ResizeImageWrapper(gym.ObservationWrapper):
#     """
#     Resizes images from a robot environment to the size the model expects.

#     We attempt to match the resizing operations done in the model's data pipeline.
#     First, we resize the image using lanczos interpolation to match the resizing done
#     when converting the raw data into RLDS. Then, we crop and resize the image with
#     bilinear interpolation to match the average of the crop and resize image augmentation
#     performed during training.
#     """

#     def __init__(
#         self,
#         env: gym.Env,
#         resize_size: Optional[Dict[str, Tuple]] = None,
#         augmented_keys: Sequence[str] = ("image_primary",),
#         avg_scale: float = 0.9,
#         avg_ratio: float = 1.0,
#     ):
#         super().__init__(env)
#         assert isinstance(
#             self.observation_space, gym.spaces.Dict
#         ), "Only Dict observation spaces are supported."
#         spaces = self.observation_space.spaces
#         self.resize_size = resize_size
#         self.augmented_keys = augmented_keys
#         if len(self.augmented_keys) > 0:
#             new_height = tf.clip_by_value(tf.sqrt(avg_scale / avg_ratio), 0, 1)
#             new_width = tf.clip_by_value(tf.sqrt(avg_scale * avg_ratio), 0, 1)
#             height_offset = (1 - new_height) / 2
#             width_offset = (1 - new_width) / 2
#             self.bounding_box = tf.stack(
#                 [
#                     height_offset,
#                     width_offset,
#                     height_offset + new_height,
#                     width_offset + new_width,
#                 ],
#             )

#         if resize_size is None:
#             self.keys_to_resize = {}
#         else:
#             self.keys_to_resize = {
#                 f"image_{i}": resize_size[i] for i in resize_size.keys()
#             }
#         logging.info(f"Resizing images: {self.keys_to_resize}")
#         for k, size in self.keys_to_resize.items():
#             spaces[k] = gym.spaces.Box(
#                 low=0,
#                 high=255,
#                 shape=size + (3,),
#                 dtype=np.uint8,
#             )
#         self.observation_space = gym.spaces.Dict(spaces)

#     def observation(self, observation):
#         for k, size in self.keys_to_resize.items():
#             image = tf.image.resize(
#                 observation[k], size=size, method="lanczos3", antialias=True
#             )

#             # if this image key was augmented with random resizes and crops,
#             # we perform the average of the augmentation here
#             if k in self.augmented_keys:
#                 image = tf.image.crop_and_resize(
#                     image[None], self.bounding_box[None], [0], size
#                 )[0]

#             image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()

#             observation[k] = image
#         return observation


class ResizeImageWrapper(gym.ObservationWrapper):
    """
    Resizes images from a robot environment to the size the model expects.

    We attempt to match the resizing operations done in the model's data pipeline.
    First, we resize the image using lanczos interpolation to match the resizing done
    when converting the raw data into RLDS. Then, we crop and resize the image with
    bilinear interpolation to match the average of the crop and resize image augmentation
    performed during training.
    """

    def __init__(
        self,
        env: gym.Env,
        resize_size: Optional[Dict[str, Tuple]] = None,
    ):
        super().__init__(env)
        assert isinstance(
            self.observation_space, gym.spaces.Dict
        ), "Only Dict observation spaces are supported."
        spaces = self.observation_space.spaces
        self.resize_size = resize_size

        if resize_size is None:
            self.keys_to_resize = {}
        else:
            self.keys_to_resize = {
                f"image_{i}": resize_size[i]
                for i in resize_size.keys()
                if resize_size[i] is not None
            }
        logging.info(f"Resizing images: {self.keys_to_resize}")
        for k, size in self.keys_to_resize.items():
            spaces[k] = gym.spaces.Box(
                low=0,
                high=255,
                shape=size + (3,),
                dtype=np.uint8,
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        for k, size in self.keys_to_resize.items():
            image = cv2.resize(observation[k], size, interpolation=cv2.INTER_LANCZOS4)
            image = np.clip(np.round(image), 0, 255).astype(np.uint8)
            # image = tf.image.resize(
            #     observation[k], size=size, method="lanczos3", antialias=True
            # )
            # image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
            observation[k] = image
        return observation


class NormalizeState(gym.ObservationWrapper):
    """
    Un-normalizes the state.
    """

    def __init__(
        self,
        env: gym.Env,
        action_state_metadata: dict,
    ):
        self.action_state_metadata = jax.tree_map(
            lambda x: np.array(x),
            action_state_metadata,
            is_leaf=lambda x: isinstance(x, list),
        )
        super().__init__(env)

    def normalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        return np.where(
            mask,
            (data - metadata["mean"]) / (metadata["std"] + 1e-8),
            data,
        )

    def observation(self, obs):
        if "state" in self.action_state_metadata:
            obs["state"] = self.normalize(
                obs["state"], self.action_state_metadata["state"]
            )
        else:
            assert "state" not in obs, "Cannot normalize state without metadata."
        return obs
