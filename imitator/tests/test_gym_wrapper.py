#!/usr/bin/env python

import h5py
import gymnasium as gym
from imitator.utils.env.gym_wrapper import (
    RoboMimicEnv,
    HistoryWrapper,
    RHCWrapper,
    ResizeImageWrapper,
    TemporalEnsembleWrapper,
    UnnormalizeActionProprio,
)

if __name__ == "__main__":
    window_size = 5
    chunk_size = 10
    # env = gym.make("RoboMimic-v0", env_name="Lift", render_mode="human", **kwargs)
    env = RoboMimicEnv(
        env_name="Lift",
        render_mode="human",
        proprio_mode="JOINT",
        **{
            "has_renderer": True,
        },
    )
    env = ResizeImageWrapper(env, resize_size=(112, 112))
    env = HistoryWrapper(env, horizon=window_size)
    env = RHCWrapper(env, exec_horizon=chunk_size)
    # env = TemporalEnsembleWrapper(env, pred_horizon=chunk_size)

    dataset = h5py.File("/home/oh/.imitator/robomimic_lift/data/dataset.hdf5", "r")
    demo = dataset["data"]["demo_0"]
    actions = demo["actions"]
    demo_length = actions.shape[0]

    obs = env.reset()
    for i in range(0, demo_length, chunk_size):
        # action = env.action_space.sample()
        action_chunk = actions[i : i + chunk_size]
        obs, reward, done, truncated, info = env.step(action_chunk)
        print(obs.keys())
        if done:
            obs = env.reset()
