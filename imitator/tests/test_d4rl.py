#!/usr/bin/env python

import h5py
import gymnasium as gym
import minari

if __name__ == "__main__":
    dataset = minari.load_dataset("kitchen-complete-v1", download=True)
    env = dataset.recover_environment(render_mode="human")

    episode = dataset.sample_episodes(n_episodes=1)[0]
    actions = episode.actions

    obs, info = env.reset()

    for action in actions:
        obs, reward, done, _, info = env.step(action)
        if done:
            obs, info = env.reset()
