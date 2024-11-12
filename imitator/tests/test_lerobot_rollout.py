#!/usr/bin/env python3

"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""

from pathlib import Path

import time
import imageio
import numpy
import torch

from imitator.utils.env.robomimic_env import RoboMimicEnv
from moviepy.editor import ImageSequenceClip



env = RoboMimicEnv(
    env_name="Lift",
    render_mode=None,
    **{

        "has_renderer":False,
        #  "has_offscreen_renderer": True,
         "use_camera_obs": True,  # 启用相机观察
         "camera_names": ["agentview", "frontview"] ,  # 选择相机名称
         "camera_heights" :320, 
         "camera_widths":240
    },
)

# Reset the policy and environmens to prepare for rollout
numpy_observation, info = env.reset()
print("Keys in numpy_observation:", numpy_observation.keys())

state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
image_keys = ["agentview_image", "frontview_image"]

# Create a directory to store the video of the evaluation
output_directory = Path("outputs/eval/robomimic_lift")
output_directory.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
pretrained_policy_path = Path("/home/hxd/lerobot2/imitator/outputs/last/pretrained_model")
policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.eval()
policy.to(device)
policy.reset()
print("Policy loaded.")


for i in range(10):
    step = 0
    done = False
    rewards = []
    frames1 = []
    frames2=[]
    numpy_observation, info = env.reset()
    frames1.append(numpy_observation["agentview_image"])
    frames2.append(numpy_observation["frontview_image"])
    
    while step < 200 and not done:
        # Prepare observation for the policy running in Pytorch
        state = torch.concatenate([torch.from_numpy(numpy_observation[key]) for key in state_keys])
        agentview_image = torch.from_numpy(numpy_observation["agentview_image"].copy())
        frontview_image = torch.from_numpy(numpy_observation["frontview_image"].copy())

        # Convert to float32 with image from channel first in [0,255]
        # to channel last in [0,1]
        state = state.to(torch.float32)
        agentview_image = agentview_image.to(torch.float32) / 255
        agentview_image = agentview_image.permute(2, 0, 1)
        frontview_image = frontview_image.to(torch.float32) / 255
        frontview_image = frontview_image.permute(2, 0, 1)

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        agentview_image = agentview_image.to(device, non_blocking=True)
        frontview_image = frontview_image.to(device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
        state = state.unsqueeze(0)
        agentview_image = agentview_image.unsqueeze(0)
        frontview_image = frontview_image.unsqueeze(0)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.images.agentview_image": agentview_image,
            "observation.images.frontview_image": frontview_image,
        }

        # Predict the next action with respect to the current observation
        start_time = time.time()
        with torch.inference_mode():
            action = policy.select_action(observation)
        print(f"Action selection took {time.time() - start_time:.2f} seconds.")

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()

        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        

        # Keep track of all the rewards and frames
        rewards.append(reward)
        
        frames1.append(numpy_observation["agentview_image"])
        frames2.append(numpy_observation["frontview_image"])

        # The rollout is considered done when the success state is reach (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1

    if terminated:
        print("Success!")
    else:
        print("Failure!")

    # Get the speed of environment (i.e. its number of frames per second).
    fps = 20

    # Encode all frames into a mp4 video.
    video_path = output_directory / f"rollout_agentview_{i:03d}.mp4"
    imageio.mimsave(str(video_path), numpy.stack(frames1), fps=fps)
    clip = ImageSequenceClip(frames1, fps=fps)
    clip.write_gif(
        str(output_directory / f"rollout_{i:03d}.gif"),
        fps=fps,
        verbose=False,
    )

    video_path = output_directory / f"rollout_hand_{i:03d}.mp4"
    imageio.mimsave(str(video_path), numpy.stack(frames2), fps=fps)



env.close()

