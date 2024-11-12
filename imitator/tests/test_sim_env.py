
#这部分是在线渲染
 #!/usr/bin/env python

# import gymnasium as gym
# from imitator.utils.env.robomimic_env import RoboMimicEnv

# if __name__ == "__main__":
#     env = RoboMimicEnv(
#         env_name="Can",
#         render_mode="human",
#         **{
#             "has_renderer": True,
##         },
#     )

#     obs, info = env.reset()

#     for i in range(1000):
#         action = env.action_space.sample()
#         obs, reward, done, _, info = env.step(action)
#         print(obs.keys())
#         if done:
#             obs, info = env.reset()

#这里是保存视频文件的部分（offscreen渲染）

#!/usr/bin/env python
import gymnasium as gym
from imitator.utils.env.robomimic_env import RoboMimicEnv
import imageio
import numpy as np

if __name__ == "__main__":
    # 创建环境
    env = RoboMimicEnv(
        env_name="Lift",
        render_mode=None,  # 不使用渲染器
        **{
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,  # 启用相机观察
            "camera_names": ["frontview", "agentview"]
            
        },
        # camera_names=('frontview','agentview')
    )
    # Available "camera" names = ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand').

    # 初始化环境
    obs, info = env.reset()

    # 用于存储图像帧的列表
    combined_frames = []

    for i in range(200):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)

        # 获取离屏渲染的图像
        frontview_frame = obs["frontview_image"]
        agentview_frame = obs["agentview_image"]
        
        # 将两个图像水平拼接在一起
        combined_frame = np.hstack((frontview_frame, agentview_frame))
        combined_frames.append(combined_frame)

        if done:
            obs, info = env.reset()

    # 将图像帧保存为视频
    video_path = "output_combined_video.mp4"
    imageio.mimsave(video_path, combined_frames, fps=30)

    print(f"Combined video saved to {video_path}")

    # 关闭环境
    env.close()