import numpy as np
import gymnasium as gym
from gymnasium import spaces

from imitator.utils.env import ROBOMIMIC_ENV_METADATA

try:
    import robosuite
    from robosuite.wrappers import Wrapper
except ImportError:
    print("Robosuite not installed. Please install it to use RobosuiteGymWrapper.")


class RoboMimicEnv(Wrapper, gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env_name="Lift", render_mode=None, keys=None, **kwargs):
        assert (
            env_name in ROBOMIMIC_ENV_METADATA
        ), f"Environment {env_name} not found in ROBOMIMIC_ENV_METADATA"
        env_meta = ROBOMIMIC_ENV_METADATA[env_name]
        env_name = env_meta["env_name"]
        env_kwargs = env_meta["env_kwargs"]
        env_kwargs.update(kwargs)
        try:
            env = robosuite.make(
                env_name=env_name,
                **env_kwargs,
            )
        except Exception as e:
            print(f"Error creating environment: {e}")
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join(
            [type(robot.robot_model).__name__ for robot in self.env.robots]
        )
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                # add all state keys that contain "robot*"
                keys += [
                    key
                    for key in self.env.observation_spec().keys()
                    if f"robot{idx}" in key
                ]
        self.keys = keys

        # Gym specific attributes
        # self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        self.modality_dims["object"] = obs["object-state"].shape
        del self.modality_dims["object-state"]  # for robomimic consistency
        spaces_dict = {
            key: (
                gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
                if "image" in key
                else gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
            )
            for key, shape in self.modality_dims.items()
        }
        self.observation_space = spaces.Dict(spaces_dict)
        self.action_space = spaces.Box(*self.env.action_spec)

        # Set render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        ob_dict["object"] = ob_dict["object-state"]
        del ob_dict["object-state"]
        for key in ob_dict.keys():
            if "image" in key:
                ob_dict[key] = ob_dict[key][::-1] # flip image
        info = {}  # TODO
        if self.render_mode == "human":
            self.env.render()
        return ob_dict, info

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        if reward > 0: # success
            terminated = True

        # rename object-state to object
        ob_dict["object"] = ob_dict["object-state"]
        del ob_dict["object-state"]

        # flip image obs
        for key in ob_dict.keys():
            if "image" in key:
                ob_dict[key] = ob_dict[key][::-1]  # flip image

        if self.render_mode == "human":
            self.env.render()
        return ob_dict, reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
