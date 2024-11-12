import os
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 9):  # for python <3.9
    install_requires = [
        "numpy",
        "scipy",
        "opencv-python>=4.9.0",
        "psutil",
        "tqdm",
        "imageio[ffmpeg]>=2.34.0",
        "gymnasium>=0.29.1",
        "cmake>=3.29.0.1",
        "absl-py",
        "Pillow",
        "ml-collections",
        "plotly",
        "matplotlib",
        "h5py>=3.10.0",
        "wandb>=0.16.3",
        "termcolor>=2.4.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.2",
        "gdown>=5.1.0",
        "pymunk>=6.6.0",
        "zarr",
        "numba",
        "imagecodecs",
        "moviepy>=1.0.3",
        "rerun-sdk>=0.15.1",
        "deepdiff>=7.0.1",
    ]
else:  # for python >=3.9
    install_requires = [
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.13.0",
        "opencv-python>=4.9.0",
        "psutil",
        "tqdm",
        "imageio[ffmpeg]>=2.34.0",
        "gymnasium>=0.29.1",
        "cmake>=3.29.0.1",
        "absl-py",
        "robosuite",#仿真
        "minari",#用于d4rl数据集
        "gymnasium-robotics",#仿真
        "Pillow",
        "ml-collections",
        "plotly",
        "matplotlib",
        "h5py>=3.10.0",
        "wandb>=0.16.3",
        "termcolor>=2.4.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.2",
        "gdown>=5.1.0",
        "pymunk>=6.6.0",
        "zarr>=2.17.0",
        "numba>=0.59.0",
        "imagecodecs>=2024.1.1",
        "pyav>=12.0.5",
        "moviepy>=1.0.3",
        "rerun-sdk>=0.15.1",
        "deepdiff>=7.0.1",
    ]


def _post_install():
    PROJECT_ROOT = os.path.expanduser("~/.imitator")
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    CACHE_DIR = os.path.expanduser("~/.cache/imitator")
    os.makedirs(CACHE_DIR, exist_ok=True)


setup(
    name="imitator",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "jax": [
            "jax>=0.4.20",
            "flax>=0.7.5",
            "optax>=0.1.5",
            "orbax>=0.1.9",
            "distrax>=0.1.5",
            "chex>=0.1.85",
            "clu>=0.0.12",
            # "tensorflow==2.15.0",
            # "tensorflow_probability==0.23.0",
            # "tensorflow_hub>=0.14.0",
            # "tensorflow_text>=2.13.0",
            # "tensorflow-datasets>=4.9.2",
            # "tensorflow_graphics==2021.12.3",
            "transformers>=4.34.1",
            "huggingface-hub[hf-transfer]>=0.23.0",
            "einops>=0.8.0",
            "dlimp@git+https://github.com/kvablack/dlimp.git",
        ],
        "torch": [
            "torch>=2.2.1,<3.0.0",
            "torchvision>=0.17.1",
            "einops>=0.8.0",
            "diffusers>=0.27.2",
            "datasets>=2.19.0",
            "huggingface-hub[hf-transfer]>=0.23.0",
            "lerobot@git+https://github.com/huggingface/lerobot.git",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",
            "black",
            "isort",
            "mypy",
            "pre-commit",
            "jupyter",
            "jupyterlab",
            "ipywidgets",
            "nbdev",
        ],
    },
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3.8",
    description="imitator: Imitation learning library for robotics",
    author="Jihoon Oh",
    url="https://github.com/ojh6404/imitator",
    author_email="oh@jsk.imi.i.u-tokyo.ac.jp",
    version="0.0.4",
    entry_points={
        "console_scripts": [
            "imitator=imitator.utils.file_utils:main",
        ]
    },
)

_post_install()
