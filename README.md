# 0.环境配置

0.1创建虚拟conda环境

    conda create -y -n hxd_lerobot python=3.10

    conda activate lerobot

---

0.2安装相应版本的[pytorch](https://pytorch.org/)

    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

0.3下载并解压`imitator`  

    cd /path/to/imitator
    pip install -e ".[torch]" -i https://mirrors.aliyun.com/pypi/simple/

# 一.RoboMimic_dataset处理

___

1.准备数据集：下载仿真数据集文件。准备数据集low_dim或非low_dim（二者内容量无区别，都不含观测图像，需通过仿真生成）

**2.获取图像数据集：通过`dataset_states_to_obs.py`相关脚本可从原始low_dim数据集中经过模拟器获得带图像信息的数据集(hdf5)。**

仅用于低维观察

    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2

**提取640x480 图像观测**

    python robomimic/scripts/dataset_states_to_obs.py --dataset datasets/can/ph/low_dim_v141.hdf5  --output_name v141_high_image.hdf5 --n 50 --done_mode 2 --camera_names frontview robot0_eye_in_hand --camera_height 640 --camera_width 480 --compress 

提取 84x84 图像和深度观测

    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name depth.hdf5 --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --depth

Using dense rewards

    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense.hdf5 --done_mode 2 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

（节省空间选项）提取 84x84 图像观测并压缩，不提取下一个观测 (not needed for pure imitation learning algos)

    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
    --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
    --compress --exclude-next-obs

所有观察视角选项(设定位于robosuite机器人原始模型文件和环境模型文件)

    {
    frontview
    birdview
    sideview
    agentview
    
    robot0_eye_in_hand
    robot0_robotview
    
    
    }

Only writing done at the end of the trajectory

    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_done_1.hdf5 --done_mode 1 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

查看所有可用命令行参数的描述

    python dataset_states_to_obs.py --help

ps.可从获取到的图像数据集合经过`playback_dataset.py`直接获取数据集内部观察量(xx_image.h5py)或是间接仿真获取(输出mp4文件)

---

## 2.查看数据集信息

（metadata）借助robosuite创建环境` get_dataset_info.py `查看hdf5数据集元信息

    python robomimic/scripts/get_dataset_info.py  --dataset datasets/can/ph/v141_high_image.hdf5   

查看指定数据集内容`peek_h5df.py`

---

# 二.RoboMimic_dataset处理

1.对获取到的带图像信息的`robomimic`数据集进行处理，生成符合lerobot格式的`lerobot_dataset`。

    python imitator/scripts/lerobot_dataset_builder.py -pn lerobot_test

2.对转换后的`lerobot_dataset`进行可视化

    python imitator/scripts/visualize_lerobot_dataset.py --episode-index 0 --root /home/hxd/lerobot2/imitator/lerobot_test/data

在仿真环境下可用的相机视角为`('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand')`，其中`frontview`为机器人前方视角，
`sideview`为机器人侧边视角，`agentview`为机器人Agent视角，`robot0_eye_in_hand`为机器人抓取视角，`robot0_robotview`为机器人自身视角。

2.1远程可视化，基于rerun.io

    python imitator/scripts/visualize_lerobot_dataset.py --episode-index 1 --mode distant --web-port 8080 --ws-port 8081 --root /home/ghb/hxd/imitator/lerobot_test/data

3.基于转换后数据集进行训练

    python imitator/scripts/lerobot_trainer.py hydra.job.name=lerobot_test    device=cuda    env=robomimic    dataset_repo_id=data/lerobot_dataset   policy=diffusion   training.eval_freq=-1   wandb.enable=false root=/home/ghb/hxd/imitator/lerobot_test 

3.1从检查点恢复训练

    python imitator/scripts/lerobot_trainer.py hydra.job.name=lerobot_test    device=cuda    env=robomimic    dataset_repo_id=data/lerobot_dataset   policy=diffusion   training.eval_freq=-1 wandb.enable=false root=/home/ghb/hxd/imitator/lerobot_test hydra.run.dir=/home/ghb/hxd/imitator/outputs/train/2024-11-11/03-53-38_robomimic_diffusion_lerobot_test resume=true

# 三.`imitator`基于转换后数据集进行推理

1.仿真环境无屏幕渲染，保存视频`test_sim_env.py`

2.部署并保存
