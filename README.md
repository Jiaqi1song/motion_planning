# CMU 18-744 Project: Motion Planning in Structured Area with Carla

### Initialization
Create the environment and install the requirements:

```bash
conda create -n carla python=3.7.16
conda activate carla
pip install -r requirements.txt
```

Manully install the carla `0.9.13` package with:
```bash
pip install PythonAPI/carla/dist/carla-0.9.13-cp37-cp37m-win_amd64.whl
```

Start Carla server `0.9.13` in Windows11 first with:

```bash
./CarlaUE4.exe -fps=15
```

### Configuration file
The configuration is located in `config.py`. It contains the following parameters:
- `algorithm`: The RL algorithm to use. All algorithms from Stable Baselines 3 are supported.
- `algoritm_params`: The parameters of the algorithm. See the Stable Baselines 3 documentation for more information.
- `state`: The state to use as a list of atributes. For example, `steer, throttle, speed, angle_next_waypoint, maneuver, waypoints, rgb_camera, seg_camera, end_wp_vector, end_wp_fixed, distance_goal`
 See the `carla_env/state_commons.py` file for more information.
- `vae_model`: The VAE model to use. This repo contains two pretrained models: `vae_64` and `vae_64_augmentation`. If `None`, no VAE is used.
- `action_smoothing`: Whether to use action smoothing or not.
- `reward_fn`: The reward function to use. See the `carla_env/reward_functions.py` file for more information.
- `reward_params`: The parameters of the reward function.
- `obs_res`: The resolution of the observation. It's recommended to use `(160, 80)`
- `seed`: The random seed to use.
- `wrappers`: A list of wrappers to use. Currently there are two implemented: `HistoryWrapperObsDict` and `FrameSkip`. See the `carla_env/wrappers.py` file for more information.

### Train VAE models
We need to train a VAE model first with:

```bash
python vae/train_vae.py --epochs 1000
```

### Training
Train the model with:
```bash
python train.py --config 1 --total_timesteps 100
```

The training results will be saved in the `tensorboard` folder. You can open it with:
```bash
tensorboard --logdir tensorboard
```

### Evaluation
Evaluate the model with:

```bash
python eval.py --config 1 --model "./tensorboard/PPO_1741216403_id1/model_1020_steps.zip"
```

The evaluation routes can be changed inside `carla_env/envs/carla_env.py` in the `eval_routes` variable. Choose two points in the map and add them to the list.


### Train and evaluate multiple models
To train and evaluate multiple models run the `run_experiments.py` script. It will train and evaluate all the models specified in the `run_experiments.py` file.

```bash
python run_experiments.py
```

### Collect data for VAE models
There are also some script to recollect data (RGB and segmentation images) from CARLA. To collect data from CARLA manually, run:
```bash
python carla_env/envs/collect_data_manual_env.py
```

To collect data from CARLA automatically using a RL agent in early stages, run:
```bash
python carla_env/envs/collect_data_rl_env.py
```

