import argparse
import config
import time
from tqdm import tqdm
parser = argparse.ArgumentParser(description="Trains a CARLA agent")
parser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--total_timesteps", type=int, default=1_000_00, help="Total timestep to train for")
parser.add_argument("--reload_model", type=str, default="", help="Path to a model to reload")
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
parser.add_argument("--num_checkpoints", type=int, default=20, help="Checkpoint frequency")
parser.add_argument("--config", type=str, default="1", help="Config to use (default: 1)")
parser.add_argument("--map", type=str, default="Town10HD", help="Map used in the environment (default: Town07)")
args = vars(parser.parse_args())
config.set_config(args["config"])

from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from carla_env.envs.carla_driving_env import CarlaDrivingEnv

from vae.utils.misc import LSIZE
from carla_env.state_commons import create_encode_state_fn, load_vae, encode_state_dqgat
from carla_env.rewards import reward_functions
from utils import HParamCallback, TensorboardCallback, write_json, parse_wrapper_class
from model.dq_gat import *
import os

from config import CONFIG
print(CONFIG)
log_dir = 'tensorboard'
os.makedirs(log_dir, exist_ok=True)
reload_model = args["reload_model"]
total_timesteps = args["total_timesteps"]

seed = CONFIG["seed"]

algorithm_dict = {"PPO": PPO, "DDPG": DDPG, "SAC": SAC}
if CONFIG["algorithm"] not in algorithm_dict:
    raise ValueError("Invalid algorithm name")

AlgorithmRL = algorithm_dict[CONFIG["algorithm"]]
dqgat_model = DQGAT()
encode_state_fn = encode_state_dqgat(dqgat_model)
from carla_env.envs.carla_autopilot_env import CarlaAutoPilotEnv
env = CarlaAutoPilotEnv(
    host=args["host"], port=args["port"],
                    reward_fn=reward_functions[CONFIG["reward_fn"]],
                    encode_state_fn=encode_state_fn, 
                    fps=args["fps"], action_smoothing=CONFIG["action_smoothing"],
                    action_space_type='continuous', activate_render=args["no_render"], map=args["map"],
                    training_scene_names=CONFIG["training_scenes"]
)   # <- flag
obs = env.reset()
done = False
target_frames = 150000

# Initialize tqdm progress bar
with tqdm(total=target_frames, desc="Recording frames") as pbar:
    while env.recorder.frames < target_frames:
        prev_frames = env.recorder.frames

        # Step the environment
        obs, _, done, _ = env.step(None)

        # Update progress bar by the number of new frames recorded
        pbar.update(env.recorder.frames - prev_frames)

        if done:
            env.reset()  # Reset if episode ends

env.close()