import os
import torch
import torch.nn as nn
from tqdm import trange
from collections import deque
import random
import argparse
import time
from model.dq_gat import BCPolicy, DQGAT

import config
parser = argparse.ArgumentParser(description="Trains a CARLA agent")
parser.add_argument("--host", default="localhost", type=str)
parser.add_argument("--port", default=2000, type=int)
parser.add_argument("--total_timesteps", type=int, default=234)
parser.add_argument("--reload_model", type=str, default="")
parser.add_argument("--no_render", action="store_false")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--num_checkpoints", type=int, default=50)
parser.add_argument("--config", type=str, default="1")
parser.add_argument("--map", type=str, default="Town10HD")
args = vars(parser.parse_args())
config.set_config(args["config"])
from config import CONFIG
from carla_env.envs.carla_autopilot_env import CarlaAutoPilotEnv
from carla_env.state_commons import encode_state_dqgat
from carla_env.rewards import reward_functions
# ------------------ Argument Parsing ------------------ #



print(CONFIG)

# ------------------ Logging & Setup ------------------ #
log_dir = 'tensorboard'
os.makedirs(log_dir, exist_ok=True)
reload_model = args["reload_model"]
NUM_EPISODE = args["total_timesteps"]
SAVE_FREQ = NUM_EPISODE // args["num_checkpoints"]
LR = 4e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Replay Buffer ------------------ #
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, act):
        self.buffer.append((obs, act))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        obs_batch, act_batch = zip(*samples)
        return obs_batch, act_batch

    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer()
batch_size = 64
update_freq = 64

# ------------------ Load Policy ------------------ #
policy = BCPolicy().to(device)
policy.load_state_dict(torch.load("weights/bc_policy_best.pt"))
policy.train()

optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPISODE)
loss_fn = nn.MSELoss()
alpha = 0.98  # for running loss

# ------------------ Create Env ------------------ #
dqgat_model = DQGAT()
encode_state_fn = encode_state_dqgat(dqgat_model)
env = CarlaAutoPilotEnv(
    host=args["host"],
    port=args["port"],
    reward_fn=reward_functions[CONFIG["reward_fn"]],
    encode_state_fn=encode_state_fn,
    fps=args["fps"],
    action_smoothing=CONFIG["action_smoothing"],
    action_space_type='continuous',
    activate_render=args["no_render"],
    map=args["map"],
    training_scene_names=CONFIG["training_scenes"]
)

obs = env.reset()
pbar = trange(NUM_EPISODE, desc="evals", dynamic_ncols=True)
success_cnt = 0
time_vec = []
start = time.perf_counter()
env.teacher_percent = 0
# ------------------ Training Loop ------------------ #
for step in pbar:
    while True:
        obs_torch = {
            "bev_image": torch.tensor(obs["bev_image"], dtype=torch.float32).unsqueeze(0).to(device),
            "agent_feats": torch.tensor(obs["agent_feats"], dtype=torch.float32).to(device)
        }

        model_action = policy(obs_torch).squeeze(0)
        # teacher_action = torch.tensor(env.get_teacher_action(), dtype=torch.float32).to(device)

        # Add to buffer
        # replay_buffer.add(obs, teacher_action)

        # Dynamic blend: alpha grows linearly with step
        obs, _, done, _ = env.step(model_action.detach().cpu().numpy())

        
        if done:
            # print(start)
            end = time.perf_counter()
            # print(end)
            # print(end-start)
            if env.success_state:
                success_cnt += 1
                elapsed = end - start
                time_vec.append(elapsed)
                # print(elapsed)
                
            obs = env.reset()
            start = time.perf_counter()
            break
        pbar.set_postfix(success_cnt=success_cnt)
print(f"success_rate: {success_cnt/NUM_EPISODE:.3f}")
print(f"avg_time: {sum(time_vec)/len(time_vec):.3f}")
env.close()
