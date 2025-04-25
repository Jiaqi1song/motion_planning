import os
import torch
import torch.nn as nn
from tqdm import trange
from collections import deque
import random
import argparse

from model.dq_gat import BCPolicy, DQGAT

import config
parser = argparse.ArgumentParser(description="Trains a CARLA agent")
parser.add_argument("--host", default="localhost", type=str)
parser.add_argument("--port", default=2000, type=int)
parser.add_argument("--total_timesteps", type=int, default=1_000_000)
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
NUM_STEPS = args["total_timesteps"]
SAVE_FREQ = NUM_STEPS // args["num_checkpoints"]
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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEPS)
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
pbar = trange(NUM_STEPS, desc="Training", dynamic_ncols=True)
running_loss = 0.0

# ------------------ Training Loop ------------------ #
for step in pbar:
    obs_torch = {
        "bev_image": torch.tensor(obs["bev_image"], dtype=torch.float32).unsqueeze(0).to(device),
        "agent_feats": torch.tensor(obs["agent_feats"], dtype=torch.float32).to(device)
    }

    model_action = policy(obs_torch).squeeze(0)
    teacher_action = torch.tensor(env.get_teacher_action(), dtype=torch.float32).to(device)

    # Add to buffer
    replay_buffer.add(obs, teacher_action)

    # Dynamic blend: alpha grows linearly with step
    env.teacher_percent = max(0.0, 0.6 - 0.35 * step / NUM_STEPS)
    obs, _, done, _ = env.step(model_action.detach().cpu().numpy())

    # Periodic updates
    if len(replay_buffer) >= batch_size and step % update_freq == 0:
        obs_batch, act_batch = replay_buffer.sample(batch_size)
        bev_imgs = torch.tensor([o["bev_image"] for o in obs_batch], dtype=torch.float32).to(device)
        agent_feats = torch.tensor([o["agent_feats"][0] for o in obs_batch], dtype=torch.float32).to(device)
        targets = torch.stack(act_batch).to(device)

        preds = policy({"bev_image": bev_imgs, "agent_feats": agent_feats})
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss = loss.item() if step == 0 else alpha * running_loss + (1 - alpha) * loss.item()
        pbar.set_postfix(loss=loss.item(), running_loss=running_loss, t_per=env.teacher_percent, lr=scheduler.get_last_lr()[0])

    if done:
        obs = env.reset()

    if (step + 1) % SAVE_FREQ == 0:
        os.makedirs("weights/online", exist_ok=True)
        torch.save(policy.state_dict(), f"weights/online/bc_online_step{step+1}_{running_loss:.3f}.pt")

env.close()
