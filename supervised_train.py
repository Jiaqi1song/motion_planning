# dataset.py
import h5py, torch
from torch.utils.data import Dataset

class CarlaH5Dataset(Dataset):
    """
    Exposes (bev_image, agent_feats) → action triples
    recorded by CarlaRecorder.
    """
    def __init__(self, h5_path, train=True, val_split=0.1):
        self.f      = h5py.File(h5_path, "r")
        self.ptrs   = []          # (ep_key, idx_inside_ep)
        for ep in self.f.keys():
            if not all(k in self.f[ep] for k in ["bev", "agents", "action"]):
                print(f"[WARN] Skipping broken episode: {ep}")
                continue
            n = self.f[ep]["bev"].shape[0]
            self.ptrs.extend([(ep, i) for i in range(n)])

        # simple train/val split
        val_len = int(len(self.ptrs)*val_split)
        if train:
            self.ptrs = self.ptrs[val_len:]
        else:
            self.ptrs = self.ptrs[:val_len]

        # imagenet stats for BEV RGB images (already in your env)
        self.mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    def __len__(self): return len(self.ptrs)

    def __getitem__(self, idx):
        ep,i = self.ptrs[idx]
        bev_u8  = self.f[ep]["bev"][i]           # (3,224,224) uint8
        agents  = self.f[ep]["agents"][i]        # (20,12) float32
        action  = self.f[ep]["action"][i]        # (3,)   float32

        bev = torch.from_numpy(bev_u8).float()/255.0
        bev = (bev - self.mean)/self.std         # normalise

        agents = torch.from_numpy(agents).float()
        act    = torch.from_numpy(action).float()

        return {"bev_image": bev, "agent_feats": agents}, act
# main.py (snippet)
import torch, tqdm, math
from model.dq_gat import BCPolicy


device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy     = BCPolicy().to(device)
# print(policy)
from torch.utils.data import DataLoader
train_ds = CarlaH5Dataset("carla_logs/teacher.hdf5", train=True)
val_ds   = CarlaH5Dataset("carla_logs/teacher.hdf5", train=False)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=0)

optim      = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-2)
criterion  = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, mode="min", factor=0.5, patience=3, verbose=True)

def run_epoch(loader, train=True):
    policy.train(train)
    loss_sum, n = 0.0, 0
    torch.set_grad_enabled(train)

    pbar = tqdm.tqdm(loader, leave=False)
    for obs, act in pbar:
        obs = {k: v.to(device, non_blocking=True) for k, v in obs.items()}
        act = act.to(device)

        pred = policy(obs)
        loss = criterion(pred, act)

        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_sum += loss.item() * act.size(0)
        n += act.size(0)

        # 🔥 Update tqdm bar with average loss so far
        pbar.set_postfix(loss=f"{loss_sum / n:.4f}")

    return loss_sum / n

best_val = math.inf
for epoch in range(50):
    train_loss = run_epoch(train_loader, train=True)
    val_loss   = run_epoch(val_loader,   train=False)
    scheduler.step(val_loss)
    print(f"Epoch {epoch:02d} │ train {train_loss:.4f}  val {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(policy.state_dict(), "weights/bc_policy_best.pt")
