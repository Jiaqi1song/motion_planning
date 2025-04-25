# recorder.py
import os, h5py, numpy as np

class CarlaRecorder:
    """
    Streams (obs, act) pairs into an HDF5 file without filling RAM.
    Each episode is stored as its own group: /ep_<idx>/…
    """
    def __init__(self, out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.f = h5py.File(out_path, "w")
        self.ep  = None
        self.i   = 0      # sample index in current episode
        self.eidx= 0      # episode counter
        self.frames = 0

    def start_episode(self):
        self.ep  = self.f.create_group(f"ep_{self.eidx:05d}")
        self.i   = 0
        self.eidx+=1

    def add(self, bev_img, agent_feats, teacher_act):
        g   = self.ep
        # create datasets lazily on first sample ─ fixed‑length chunks keep I/O quick
        if self.i == 0:
            g.create_dataset("bev",       (0,)+bev_img.shape, maxshape=(None,)+bev_img.shape,
                             dtype=np.uint8,  chunks=True)
            g.create_dataset("agents",    (0,)+agent_feats.shape, maxshape=(None,)+agent_feats.shape,
                             dtype=np.float32, chunks=True)
            g.create_dataset("action",    (0,teacher_act.shape[-1]), maxshape=(None,teacher_act.shape[-1]),
                             dtype=np.float32, chunks=True)
        for name, arr in zip(("bev","agents","action"),
                             (bev_img, agent_feats, teacher_act)):
            dset = g[name]
            dset.resize(self.i+1, axis=0)
            dset[self.i] = arr
        self.i += 1
        self.frames += 1

    def close(self):
        self.f.close()
