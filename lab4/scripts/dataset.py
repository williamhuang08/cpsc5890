    
import numpy as np

from typing import Optional, Tuple, Union, Dict, List

import torch
from torch.utils.data import Dataset


import os
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm

def _list_episode_files(data_dir: str) -> List[str]:
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".npy")
    ]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No .npy episode files found in: {data_dir}")
    return files


def _load_episode_steps(path: str) -> np.ndarray:
    """
    Returns a 1D object array of length T where each element is a dict.
    """
    ep = np.load(path, allow_pickle=True)
    if not (isinstance(ep, np.ndarray) and ep.dtype == object and ep.ndim == 1):
        raise TypeError(f"{path}: expected 1D object array, got dtype={getattr(ep,'dtype',None)} shape={getattr(ep,'shape',None)}")
    if ep.size == 0:
        raise ValueError(f"{path}: empty episode")
    if not isinstance(ep[0], dict):
        raise TypeError(f"{path}: expected ep[0] to be dict, got {type(ep[0])}")
    return ep


def load_dir_episodes(data_dir: str) -> Dict:
    """
    Each episode file is a .npy containing a 1D object array of length T.
    Each element is a dict with keys:
        - "state": (D_obs,)
        - "action": (D_act,)
        - "image": (H, W, 3)
        - "wrist_image": (H, W, 3)
    """
    episode_files = _list_episode_files(data_dir)

    # -------------------------
    # First pass: count total T
    # -------------------------
    total = 0
    ranges: list[tuple[int, int]] = []
    ep_lengths: list[int] = []

    for p in tqdm(episode_files, desc="Counting datapoints"):
        ep = _load_episode_steps(p)
        T = int(len(ep))
        ranges.append((total, total + T))
        ep_lengths.append(T)
        total += T

    # -------------------------
    # Infer dimensions from first step
    # -------------------------
    ep0 = _load_episode_steps(episode_files[0])
    step0 = ep0[0]

    state0 = np.asarray(step0["state"])
    action0 = np.asarray(step0["action"])
    img0 = np.asarray(step0["image"])
    wst0 = np.asarray(step0["wrist_image"])

    if state0.ndim != 1:
        raise ValueError(f"state must be (D,), got {state0.shape}")
    if action0.ndim != 1:
        raise ValueError(f"action must be (D,), got {action0.shape}")
    if img0.ndim != 3 or img0.shape[-1] != 3:
        raise ValueError(f"image must be (H,W,3), got {img0.shape}")
    if wst0.ndim != 3 or wst0.shape[-1] != 3:
        raise ValueError(f"wrist_image must be (H,W,3), got {wst0.shape}")

    D_obs = int(state0.shape[0])
    D_act = int(action0.shape[0])

    H_ext, W_ext = int(img0.shape[0]), int(img0.shape[1])
    H_wst, W_wst = int(wst0.shape[0]), int(wst0.shape[1])

    # -------------------------
    # Allocate
    # -------------------------
    obs = np.empty((total, D_obs), dtype=np.float32)
    act = np.empty((total, D_act), dtype=np.float32)
    img_ext = np.empty((total, H_ext, W_ext, 3), dtype=np.uint8)
    img_wst = np.empty((total, H_wst, W_wst, 3), dtype=np.uint8)

    # -------------------------
    # Fill
    # -------------------------
    cursor = 0
    for p in tqdm(episode_files, desc="Loading episodes"):
        ep = _load_episode_steps(p)
        T = int(len(ep))

        # stack per-step
        # (Using list -> np.asarray is fine; if you want faster, we can pre-allocate per-episode and fill.)
        state = np.asarray([s["state"] for s in ep], dtype=np.float32)          # (T, D_obs)
        action = np.asarray([s["action"] for s in ep], dtype=np.float32)        # (T, D_act)
        ext = np.asarray([s["image"] for s in ep], dtype=np.uint8)              # (T, H, W, 3)
        wst = np.asarray([s["wrist_image"] for s in ep], dtype=np.uint8)        # (T, H, W, 3)

        # sanity checks
        if state.shape != (T, D_obs):
            raise ValueError(f"{p}: state shape {state.shape}, expected {(T, D_obs)}")
        if action.shape != (T, D_act):
            raise ValueError(f"{p}: action shape {action.shape}, expected {(T, D_act)}")
        if ext.shape != (T, H_ext, W_ext, 3):
            raise ValueError(f"{p}: image shape {ext.shape}, expected {(T, H_ext, W_ext, 3)}")
        if wst.shape != (T, H_wst, W_wst, 3):
            raise ValueError(f"{p}: wrist_image shape {wst.shape}, expected {(T, H_wst, W_wst, 3)}")

        obs[cursor:cursor + T] = state
        act[cursor:cursor + T] = action
        img_ext[cursor:cursor + T] = ext
        img_wst[cursor:cursor + T] = wst

        cursor += T

    return {
        "obs": torch.from_numpy(obs),
        "act": torch.from_numpy(act),
        "img_ext": torch.from_numpy(img_ext),
        "img_wst": torch.from_numpy(img_wst),
        "episode_ranges": ranges,
    }

class DiffusionDatasetWrist(Dataset):
    """Episode-aware dataset with train-only normalization stats (supports two image streams)."""
    def __init__(
        self,
        obs_data: torch.Tensor,                 # (N, D_obs) float32
        action_data: torch.Tensor,              # (N, D_act) float32
        img_wst: torch.Tensor,                  # (N, H, W, C) uint8
        episode_ranges: List[Tuple[int, int]],  # [(start, end_exclusive), ...]
        indices: Optional[np.ndarray],          # valid window end indices
        obs_horizon: int,
        action_horizon: int,
        pred_horizon: int,
        # normalization stats (computed on train set)
        obs_mean: Optional[torch.Tensor] = None,
        obs_std: Optional[torch.Tensor] = None,
        action_mean: Optional[torch.Tensor] = None,
        action_std: Optional[torch.Tensor] = None,
        img_wst_mean: Optional[torch.Tensor] = None,
        img_wst_std: Optional[torch.Tensor] = None,
    ):

        self.obs_data = obs_data
        self.action_data = action_data
        self.img_wst = img_wst
        self.episode_ranges = episode_ranges

        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon

        # build window indices inside episodes if not provided
        if indices is None:
            self.valid_indices = self._build_episode_window_indices()
        else:
            self.valid_indices = np.asarray(indices, dtype=np.int64)

        # stats
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.img_wst_mean = img_wst_mean
        self.img_wst_std = img_wst_std

        # sanity checks
        n = self.obs_data.shape[0]
        assert self.action_data.shape[0] == n
        # assert self.img_ext.shape[0] == n
        assert self.img_wst.shape[0] == n

    # --------- helpers ---------

    def _build_episode_window_indices(self) -> np.ndarray:
        """All valid window end indices that stay fully inside each episode."""
        idxs = []
        H_obs, H_act = self.obs_horizon, self.action_horizon
        for (s, e) in self.episode_ranges:
            i_min = s + H_obs - 1
            i_max = e - H_act
            if i_min < i_max:
                idxs.extend(range(i_min, i_max))
        return np.array(idxs, dtype=np.int64)

    @staticmethod
    def _compute_obs_action_stats_subset(
        obs: torch.Tensor,
        act: torch.Tensor,
        indices: np.ndarray,
        obs_h: int,
        act_h: int
    ):
        """Compute obs mean/std but ONLY min/max for actions (no action mean/std)."""

        # ----- gather indices -----
        obs_rows, act_rows = [], []
        for i in indices.tolist():
            obs_rows.extend(range(i - obs_h + 1, i + 1))
            act_rows.extend(range(i, i + act_h))

        obs_rows = torch.as_tensor(sorted(set(obs_rows)), dtype=torch.long)
        act_rows = torch.as_tensor(sorted(set(act_rows)), dtype=torch.long)

        obs_sel = obs[obs_rows].to(torch.float64)
        act_sel = act[act_rows].to(torch.float64)

        # ----- obs stats stay mean/std -----
        obs_mean = obs_sel.mean(dim=0).to(torch.float32)
        obs_std  = obs_sel.std(dim=0, unbiased=False).to(torch.float32) + 1e-8

        # ✅ identity stats for all dims starting from the 4th (index 3)
        obs_mean[3:] = 0.0
        obs_std[3:]  = 1.0

        act_mean = act_sel.mean(dim=0).to(torch.float32)
        act_std = act_sel.std(dim=0, unbiased=False).to(torch.float32) + 1e-8

        # ----- action min-max ONLY -----
        act_min = act_sel.min(dim=0).values.to(torch.float32)
        act_max = act_sel.max(dim=0).values.to(torch.float32)

        return obs_mean, obs_std, act_mean, act_std, act_min, act_max

    # @staticmethod
    # def _compute_obs_action_stats_subset(obs: torch.Tensor, act: torch.Tensor, indices: np.ndarray, obs_h: int, act_h: int):
    #     """Compute mean/std over subset of timesteps used in given windows (for train only)."""
    #     obs_rows, act_rows = [], []
    #     for i in indices.tolist():
    #         obs_rows.extend(range(i - obs_h + 1, i + 1))
    #         act_rows.extend(range(i, i + act_h))
    #     obs_rows = torch.as_tensor(sorted(set(obs_rows)), dtype=torch.long)
    #     act_rows = torch.as_tensor(sorted(set(act_rows)), dtype=torch.long)

    #     obs_sel = obs[obs_rows].to(torch.float64)
    #     act_sel = act[act_rows].to(torch.float64)

    #     obs_mean = obs_sel.mean(dim=0).to(torch.float32)
    #     obs_std = obs_sel.std(dim=0, unbiased=False).to(torch.float32) + 1e-8
    #     # act_mean = act_sel.mean(dim=0).to(torch.float32)
    #     # act_std = act_sel.std(dim=0, unbiased=False).to(torch.float32) + 1e-8
    #     return obs_mean, obs_std, 

    @staticmethod
    def _compute_image_stats_subset_chunked(img: torch.Tensor, indices: np.ndarray, obs_h: int, chunk: int = 8192):
        """Compute per-channel mean/std of images appearing in the given observation windows."""
        N, H, W, C = img.shape
        used_rows = []
        for i in indices.tolist():
            used_rows.extend(range(i - obs_h + 1, i + 1))
        used = np.array(sorted(set(used_rows)), dtype=np.int64)

        total = len(used) * H * W
        mean_sum = torch.zeros(C, dtype=torch.float64)
        sq_sum = torch.zeros(C, dtype=torch.float64)

        for start in range(0, len(used), chunk):
            sel = used[start:start + chunk]
            batch = img[sel].to(torch.float32).div_(255.0)  # (B, H, W, C)
            batch = batch.permute(0, 3, 1, 2)               # (B, C, H, W)
            mean_sum += batch.sum(dim=(0, 2, 3)).double()
            sq_sum += (batch ** 2).sum(dim=(0, 2, 3)).double()
            del batch

        mean = mean_sum / total
        var = sq_sum / total - mean ** 2
        img_mean = mean.to(torch.float32)
        img_std = var.clamp_min(1e-8).sqrt().to(torch.float32)
        return img_mean, img_std

    # --------- Dataset API ---------

    def __len__(self) -> int:
        return self.valid_indices.shape[0]

    def __getitem__(self, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            obs_normalized:    (obs_horizon, D_obs)
            img_ext_norm:      (obs_horizon, C, H, W)
            img_wst_norm:      (obs_horizon, C, H, W)
            act_normalized:    (action_horizon, D_act)
        """
        i = int(self.valid_indices[k])

        # window boundaries
        os, oe = i - self.obs_horizon + 1, i + 1
        as_, ae = i, i + self.action_horizon

        obs_w = self.obs_data[os:oe]
        act_w = self.action_data[as_:ae]
        img_wst_w = self.img_wst[os:oe]

        # normalize
        obs = (obs_w - self.obs_mean) / self.obs_std
        act = (act_w - self.action_mean) / self.action_std

        img_wst = img_wst_w.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
        img_wst = (img_wst - self.img_wst_mean.view(1, -1, 1, 1)) / self.img_wst_std.view(1, -1, 1, 1)

        return obs.contiguous(), img_wst.contiguous(), act.contiguous()

class DiffusionDatasetBoth(DiffusionDatasetWrist):
    """Episode-aware dataset with train-only normalization stats (supports two image streams)."""
    def __init__(
        self,
        obs_data: torch.Tensor,                 # (N, D_obs) float32
        action_data: torch.Tensor,              # (N, D_act) float32
        img_ext: torch.Tensor,                  # (N, H, W, C) uint8
        img_wst: torch.Tensor,                  # (N, H, W, C) uint8
        episode_ranges: List[Tuple[int, int]],  # [(start, end_exclusive), ...]
        indices: Optional[np.ndarray],          # valid window end indices
        obs_horizon: int,
        action_horizon: int,
        pred_horizon: int,
        # normalization stats (computed on train set)
        obs_mean: Optional[torch.Tensor] = None,
        obs_std: Optional[torch.Tensor] = None,
        action_mean: Optional[torch.Tensor] = None,
        action_std: Optional[torch.Tensor] = None,
        action_min: Optional[torch.Tensor] = None,
        action_max: Optional[torch.Tensor] = None,
        img_ext_mean: Optional[torch.Tensor] = None,
        img_ext_std: Optional[torch.Tensor] = None,
        img_wst_mean: Optional[torch.Tensor] = None,
        img_wst_std: Optional[torch.Tensor] = None,
    ):
        self.obs_data = obs_data
        self.action_data = action_data
        self.img_ext = img_ext
        self.img_wst = img_wst
        self.episode_ranges = episode_ranges

        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon

        # build window indices inside episodes if not provided
        if indices is None:
            self.valid_indices = self._build_episode_window_indices()
        else:
            self.valid_indices = np.asarray(indices, dtype=np.int64)

        # stats
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.action_min = action_min
        self.action_max = action_max
        self.img_ext_mean = img_ext_mean
        self.img_ext_std = img_ext_std
        self.img_wst_mean = img_wst_mean
        self.img_wst_std = img_wst_std

        # sanity checks
        n = self.obs_data.shape[0]
        assert self.action_data.shape[0] == n
        assert self.img_ext.shape[0] == n
        assert self.img_wst.shape[0] == n

    def norm_data(self, data: dict) -> dict:
        """
        Normalize fields in `data`.
        Supported keys:
          - 'obs'     : (B, D_obs)
          - 'img_ext' : (B, H, W, C)
          - 'img_wst' : (B, H, W, C)
          - 'act'     : (B, D_act)

        Any missing key is ignored. Returns a new dict.
        """
        out = dict(data)  # shallow copy

        # ---- obs ----
        if 'obs' in data and data['obs'] is not None:
            obs = data['obs']
            obs = (obs - self.obs_mean.to(obs.device, obs.dtype)) / self.obs_std.to(obs.device, obs.dtype)
            out['obs'] = obs.contiguous()

        # ---- external images ----
        if 'img_ext' in data and data['img_ext'] is not None:
            img_ext = data['img_ext']
            # (B, H, W, C) -> (B, C, H, W), [0,255] -> [0,1]
            img_ext = img_ext.permute(0, 3, 1, 2).to(torch.float32) / 255.0
            img_ext = (img_ext - self.img_ext_mean.view(1, -1, 1, 1)) / \
                      self.img_ext_std.view(1, -1, 1, 1)
            out['img_ext'] = img_ext.contiguous()

        # ---- wrist images ----
        if 'img_wst' in data and data['img_wst'] is not None:
            img_wst = data['img_wst']
            img_wst = img_wst.permute(0, 3, 1, 2).to(torch.float32) / 255.0
            img_wst = (img_wst - self.img_wst_mean.view(1, -1, 1, 1)) / \
                      self.img_wst_std.view(1, -1, 1, 1)
            out['img_wst'] = img_wst.contiguous()

        # ---- actions ----
        # map [action_min, action_max] -> [-1, 1]
        if 'act' in data and data['act'] is not None:
            act = data['act']
            act = 2 * (act - self.action_min) / (self.action_max - self.action_min) - 1
            out['act'] = act.contiguous()

        return out

    def unnorm_data(self, data: dict, return_uint8: bool = False) -> dict:
        """
        Inverse of norm_data.

        Input (normalized) data:
          - 'obs'     : normalized obs (B, D_obs)
          - 'img_ext' : normalized ext (B, C, H, W)
          - 'img_wst' : normalized wst (B, C, H, W)
          - 'act'     : actions in [-1, 1]

        If return_uint8=True:
          - images are returned as (B, H, W, C) uint8 in [0, 255]
        Else:
          - images are float32 (B, H, W, C) in [0, 1]
        """
        out = dict(data)  # shallow copy

        # ---- obs ----
        if 'obs' in data and data['obs'] is not None:
            obs = data['obs']
            obs = obs * self.obs_std + self.obs_mean
            out['obs'] = obs.contiguous()

        # ---- external images ----
        if 'img_ext' in data and data['img_ext'] is not None:
            img_ext = data['img_ext']
            img_ext = img_ext * self.img_ext_std.view(1, -1, 1, 1) + \
                      self.img_ext_mean.view(1, -1, 1, 1)
            img_ext = img_ext.clamp(0.0, 1.0)
            # (B, C, H, W) -> (B, H, W, C)
            img_ext = img_ext.permute(0, 2, 3, 1)
            if return_uint8:
                img_ext = (img_ext * 255.0).round().clamp(0, 255).to(torch.uint8)
            out['img_ext'] = img_ext.contiguous()

        # ---- wrist images ----
        if 'img_wst' in data and data['img_wst'] is not None:
            img_wst = data['img_wst']
            img_wst = img_wst * self.img_wst_std.view(1, -1, 1, 1) + \
                      self.img_wst_mean.view(1, -1, 1, 1)
            img_wst = img_wst.clamp(0.0, 1.0)
            img_wst = img_wst.permute(0, 2, 3, 1)
            if return_uint8:
                img_wst = (img_wst * 255.0).round().clamp(0, 255).to(torch.uint8)
            out['img_wst'] = img_wst.contiguous()

        # ---- actions ----
        # map [-1, 1] -> [action_min, action_max]
        if 'act' in data and data['act'] is not None:
            act = data['act']
            act = (act + 1) / 2.0
            act = act * (self.action_max - self.action_min) + self.action_min
            out['act'] = act.contiguous()

        return out

    def __getitem__(self, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            obs_normalized:    (obs_horizon, D_obs)
            img_ext_norm:      (obs_horizon, C, H, W)
            img_wst_norm:      (obs_horizon, C, H, W)
            act_normalized:    (action_horizon, D_act)
        """
        i = int(self.valid_indices[k])

        # window boundaries
        os, oe = i - self.obs_horizon + 1, i + 1
        as_, ae = i, i + self.action_horizon

        obs_w = self.obs_data[os:oe]
        img_ext_w = self.img_ext[os:oe]
        img_wst_w = self.img_wst[os:oe]
        act_w = self.action_data[as_:ae]

        # build batch dict for normalization
        batch = {
            "obs": obs_w,
            "img_ext": img_ext_w,
            "img_wst": img_wst_w,
            "act": act_w,
        }

        batch_norm = self.norm_data(batch)

        obs = batch_norm["obs"]      # (To, D_obs)
        img_ext = batch_norm["img_ext"]  # (To, C, H, W)
        img_wst = batch_norm["img_wst"]  # (To, C, H, W)
        act = batch_norm["act"]      # (Ta, D_act)

        return obs, img_ext, img_wst, act