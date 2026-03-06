# policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from scripts.bc import BCConvMLPPolicy

import torch

@dataclass
class PolicyOut:
    action: np.ndarray
    info: Optional[Dict[str, Any]] = None

class UniversalPolicy:
    """
    Students implement:
      - reset()
      - step(obs) -> PolicyOut

    obs (from RobotEnv) will typically include:
      obs["joint_positions"] : (D,)
      obs["base_rgb"]        : (H,W,3) uint8
      obs["wrist_rgb"]       : (H,W,3) uint8
    """
    def __init__(self):
        # TODO: load model, init buffers, etc.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(
            "asset/checkpoints/bcconv_final.pt",
            map_location=self.device,
            weights_only=False
        )
        self.stats = ckpt["stats"]
        self.model = BCConvMLPPolicy(**ckpt["model_kwargs"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.obs_horizon = ckpt["model_kwargs"]["obs_horizon"]
        self.state_buffer = []
        self.img_buffer = []
        self.wimg_buffer = []

    def reset(self) -> None:
        # TODO: reset hidden state / buffers
        PolicyOut(action = np.deg2rad(np.array([-0.2,-45.4,-0.3,21.8,0.4,67.2,0.6])))
        self.state_buffer = []
        self.img_buffer = []
        self.wimg_buffer = []
        
    def step(self, obs: Dict[str, Any]) -> PolicyOut:

    #   def _center_crop_hwc(img: np.ndarray, crop_h: int = 96, crop_w: int = 96) -> np.ndarray:
    #       """
    #       img: (H, W, 3) numpy array
    #       return: (3, crop_h, crop_w) numpy array (CHW)
    #       """
    #       H, W, C = img.shape
    #       assert C == 3, f"Expected 3 channels, got {C}"

    #       if H < crop_h or W < crop_w:
    #           raise ValueError(f"Image too small to crop: {(H, W)} < {(crop_h, crop_w)}")

    #       top = (H - crop_h) // 2
    #       left = (W - crop_w) // 2

    #       cropped = img[top:top + crop_h, left:left + crop_w, :]  # (96,96,3)
    #       cropped = np.transpose(cropped, (2, 0, 1))               # (3,96,96)
    #       return cropped
    
      joints = np.asarray(obs["joint_positions"], dtype=np.float32)
      img = np.asarray(obs["base_rgb"], dtype=np.float32)
      wimg = np.asarray(obs["wrist_rgb"], dtype=np.float32)

    #   img = _center_crop_hwc(img, 96, 96)
    #   wimg = _center_crop_hwc(wimg, 96, 96)

      img_mean = self.stats["img_mean"].reshape(3, 1, 1)
      img_std  = self.stats["img_std"].reshape(3, 1, 1)

      wimg_mean = self.stats["wimg_mean"].reshape(3, 1, 1)
      wimg_std  = self.stats["wimg_std"].reshape(3, 1, 1)

      joints = (joints - self.stats["s_mean"])/self.stats["s_std"]
      img = (img - img_mean)/img_std
      wimg = (wimg - wimg_mean)/wimg_std

      self.state_buffer.append(joints.copy())
      self.img_buffer.append(img)
      self.wimg_buffer.append(wimg)
      
      if len(self.state_buffer) > self.obs_horizon:
        self.state_buffer = self.state_buffer[-self.obs_horizon:]
      if len(self.img_buffer) > self.obs_horizon:
        self.img_buffer = self.img_buffer[-self.obs_horizon:]
      if len(self.wimg_buffer) > self.obs_horizon:
        self.wimg_buffer = self.wimg_buffer[-self.obs_horizon:]
  
      state_seq = np.stack(self.state_buffer, axis=0).astype(np.float32)  # (T,D)
      img_seq   = np.stack(self.img_buffer, axis=0).astype(np.float32)    # (T,3,96,96)
      wimg_seq  = np.stack(self.wimg_buffer, axis=0).astype(np.float32)   # (T,3,96,96)

      state_t = torch.from_numpy(state_seq).unsqueeze(0).to(self.device)  # (1,T,D)
      img_t   = torch.from_numpy(img_seq).unsqueeze(0).to(self.device)    # (1,T,3,96,96)
      wimg_t  = torch.from_numpy(wimg_seq).unsqueeze(0).to(self.device)   # (1,T,3,96,96)

      action = self.model(state_t, img_t, wimg_t)
      action = action.detach().cpu().numpy()
      action = action * self.stats["a_std"] + self.stats["a_mean"]

      # normalize shape to (K, 8)
      action = np.asarray(action, dtype=np.float32)
      if action.ndim == 1:          # (8,)
          action = action[None, :]  # (1,8)
      elif action.ndim == 2:        # (1,8) or (K,8)
          pass
      elif action.ndim == 3:        # (1,K,8)
          action = action[0]        # (K,8)
      else:
          raise ValueError(f"Unexpected action shape: {action.shape}")

      return action