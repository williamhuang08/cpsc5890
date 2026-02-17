# policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from bc import BCConvMLPPolicy
from action_vae import ActionVAE

import numpy as np
import torch

from collections import deque
import cv2

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

      self.device = "cuda" if torch.cuda.is_available() else "cpu"
      # bc_model_kwargs = dict(
      #   action_dim=8,
      #   obs_dim=8,
      #   obs_horizon=2,
      #   pred_horizon=16,
      #   image_type="both",
      # )

      # self.model = BCConvMLPPolicy(**bc_model_kwargs).to(device)

      # TODO: load model, init buffers, etc.
      ckpt_path = "asset/checkpoints/bcconv_latent_final.pt"
      # ckpt_path = "asset/checkpoints/bcconv_final.pt"
      ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

      # self.model = BCConvMLPPolicy(**ckpt["model_kwargs"]).to(self.device)
      self.model = ActionVAE(**ckpt["policy_kwargs"]).to(self.device)
      self.model.load_state_dict(ckpt["model_state_dict"])

      self.model.eval()
      self.stats = ckpt["stats"]

      self.obs_horizon = self.model.obs_horizon
      self.obs_dim = self.model.obs_dim
      self.pred_horizon = self.model.pred_horizon
    
      self.buffer = deque()

    def reset(self) -> None:
        # TODO: reset hidden state / buffers
        self.buffer = deque()

    def step(self, obs: Dict[str, Any]) -> PolicyOut:
        joints = np.asarray(obs["joint_positions"], dtype=np.float32)

        H, W = 480, 640
        if len(self.buffer) == 0:
          for _ in range(2):
            # zero_obs_jp = torch.zeros((self.obs_dim))
            # zero_obs_base = torch.zeros(H, W, 3)
            # zero_obs_wrist = torch.zeros(H, W, 3)
            zero_obs_jp = obs["joint_positions"]
            zero_obs_base = obs["base_rgb"]
            zero_obs_wrist = obs["wrist_rgb"]

            zero_obs = dict(
              joint_positions = zero_obs_jp,
              base_rgb = zero_obs_base,
              wrist_rgb = zero_obs_wrist
            )
            self.buffer.append(zero_obs)

        # TODO: replace with your model inference
        # action = joints.copy()  # safe default: hold
        self.buffer.popleft()
        self.buffer.append(obs)

        obs_jp_h, obs_base_h, obs_wrist_h = [], [], []
        for i in range(len(self.buffer)):
          obsi = self.buffer[i]
          obs_jp = obsi["joint_positions"]
          obs_base = cv2.resize(np.array(obsi["base_rgb"][:, 80:561, :]), (96, 96))
          obs_base = np.transpose(obs_base, (2, 0, 1))
          obs_wrist = cv2.resize(np.array(obsi["wrist_rgb"][:, 80:561, :]), (96, 96))
          obs_wrist = np.transpose(obs_wrist, (2, 0, 1))

          obs_jp_h.append(obs_jp)
          obs_base_h.append(obs_base)
          obs_wrist_h.append(obs_wrist)

        # print(obs_base_h)
        obs_jp_h = np.array(obs_jp_h)
        obs_base_h = np.array(obs_base_h)
        obs_wrist_h = np.array(obs_wrist_h)

        obs_jp_h = torch.tensor(np.expand_dims(obs_jp_h, axis = 0), device=self.device).float()
        obs_base_h = torch.tensor(np.expand_dims(obs_base_h, axis =  0), device = self.device).float()
        obs_wrist_h = torch.tensor(np.expand_dims(obs_wrist_h, axis=0), device=self.device).float()

        with torch.no_grad():
          action = self.model(obs_jp_h, obs_base_h, obs_wrist_h).cpu().squeeze(0)
          print(obs_jp_h)
          print(action[0])

        return PolicyOut(action=action, info=None)
