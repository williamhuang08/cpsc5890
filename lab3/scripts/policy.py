# policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from bc import BCConvMLPPolicy
from action_vae import ActionVAE

import numpy as np

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
      device = "cuda" if torch.cuda.is_available() else "cpu"
      bc_model_kwargs = dict(
        action_dim=8,
        obs_dim=8,
        obs_horizon=2,
        pred_horizon=16,
        image_type="both",
      )
      # TODO: load model, init buffers, etc.
      self.model = BCConvMLPPolicy(**bc_model_kwargs).to(device)

      ckpt_path = "asset/checkpoints/bcconv_latent_final.pt"
      ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
      self.model = ActionVAE(**ckpt).to(device)

    def reset(self) -> None:
        # TODO: reset hidden state / buffers
        pass

    def step(self, obs: Dict[str, Any]) -> PolicyOut:
        joints = np.asarray(obs["joint_positions"], dtype=np.float32)

        # TODO: replace with your model inference
        action = joints.copy()  # safe default: hold

        return PolicyOut(action=action, info=None)