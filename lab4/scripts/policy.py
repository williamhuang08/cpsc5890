# policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from diffusers import DDIMScheduler

from scripts.unet import DiffusionPolicyUNet

import torch

@dataclass
class PolicyOut:
    action: np.ndarray
    info: Optional[Dict[str, Any]] = None

class UniversalPolicy:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(
            "asset/policy/DiT_w_stats_state_lift_cube_final.pth",
            map_location=self.device,
        )

        meta = ckpt["meta"]
        self.pred_horizon = meta["pred_horizon"]
        self.obs_horizon = meta["obs_horizon"]
        self.action_dim = meta["action_dim"]
        self.model = DiffusionPolicyUNet(obs_low_dim=meta["obs_dim"],action_dim=meta["action_dim"],action_horizon=meta["pred_horizon"],obs_horizon=meta["obs_horizon"],image_type="both",).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.scheduler = DDIMScheduler(
            num_train_timesteps=meta["num_diffusion_steps"],
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )

        self.scheduler.set_timesteps(10)  

        self.stats = ckpt.get("stats", None)

        self.state_buffer = []
        self.img_buffer = []
        self.wimg_buffer = []

    def reset(self) -> None:
        PolicyOut(action = np.deg2rad(np.array([-0.2,-45.4,-0.3,21.8,0.4,67.2,0.6])))
        self.state_buffer = []
        self.img_buffer = []
        self.wimg_buffer = []

    def step(self, obs: Dict[str, Any]) -> PolicyOut:

      joints = np.asarray(obs["joint_positions"], dtype=np.float32)
      img = np.asarray(obs["base_rgb"], dtype=np.float32)
      wimg = np.asarray(obs["wrist_rgb"], dtype=np.float32)

      img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).to(self.device) 
      wimg = torch.tensor(wimg, dtype=torch.float32).permute(2, 0, 1).to(self.device)  

      img_mean = torch.tensor(self.stats["img_mean"], dtype=torch.float32).reshape(3,1,1).to(self.device)
      img_std  = torch.tensor(self.stats["img_std"],  dtype=torch.float32).reshape(3,1,1).to(self.device)

      wimg_mean = torch.tensor(self.stats["wimg_mean"], dtype=torch.float32).reshape(3,1,1).to(self.device)
      wimg_std  = torch.tensor(self.stats["wimg_std"],  dtype=torch.float32).reshape(3,1,1).to(self.device)

      img = (img - img_mean) / img_std
      wimg = (wimg - wimg_mean) / wimg_std

    #   img = _center_crop_hwc(img, 96, 96)
    #   wimg = _center_crop_hwc(wimg, 96, 96)


      joints = (torch.tensor(joints).to(self.device) - self.stats["s_mean"])/self.stats["s_std"]


      self.state_buffer.append(joints.detach().clone())
      self.img_buffer.append(img)
      self.wimg_buffer.append(wimg)
      
      if len(self.state_buffer) > self.obs_horizon:
        self.state_buffer = self.state_buffer[-self.obs_horizon:]
      if len(self.img_buffer) > self.obs_horizon:
        self.img_buffer = self.img_buffer[-self.obs_horizon:]
      if len(self.wimg_buffer) > self.obs_horizon:
        self.wimg_buffer = self.wimg_buffer[-self.obs_horizon:]
  
      state_seq = torch.stack(self.state_buffer, dim=0).float()   # (T,D)
      img_seq   = torch.stack(self.img_buffer, dim=0).float()     # (T,3,H,W)
      wimg_seq  = torch.stack(self.wimg_buffer, dim=0).float()    # (T,3,H,W)

      state_t = state_seq.unsqueeze(0)  # (1,T,D)
      img_t   = img_seq.unsqueeze(0)    # (1,T,3,H,W)
      wimg_t  = wimg_seq.unsqueeze(0)   # (1,T,3,H,W)
      with torch.no_grad():
          sample = torch.randn((1, self.pred_horizon, self.action_dim),device=self.device,
          )

          for t in self.scheduler.timesteps:
              noise_pred = self.model(
                noisy_actions=sample,
                timesteps=t,
                observations=state_t,
                img_ext=img_t,
                img_wst=wimg_t,
              )

              sample = self.scheduler.step(
                  model_output=noise_pred,
                  timestep=t,
                  sample=sample,).prev_sample

      action_seq = sample[0]
      action_seq = action_seq * torch.tensor(self.stats["a_std"], device=self.device) + torch.tensor(self.stats["a_mean"], device=self.device)
      action = action_seq[0].cpu().numpy() 

      return PolicyOut(action=action)