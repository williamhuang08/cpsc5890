
import os
import csv
import yaml
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers import DDIMScheduler

from scripts.dataset import DiffusionDatasetBoth, load_dir_episodes
from scripts.unet import DiffusionPolicyUNet

from scripts.vis_utils import (
    denoise_actions_to_ee_trajs,
    joint_actions_to_q_seq,
    q_seq_to_ee_traj_xyz,
    subsample_list,
    render_denoise_gif_3d,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiffusionPolicyTrainer:
    """
    Trains DiffusionPolicyUNet1D to predict noise on action sequences with global conditioning on observations.
    Uses cosine beta schedule by default.
    """
    def __init__(
        self,
        task_name,
        data_dir,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        num_diffusion_steps: int = 100,
        down_dims=(256, 512, 1024),
        diffusion_step_embed_dim: int = 256,
        kernel_size: int = 5,
        n_groups: int = 8,
        device: Optional[str] = None,
        seed: int = 42,
        num_inference_steps: int = 10,
        eta: float = 0.0,
        image_type: str = "both",
        schedule_type = None
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.num_diffusion_steps = num_diffusion_steps
        self.image_type = image_type
        self.data_dir = data_dir

        self.task_params = yaml.safe_load(open(f"config/{task_name}.yaml"))
        self.build_train_val_datasets(
            data_dir=self.data_dir,
            obs_horizon=self.obs_horizon,
            action_horizon=self.pred_horizon,
            pred_horizon=self.pred_horizon,
            val_ratio=0.1
        )
        # DDIM params
        self.num_inference_steps = num_inference_steps    # number of sampling steps (<= num_inference_steps)
        self.eta = eta                                    # DDIM stochasticity (0 = deterministic)

        # Model
        self.model = DiffusionPolicyUNet(
            obs_low_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            image_type=image_type
            # Any other kwargs like down_dims, kernel_size, n_groups
        ).to(self.device)
        print("Intialized unet")

        self.save_path = "asset/policy/"

        self.noise_scheduler = DDPMScheduler(   
            num_train_timesteps=num_diffusion_steps,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule=schedule_type,
            #'squaredcos_cap_v2', 'linear'
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_steps,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule=schedule_type,
            # clip output to [-1,1] to improve stability
            clip_sample=False,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

    def build_train_val_datasets(self,
        data_dir,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        pred_horizon: int = 8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[DiffusionDatasetBoth, DiffusionDatasetBoth]:
        """Episode split, train-only stats, shared to val (with two image streams)."""
        data = load_dir_episodes(data_dir=data_dir)
        obs = data["obs"]
        act = data["act"]
        img_ext = data["img_ext"]
        img_wst = data["img_wst"]
        ep_ranges = data["episode_ranges"]
        E = len(ep_ranges)

        # episode split
        rng = np.random.default_rng(seed)
        ep_ids = np.arange(E)
        rng.shuffle(ep_ids)
        n_val = max(1, int(round(E * val_ratio)))
        val_eps = set(ep_ids[:n_val].tolist())
        train_eps = set(ep_ids[n_val:].tolist())

        # build window indices per split
        def make_indices_for(eps: set[int]) -> np.ndarray:
            idxs = []
            for eid, (s, e) in enumerate(ep_ranges):
                if eid not in eps:
                    continue
                i_min = s + obs_horizon - 1
                i_max = e - action_horizon
                if i_min < i_max:
                    idxs.extend(range(i_min, i_max))
            return np.array(idxs, dtype=np.int64)

        train_indices = make_indices_for(train_eps)
        val_indices = make_indices_for(val_eps)

        # compute stats on train subset only
        obs_mean, obs_std, act_mean, act_std, act_min, act_max = DiffusionDatasetBoth._compute_obs_action_stats_subset(
            obs, act, train_indices, obs_horizon, action_horizon
        )
        img_ext_mean, img_ext_std = DiffusionDatasetBoth._compute_image_stats_subset_chunked(
            img_ext, train_indices, obs_horizon, chunk=1024
        )
        img_wst_mean, img_wst_std = DiffusionDatasetBoth._compute_image_stats_subset_chunked(
            img_wst, train_indices, obs_horizon, chunk=1024
        )

        # create datasets sharing stats
        self.train_ds = DiffusionDatasetBoth(
            obs, act,
            img_ext=img_ext,
            img_wst=img_wst,
            episode_ranges=ep_ranges, indices=train_indices,
            obs_horizon=obs_horizon, action_horizon=action_horizon, pred_horizon=pred_horizon,
            obs_mean=obs_mean, obs_std=obs_std,
            action_mean=act_mean, action_std=act_std,
            action_min=act_min, action_max=act_max,
            img_ext_mean=img_ext_mean, img_ext_std=img_ext_std,
            img_wst_mean=img_wst_mean, img_wst_std=img_wst_std
        )

        self.val_ds = DiffusionDatasetBoth(
            obs, act,
            img_ext=img_ext,
            img_wst=img_wst,
            episode_ranges=ep_ranges, indices=val_indices,
            obs_horizon=obs_horizon, action_horizon=action_horizon, pred_horizon=pred_horizon,
            obs_mean=obs_mean, obs_std=obs_std,
            action_mean=act_mean, action_std=act_std,
            action_min=act_min, action_max=act_max,
            img_ext_mean=img_ext_mean, img_ext_std=img_ext_std,
            img_wst_mean=img_wst_mean, img_wst_std=img_wst_std
        )

        sample_obs, sample_ext_imgs, sample_wst_imgs, sample_actions = self.train_ds[0]

        self.obs_dim = sample_obs.shape[-1]
        self.action_dim = sample_actions.shape[-1]
        self.ext_img_dim = sample_ext_imgs.shape[1:]
        self.wst_img_dim = sample_wst_imgs.shape[1:]

    def train(self,
                num_epochs=100,
                batch_size=32,
                learning_rate=1e-4,
                unet_weight_decay= 1e-3,
                obs_encoder_weight_decay= 1e-6,
                betas= (0.9, 0.95),
                max_norm = 1.0,
                num_diffusion_steps=100,
                model_name=None,
                checkpoint_interval = 10):

        train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,) #pin_memory=True)
        val_loader   = DataLoader(self.val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,)# pin_memory=True)

        # Create directory for logging
        os.makedirs("asset/training", exist_ok=True)
        csv_path = os.path.join("asset/training/", f"DiT_w_stats_state_{model_name or 'model'}_loss_log.csv")

        # Initialize CSV with headers
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "action_loss"])

        # EMA for stable training
        ema = EMAModel(
            parameters=self.model.parameters(),
            power=0.75
        )

        optimizer = self.model.get_optimizer(
            learning_rate=learning_rate,
            unet_weight_decay=unet_weight_decay,
            obs_encoder_weight_decay=obs_encoder_weight_decay,
            betas=betas)

        # ============================================================
        # TODO: Create a cosine learning-rate schedule with warmup
        # ============================================================
        # Goal:
        #   - Start training with a small LR that ramps up for stability (warmup)
        #   - Then decay LR smoothly using cosine schedule over the rest of training
        #
        # Why warmup helps diffusion/DDPM training:
        #   - Early gradients can be noisy/large (esp. with UNet + image encoder)
        #   - Warmup reduces divergence and makes training more stable
        #
        # Required inputs:
        #   - optimizer: the optimizer you created (e.g., AdamW)
        #   - num_warmup_steps: how many update steps to linearly ramp LR from 0 → base LR
        #   - num_training_steps: total number of optimizer steps across all epochs
        #       typically = len(train_loader) * num_epochs
        #
        # Implementation:
        #   Use get_scheduler(...) with:
        #     name='cosine'
        #     optimizer=optimizer
        #     num_warmup_steps=...
        #     num_training_steps=...
        #
        # YOUR CODE HERE
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/optimization.py#L288
        num_training_steps = len(train_loader) * num_epochs
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps= num_training_steps // 10,
            num_training_steps=num_training_steps
        )

        # ============================================================
        # TODO: Initialize diffusion timestep schedule
        # ============================================================
        # Goal:
        #   Prepare the noise scheduler for inference (sampling).
        #
        # What this does:
        #   self.noise_scheduler.set_timesteps(num_diffusion_steps)
        #
        #   - Creates the discrete timestep sequence used during reverse diffusion.
        #   - Typically produces something like:
        #         [T-1, T-2, ..., 0]
        #     depending on the scheduler type (DDPM, DDIM, etc.).
        #
        # Why this is necessary:
        #   - During training, timesteps are sampled randomly.
        #   - During inference (sampling), we must define the *full trajectory*
        #     of timesteps used to iterativelSample diffusiony denoise.
        #
        # Important:
        #   - num_diffusion_steps determines sampling quality vs speed.
        #   - This must be called BEFORE the sampling loop.
        #   - Different schedulers (DDPM, DDIM) interpret timesteps differently.
        #
        # YOUR CODE HERE
        # https://huggingface.co/docs/diffusers/v0.26.2/api/schedulers/ddpm#diffusers.DDPMScheduler.set_timesteps

        self.noise_scheduler.set_timesteps(num_diffusion_steps, device=self.device)
        with tqdm(range(num_epochs), desc='Epoch', leave=True) as tglobal:
            for epoch_idx in tglobal:
                self.model.train()
                epoch_loss = []
                with tqdm(train_loader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # unpack tuple: (obs, img_ext, img_wst, action)
                        nobs, nimg_ext, nimg_wst, naction = nbatch

                        # move tensors to device
                        nobs      = nobs.to(self.device, non_blocking=True)
                        nimg_ext  = nimg_ext.to(self.device, non_blocking=True)
                        nimg_wst  = nimg_wst.to(self.device, non_blocking=True)
                        naction   = naction.to(self.device, non_blocking=True)

                        # ensure batch dims
                        if nobs.ndim == 2:
                            nobs = nobs.unsqueeze(0)
                        if nimg_ext.ndim == 4:
                            nimg_ext = nimg_ext.unsqueeze(0)
                        if nimg_wst.ndim == 4:
                            nimg_wst = nimg_wst.unsqueeze(0)
                        if naction.ndim == 2:
                            naction = naction.unsqueeze(0)

                        B = nobs.shape[0]

                        obs_cond     = nobs
                        img_ext_cond = nimg_ext
                        img_wst_cond = nimg_wst

                        # append observations
                        obs = {"obs": obs_cond}
                        if self.image_type == "external":
                            obs["img_ext"] = img_ext_cond
                        elif self.image_type == "wrist":
                            obs["img_wst"] = img_wst_cond
                        elif self.image_type == "both":
                            obs["img_ext"] = img_ext_cond
                            obs["img_wst"] = img_wst_cond

                            # ============================================================
                            # TODO: One diffusion training step (DDPM-style)
                            # ============================================================
                            # You will implement the standard diffusion training loop:
                            #   1) sample Gaussian noise ε
                            #   2) sample diffusion timestep t
                            #   3) create noisy input x_t = q(x_t | x_0) by adding noise to the clean action sequence
                            #   4) run the model to predict ε̂ from (x_t, t, conditioning)
                            #   5) compute MSE(ε̂, ε)
                            #   6) backprop + optimizer step + LR scheduler step
                            #   7) update EMA weights
                            #
                            # Notes on shapes:
                            #   naction:       [B, action_horizon, action_dim]  (clean x_0)
                            #   noise:         same shape as naction            (ε)
                            #   timesteps:     [B]                              (t per sample)
                            #   noisy_actions: same shape as naction            (x_t)
                            #   noise_pred:    same shape as naction            (ε̂)
                            #
                            # Gotchas:
                            #   - timesteps are sampled from [0, num_train_timesteps)
                            #   - add_noise expects (x_0, noise, timesteps) with timesteps shaped [B]
                            #   - ensure everything is on self.device
                            #   - call optimizer.zero_grad() each step (or before backward)
                            #   - if using AMP/GradScaler, these lines change
                            #   - lr_scheduler.step() is per optimizer step (not per epoch)
                            #
                            # YOUR CODE HERE
                            B = naction.shape[0]
                            noise = torch.randn(naction.shape).to(self.device)
                            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, size=(B,)).to(self.device).long()
                            x_t = self.noise_scheduler.add_noise(naction, noise, timesteps)
                            # print(x_t)    
                            ext = obs["img_ext"]
                            wst = obs["img_wst"]
                            noise_pred = self.model(x_t, timesteps, obs["obs"], ext, wst)
                            loss = nn.functional.mse_loss(noise_pred, noise)
                            # print(f"Loss = {loss}")
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step(), lr_scheduler.step()
                            ema.step(self.model.parameters())
                            # print("Updated parameters")

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(train_loss=loss_cpu)

                # ✅ Save checkpoint every epoch
                if model_name is not None and (epoch_idx + 1) % checkpoint_interval == 0:
                    self.save_model(f"{self.save_path}DiT_w_stats_state_{model_name}_epoch_{epoch_idx+1}.pth")
                    print(f"💾 Checkpoint saved at {self.save_path}DiT__w_stats_state_{model_name}_epoch_{epoch_idx+1}.pth")

                    # --- Validation loop with EMA ---
                    avg_val_loss = np.nan
                    if val_loader is not None:
                        self.model.eval()
                        val_loss = []
                        action_loss = []

                        # Use EMA weights for validation
                        ema.store(self.model.parameters())  # save current weights
                        ema.copy_to(self.model.parameters())  # load EMA weights

                        with torch.no_grad():
                            for nobs, nimg_ext, nimg_wst, naction in val_loader:
                                nobs, nimg_ext, nimg_wst, naction = nobs.to(self.device), nimg_ext.to(self.device), nimg_wst.to(self.device), naction.to(self.device)

                                B = nobs.shape[0]

                                # observation & image conditioning (keep obs_horizon)
                                obs_cond     = nobs[:, :train_loader.dataset.obs_horizon, :]
                                img_ext_cond = nimg_ext[:, :train_loader.dataset.obs_horizon, ...]
                                img_wst_cond = nimg_wst[:, :train_loader.dataset.obs_horizon, ...]

                                # append observations
                                obs = {"obs": obs_cond}
                                if self.image_type == "external":
                                    obs["img_ext"] = img_ext_cond
                                elif self.image_type == "wrist":
                                    obs["img_wst"] = img_wst_cond
                                elif self.image_type == "both":
                                    obs["img_ext"] = img_ext_cond
                                    obs["img_wst"] = img_wst_cond

                                # ============================================================
                                # TODO: Validation step for diffusion training (no grad)
                                # ============================================================
                                # Goal:
                                #   Estimate how well the model predicts the injected noise on a held-out set.
                                #   This mirrors the training objective, but:
                                #     - does NOT do backprop
                                #     - only logs scalar loss values
                                #
                                # Steps:
                                #   1) Sample Gaussian noise ε with the same shape as the clean actions x_0 (naction)
                                #   2) Sample diffusion timesteps t for each batch item (shape [B])
                                #      from [0, num_train_timesteps)
                                #   3) Forward diffusion: create noisy_actions x_t by adding noise to naction
                                #      using the scheduler (q(x_t | x_0))
                                #   4) Run the model to predict ε̂ = model(x_t, t, conditioning)
                                #   5) Compute MSE(ε̂, ε) as a scalar
                                #   6) Append the scalar to val_loss list
                                #   7) Compute running average avg_val_loss = mean(val_loss)
                                #
                                # Shapes:
                                #   naction:       [B, action_horizon, action_dim]
                                #   noise:         same as naction
                                #   timesteps:     [B]
                                #   noisy_actions: same as naction
                                #   noise_pred:    same as naction
                                #
                                # Important:
                                #   - This block should run under torch.no_grad()
                                #   - Set model to eval() before validation and restore train() after
                                #   - Use .item() to store python floats (not tensors) in val_loss
                                #
                                # YOUR CODE HERE
                                with torch.no_grad():
                                    noise = torch.randn_like(naction).to(self.device, non_blocking=True)
                                    t = torch.randint(0, self.noise_scheduler.num_train_timesteps, size=(B,)).to(self.device).long()

                                    noisy_actions = self.noise_scheduler.add_noise(naction, noise, t)
                                    if "img_ext" not in obs: obs["img_ext"] = None
                                    if "wst_ext" not in obs: obs["wst_ext"] = None
                                    pred = self.model(noisy_actions, t, obs["obs"], obs["img_ext"], obs["img_wst"])

                                    loss = nn.functional.mse_loss(pred, noise).item()
                                    val_loss.append(loss)
                                    avg_val_loss = np.mean(val_loss)

                                # ============================================================
                                # TODO: Diffusion sampling loop (reverse process) + action-space evaluation
                                # ============================================================
                                # Goal:
                                #   Generate an action sequence by running the *reverse diffusion* process
                                #   conditioned on the observation, then compare the generated action to
                                #   the ground-truth action sequence naction.
                                #
                                # High-level steps:
                                #   1) Initialize the sample with pure Gaussian noise:
                                #        rand_actions ~ N(0, I)
                                #      Shape should match the action sequence you want to generate:
                                #        [B, pred_horizon, action_dim]
                                #
                                #   2) Initialize the scheduler's inference timesteps:
                                #        noise_scheduler.set_timesteps(num_diffusion_steps)
                                #      This creates noise_scheduler.timesteps, the ordered list of timesteps
                                #      used for reverse diffusion (usually decreasing from T-1 to 0).
                                #
                                #   3) For each timestep k in noise_scheduler.timesteps:
                                #        a) Predict the noise (or velocity) at that timestep:
                                #             noise_pred = model(x_k, k, conditioning)
                                #        b) Take one reverse diffusion step to get x_{k-1}:
                                #             x_{k-1} = scheduler.step(model_output=noise_pred, timestep=k, sample=x_k).prev_sample
                                #      Update rand_actions each iteration.
                                #
                                #   4) After the loop ends, rand_actions is the final denoised sample (x_0 estimate).
                                #
                                #   5) Compute supervised action reconstruction loss:
                                #        MSE(rand_actions, naction)
                                #      Append scalar to action_loss list and compute running mean.
                                #
                                # Shapes:
                                #   rand_actions:  [B, pred_horizon, action_dim]   (current sample x_k)
                                #   k:            scalar timestep (int or 0-d tensor depending on scheduler)
                                #   noise_pred:   same shape as rand_actions
                                #   naction:      [B, pred_horizon, action_dim]   (ground-truth action sequence)
                                #
                                # Important gotchas:
                                #   - set_timesteps() is for inference/sampling; num_diffusion_steps here controls speed/quality
                                #   - Many schedulers expect timesteps passed as python int or tensor on the right device
                                #   - This loop should usually run under torch.no_grad() and model.eval()
                                #   - Ensure conditioning tensors (obs/images) are on the correct device and preprocessed
                                #   - If your model expects timesteps shape [B], you may need to expand k to [B]
                                #
                                # YOUR CODE HERE
                                # https://huggingface.co/docs/diffusers/v0.26.2/api/schedulers/ddpm#diffusers.DDPMScheduler.step
                                with torch.no_grad():
                                    rand_actions = torch.randn_like(naction).to(self.device, non_blocking=True)
                                    self.noise_scheduler.set_timesteps(num_diffusion_steps, device=self.device)

                                    for k in self.noise_scheduler.timesteps:
                                        expanded_k = torch.full((naction.shape[0],), k).to(self.device)
                                        noise_pred = self.model(rand_actions, expanded_k, obs["obs"], obs["img_ext"], obs["img_wst"])
                                        rand_actions = self.noise_scheduler.step(noise_pred, k, rand_actions).prev_sample

                                    loss = nn.functional.mse_loss(rand_actions, naction).item()
                                    action_loss.append(loss)
                                    avg_action_loss = np.mean(action_loss)

                        ema.restore(self.model.parameters())  # restore original weights

                    tqdm.write(f"Train loss: {np.mean(epoch_loss):.4f}, "
                    f"Val loss: {avg_val_loss:.4f}, "
                    f"Action loss: {avg_action_loss:.4f}")
                    tglobal.set_postfix(train_loss=np.mean(epoch_loss), val_loss=avg_val_loss, action_loss = avg_action_loss)

                    with open(csv_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch_idx + 1, np.mean(epoch_loss), avg_val_loss, avg_action_loss])

        # copy EMA weights into model for inference
        ema_unet = self.model
        ema.copy_to(ema_unet.parameters())
        self.model = ema_unet
        print("✅ Training complete — EMA weights loaded for inference.")

        # save model checkpoint if save_path is provided
        if model_name is not None:
            self.save_model(f"{self.save_path}DiT_w_stats_state_{model_name}_final.pth")
            print(f"💾 Model saved to {self.save_path}DiT_w_stats_state_{model_name}_final.pth")

    def save_model(self, path: str):

        # get directory part of the path
        dir_name = os.path.dirname(path)

        # create directory if it doesn't exist
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        train_stats = self.train_ds

        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "meta": {
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "pred_horizon": self.pred_horizon,
                    "obs_horizon": self.obs_horizon,
                    "num_diffusion_steps": self.num_diffusion_steps,
                },
                "stats": {
                    "img_mean": train_stats.img_ext_mean,
                    "img_std": train_stats.img_ext_std,
                    "wimg_mean": train_stats.img_wst_std,
                    "wimg_std": train_stats.img_wst_std,
                    "s_mean": train_stats.obs_mean,
                    "s_std": train_stats.obs_std,
                    "a_mean": train_stats.action_mean,
                    "a_std": train_stats.action_std,
                }
            },
            path,
        )
        logging.info(f"Saved checkpoint to {path}")

    def load_model(self, model_name: str):
        ckpt = torch.load(f"{self.save_path}DiT_state_{model_name}.pth", map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        m = ckpt.get("meta", {})
        logging.info(f"Loaded checkpoint from {self.save_path}DiT_state_{model_name}.pth (meta: {m})")

def main():

    parser = argparse.ArgumentParser(description="Train or test diffusion policy")
    parser.add_argument("--mode", type=str, choices=["train", "inf", "visual"], help="running mode", default="train")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument("--scheduler", type=str, choices=["squaredcos_cap_v2","linear","scaled_linear"])

    cli_args = parser.parse_args()

    # Load YAML config
    with open(cli_args.config, "r") as f:
        args = yaml.safe_load(f)
    trainer = DiffusionPolicyTrainer(
            task_name=args["task_name"] ,
            data_dir=args["data_dir"],
            pred_horizon=args["pred_horizon"],
            obs_horizon=args["obs_horizon"],
            num_diffusion_steps=args["num_diffusion_steps"],
            image_type=args["image_type"],
            eta=args["eta"],
            schedule_type=cli_args.scheduler
        )


    print(f"Train samples: {len(trainer.train_ds)}, Validation samples: {len(trainer.val_ds)}")
    sample_obs, sample_ext_imgs, sample_wst_imgs, sample_actions = trainer.train_ds[0]  # (H_o, D_o), (H_a, D_a)

    obs_dim = sample_obs.shape[1]
    action_dim = sample_actions.shape[1]
    logger.info(f"Inferred obs_dim={obs_dim}, action_dim={action_dim}")

    if cli_args.mode == "train":

        # logger.info("=== TRAINING MODE ===")
        save_dir = os.path.dirname(args["save_model_name"])
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        losses = trainer.train(
            num_epochs=args["epochs"],
            batch_size=args["batch_size"],
            learning_rate=args["lr"],
            unet_weight_decay=args["unet_weight_decay"],
            obs_encoder_weight_decay=args["obs_encoder_weight_decay"],
            betas=args["betas"],
            num_diffusion_steps=args["num_diffusion_steps"],
            model_name=args["save_model_name"],
            checkpoint_interval = args["checkpoint_interval"]
        )

        logger.info(f"Final model saved at {args['save_model_name']}")

    elif cli_args.mode == "inf":

        logger.info("=== TESTING MODE ===")

        trainer.load_model(args["load_model_name"])
        trainer.model.eval()

        # init scheduler
        trainer.ddim_scheduler.set_timesteps(args["num_inference_steps"])
        vis = args.get("denoise_vis", {})
        vis_enabled = bool(vis.get("enabled", False))

        while True:
            # --------------------------------------------------
            # 1. Sample one datapoint
            # --------------------------------------------------
            idx = np.random.randint(len(trainer.train_ds))
            obs, img_ext, img_wst, act_gt = trainer.train_ds[idx]
            obs_np, img_ext_np, img_wst_np, act_gt_np = trainer.train_ds[idx]

            device = trainer.device
            N = 1

            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            img_ext = torch.as_tensor(img_ext_np, dtype=torch.float32, device=device).unsqueeze(0)
            img_wst = torch.as_tensor(img_wst_np, dtype=torch.float32, device=device).unsqueeze(0)
            act_gt = torch.as_tensor(act_gt_np, dtype=torch.float32, device=device)  # (H,D)

            # --------------------------------------------------
            # 2. Initialize noisy action
            # --------------------------------------------------
            naction = torch.randn(
                (N, args["pred_horizon"], trainer.action_dim),
                device=device,
            )

            # --------------------------------------------------
            # 3. DDIM sampling loop
            # --------------------------------------------------\
            denoise_actions = []
            if vis_enabled:
                denoise_actions.append(naction[0].detach().cpu())  # initial noisy x0

            with torch.no_grad():
                for t in trainer.ddim_scheduler.timesteps:
                    noise_pred = trainer.model(
                        noisy_actions=naction,
                        timesteps=t,
                        observations=obs,
                        img_ext=img_ext,
                        img_wst=img_wst,
                    )
                    naction = trainer.ddim_scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=naction,
                    ).prev_sample

                    if vis_enabled:
                        denoise_actions.append(naction[0].detach().cpu())

            # --------------------------------------------------
            # 4. Loss (FIXED SHAPE)
            # --------------------------------------------------
            loss = nn.functional.mse_loss(naction[0], act_gt)

            # --------------------------------------------------
            # 5. Debug print
            # --------------------------------------------------
            torch.set_printoptions(
                sci_mode=False,
                precision=6,
                linewidth=200,
            )

            # --------------------------------------------------
            # 6. Save visuals
            # --------------------------------------------------
            if vis_enabled:
                out_dir = vis.get("out_dir", "asset/vis")
                os.makedirs(out_dir, exist_ok=True)

                fps = int(vis.get("fps", 6))
                subsample_frames = int(vis.get("subsample_frames", 0))
                overlay_gt = bool(vis.get("overlay_gt", True))

                obs_q_idx = tuple(vis.get("obs_q_idx", [0,1,2,3,4,5,6]))
                action_mode = vis.get("action_mode", "delta")
                action_q_idx = vis.get("action_q_idx", None)
                fk_which = vis.get("fk_which", "modified")

                # denoise trajectories (each is (H+1,3) EE xyz)
                trajs = denoise_actions_to_ee_trajs(
                    denoise_actions,
                    obs_np,
                    obs_q_idx=obs_q_idx,
                    action_mode=action_mode,
                    action_q_idx=action_q_idx,
                    fk_which=fk_which,
                )

                # optional GT overlay: convert gt actions -> gt EE xyz trajectory too
                gt_traj = None
                if overlay_gt:
                    # build q_seq from gt actions using same interpretation
                    # (uses q0 from obs and action_mode/action_q_idx)
                    from scripts.vis_utils import get_start_q_from_obs
                    q0 = get_start_q_from_obs(obs_np, obs_q_idx=obs_q_idx, use_last=True)
                    q_seq_gt = joint_actions_to_q_seq(act_gt, q0=q0, action_mode=action_mode, action_q_idx=action_q_idx)
                    gt_traj = q_seq_to_ee_traj_xyz(q_seq_gt, which=fk_which)

                if subsample_frames and subsample_frames > 1:
                    trajs = subsample_list(trajs, subsample_frames)

                gif_path = os.path.join(out_dir, f"ddim_denoise_ee_idx{idx}.gif")
                render_denoise_gif_3d(trajs, out_path=gif_path, gt_traj=gt_traj, fps=fps)

                if bool(vis.get("save_once", True)):
                    print("[vis] save_once=True -> exiting after one visualization.")
                    return

            print("ground truth:\n", act_gt)
            print("predicted action:\n", naction[0])
            print("MSE loss:", loss.item())

    if cli_args.mode == "visual":

        logger.info("=== VISUALIZATION MODE ===")

        trainer.load_model(args["load_model_name"])
        trainer.model.eval()

        device = trainer.device

        # --------------------------------------------------
        # 1. Sample one datapoint
        # --------------------------------------------------
        idx = np.random.randint(len(trainer.train_ds))
        obs, img_ext, img_wst, act = trainer.train_ds[idx]

        act = torch.as_tensor(act, dtype=torch.float32, device=device)  # (H, D)

        H, D = act.shape

        # --------------------------------------------------
        # 2. Pick timesteps to visualize
        # --------------------------------------------------
        num_vis = 10
        timesteps = torch.linspace(
            0,
            trainer.noise_scheduler.config.num_train_timesteps - 1,
            num_vis,
            dtype=torch.long,
        )

        noise = torch.randn_like(act)

        # --------------------------------------------------
        # 3. Generate noisy versions
        # --------------------------------------------------
        noisy_actions = []

        for t in timesteps:
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)

            # ============================================================
            # TODO: Apply forward diffusion (add noise to clean action)
            # ============================================================
            # Goal:
            #   Generate x_t from clean action x_0 using the forward diffusion process:
            #
            #       x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * ε
            #
            #   Instead of implementing the formula manually, use:
            #       noise_scheduler.add_noise(...)
            #
            # Inputs:
            #   act:    [H, D]             clean action sequence (x_0)
            #   noise:  [H, D]             sampled Gaussian noise (ε)
            #   t_batch: [1] or [B]        diffusion timestep(s)
            #
            # Why unsqueeze(0)?
            #   The scheduler expects batched input:
            #       [B, H, D]
            #   So we add a batch dimension → (1, H, D)
            #
            # Output:
            #   noisy: [1, H, D]   noisy action sequence (x_t)
            #
            # Important:
            #   - All tensors must be on the same device
            #   - t_batch must match batch size (here B=1)
            #   - add_noise uses the scheduler’s internal alpha schedule
            #
            # YOUR CODE HERE
            # https://huggingface.co/docs/diffusers/v0.26.2/en/api/schedulers/ddpm#diffusers.DDPMScheduler.add_noise
            noisy = trainer.noise_scheduler.add_noise(act.unsqueeze(0), noise.unsqueeze(0), t_batch)

            
            noisy_actions.append(noisy[0].detach().cpu())

        # --------------------------------------------------
        # 4. Plot
        # --------------------------------------------------
        fig, axes = plt.subplots(
            D, num_vis,
            figsize=(3 * num_vis, 2.5 * D),
            sharex=True,
            sharey="row"
        )

        if D == 1:
            axes = axes[None, :]

        for j, t in enumerate(timesteps):
            for d in range(D):
                axes[d, j].plot(act[:, d].cpu(), label="clean", linewidth=2)
                axes[d, j].plot(noisy_actions[j][:, d], linestyle="--", label="noisy")

                if d == 0:
                    axes[d, j].set_title(f"t = {int(t.item())}")

                if j == 0:
                    axes[d, j].set_ylabel(f"action[{d}]")

        axes[0, 0].legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    main()