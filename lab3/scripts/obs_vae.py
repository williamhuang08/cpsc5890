import os, numpy as np, torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from robomimic.models.obs_nets import ObservationEncoder
from robomimic.utils.tensor_utils import time_distributed

from scripts.dataset import make_loaders


# -------------------------
# Small blocks
# -------------------------
class Conv1dBlock(nn.Module):
    def __init__(self, inp, out, k=3, n_groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(inp, out, k, padding=k//2),
            nn.GroupNorm(n_groups, out),
            nn.Mish()
        )
    def forward(self, x): return self.net(x)


def swap_bn_to_gn(module: nn.Module, max_groups: int = 32):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            C = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups=min(max_groups, C), num_channels=C))
        else:
            swap_bn_to_gn(child, max_groups)
    return module

class ObservationVAE(nn.Module):
    """
    VAE for observation sequences (state + encoded images).
    Inputs:
        obs_state: (B, H, obs_dim)
        obs_image: (B, H, C, Himg, Wimg)
        obs_wrist_image: (B, H, C, Himg, Wimg)
    Outputs:
        z: (B, H, z_dim)
        obs_hat: (B, H, obs_dim + 2*img_feat_dim)
    """
    def __init__(self, obs_dim=8, z_dim=16, hidden=256,
                 img_feat_dim=64, image_type="both", obs_encoder=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.image_type = image_type
        self.img_feat_dim = img_feat_dim

        # Use provided encoder or build one
        if obs_encoder is None:
            self.obs_encoder = ObservationEncoder(feature_activation=nn.ReLU)
        else:
            self.obs_encoder = obs_encoder

        # Input dim: state + image features
        input_dim = obs_dim
        if image_type == "both":
            input_dim += 2 * img_feat_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

        # Decoder: reconstruct state + image features
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def encode_obs(self, obs_state, obs_image=None, obs_wrist_image=None):
        """
        Encode obs_state + images into feature vector for VAE.
        """
        feats = [obs_state]  # (B,H,obs_dim)

        if self.image_type == "both" and obs_image is not None and obs_wrist_image is not None:
            ext_feat = time_distributed(obs_image, self.obs_encoder.obs_nets["external"], inputs_as_kwargs=False)
            wst_feat = time_distributed(obs_wrist_image, self.obs_encoder.obs_nets["wrist"], inputs_as_kwargs=False)
            feats += [ext_feat, wst_feat]

        x = torch.cat(feats, dim=-1)  # (B,H,input_dim)
        return x

    def encode(self, obs_state, obs_image=None, obs_wrist_image=None):
        x = self.encode_obs(obs_state, obs_image, obs_wrist_image)
        B, H, D = x.shape
        h = self.enc(x.reshape(B*H, D))
        mu = self.mu(h).reshape(B,H,self.z_dim)
        logvar = self.logvar(h).reshape(B,H,self.z_dim)
        return mu, logvar

    @staticmethod
    def reparam(mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5*logvar)

    def decode(self, z):
        B, H, Z = z.shape
        obs_hat = self.dec(z.reshape(B*H, Z)).reshape(B, H, self.obs_dim + 2*self.img_feat_dim)
        return obs_hat

    def forward(self, obs_state, obs_image=None, obs_wrist_image=None):
        """
        Forward pass of the ObservationVAE.

        Inputs:
            obs_state: (B, H, obs_dim)
            obs_image: (B, H, C, Himg, Wimg) or None
            obs_wrist_image: (B, H, C, Himg, Wimg) or None

        Outputs:
            z: latent embedding (B, H, z_dim)
            obs_hat: reconstructed obs + image features (B, H, obs_dim + 2*img_feat_dim)
            mu: mean of latent distribution
            logvar: log-variance of latent distribution
        """

        # ---- Step 1: Encode observations into latent parameters ----
        # TODO: call self.encode() with obs_state + images to get mu and logvar
        mu, logvar = self.encode(obs_state, obs_image=obs_image, obs_wrist_image=obs_wrist_image)

        # ---- Step 2: Reparameterization trick ----
        # TODO: sample z ~ N(mu, sigma^2) using self.reparam()
        z = self.reparam(mu, logvar)

        # ---- Step 3: Decode latent z to reconstruct observation ----
        # TODO: call self.decode(z) to get obs_hat
        obs_hat = self.decode(z)

        # ---- Step 4: Return latent + reconstruction ----
        return z, obs_hat, mu, logvar

def train_obs_vae(obs_vae, train_loader, test_loader, device="cuda",
                  lr=1e-3, wd=1e-6, epochs=20, beta=1e-3):
    opt = torch.optim.AdamW(obs_vae.parameters(), lr=lr, weight_decay=wd)

    for ep in range(1, epochs+1):
        obs_vae.train()
        tr_loss = 0.0; n=0
        for b in tqdm(train_loader, desc=f"ObsVAE train {ep}/{epochs}"):
            obs_state = b["obs_state"].to(device)
            obs_image = b.get("obs_image", None)
            obs_wrist_image = b.get("obs_wrist_image", None)
            if obs_image is not None: obs_image = obs_image.to(device)
            if obs_wrist_image is not None: obs_wrist_image = obs_wrist_image.to(device)

            _, obs_hat, mu, logvar = obs_vae(obs_state, obs_image, obs_wrist_image)
            
            # Reconstruct state + image features
            feats = [obs_state]
            if obs_vae.image_type == "both" and obs_image is not None and obs_wrist_image is not None:
                ext_feat = time_distributed(obs_image, obs_vae.obs_encoder.obs_nets["external"], inputs_as_kwargs=False)
                wst_feat = time_distributed(obs_wrist_image, obs_vae.obs_encoder.obs_nets["wrist"], inputs_as_kwargs=False)
                feats += [ext_feat, wst_feat]
            target = torch.cat(feats, dim=-1)

            recon = nn.functional.mse_loss(obs_hat, target)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = recon + beta*kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += loss.item() * obs_state.size(0)
            n += obs_state.size(0)

        print(f"[ObsVAE {ep:03d}] train_loss={tr_loss/n:.6f}")


# -------------------------
# 2) Policy that predicts latent actions
# -------------------------
class BCConvMLPPolicyLatent(nn.Module):
    """
    Same as your BCConvMLPPolicy, but predicts z instead of actions.

    Output:
      pred_z: (B, Hpred, z_dim)
    """
    def __init__(
        self,
        z_dim=16,
        action_dim=8,        # only used for bookkeeping / sanity
        obs_dim=8,
        obs_horizon=2,
        pred_horizon=16,
        img_backbone_kwargs=None,
        image_type="both",   # "both" | "none"
        img_feat_dim=64,
        conv_channels=256,
        conv_layers=2,
        kernel_size=3,
        mlp_hidden=512,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.image_type = image_type

        if img_backbone_kwargs is None:
            img_backbone_kwargs = {
                "input_shape": [3, 96, 96],
                "backbone_class": "ResNet18Conv",
                "backbone_kwargs": {"pretrained": False, "input_coord_conv": False},
                "pool_class": "SpatialSoftmax",
                "pool_kwargs": {"num_kp": 32},
                "feature_dimension": img_feat_dim,
            }
        img_feat_dim = img_backbone_kwargs["feature_dimension"]

        self.obs_encoder = ObservationEncoder(feature_activation=nn.ReLU)
        if image_type == "both":
            self.obs_encoder.register_obs_key("external", img_backbone_kwargs["input_shape"],
                                              net_class="VisualCore", net_kwargs=img_backbone_kwargs)
            self.obs_encoder.register_obs_key("wrist", img_backbone_kwargs["input_shape"],
                                              net_class="VisualCore", net_kwargs=img_backbone_kwargs)
        self.obs_encoder.make()

        if image_type == "both":
            swap_bn_to_gn(self.obs_encoder.obs_nets["external"])
            swap_bn_to_gn(self.obs_encoder.obs_nets["wrist"])

        per_t = obs_dim + (2 * img_feat_dim if image_type == "both" else 0)

        layers = [Conv1dBlock(per_t, conv_channels, k=kernel_size)]
        for _ in range(conv_layers - 1):
            layers.append(Conv1dBlock(conv_channels, conv_channels, k=kernel_size))
        self.temporal = nn.Sequential(*layers)

        # head predicts latent sequence
        self.head = nn.Sequential(
            nn.Linear(conv_channels, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, pred_horizon * z_dim),
        )

    def forward(self, obs_state, obs_image=None, obs_wrist_image=None):
        feats = [obs_state]  # (B,Hobs,obs_dim)

        if self.image_type == "both":
            ext = time_distributed(obs_image, self.obs_encoder.obs_nets["external"], inputs_as_kwargs=False)
            wst = time_distributed(obs_wrist_image, self.obs_encoder.obs_nets["wrist"], inputs_as_kwargs=False)
            feats += [ext, wst]  # (B,Hobs,img_feat_dim)

        x = torch.cat(feats, dim=-1)      # (B,Hobs,per_t)
        x = x.transpose(1, 2)             # (B,per_t,Hobs)

        x = self.temporal(x)              # (B,conv_channels,Hobs)
        x = x.mean(dim=-1)                # (B,conv_channels)

        y = self.head(x)                  # (B,Hpred*z_dim)
        return y.view(x.shape[0], self.pred_horizon, self.z_dim)

def train_bc_to_latent(policy, obs_vae, train_loader, test_loader, device="cuda", lr=1e-4, wd=1e-6, epochs=30):
    """
    Freeze ObservationVAE. Train policy to predict z = encode(obs_state + images).
    """
    obs_vae.eval()
    for p in obs_vae.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        policy.train()
        tr = 0.0; n = 0

        for b in tqdm(train_loader, desc=f"BC->latent train {ep}/{epochs}"):
            obs_state = b["obs_state"].to(device)
            obs_img = b.get("obs_image", None)
            obs_wimg = b.get("obs_wrist_image", None)
            if obs_img is not None: obs_img = obs_img.to(device)
            if obs_wimg is not None: obs_wimg = obs_wimg.to(device)

            # ---- Step X: Compute target latent embedding ----
            # TODO: Use the frozen ObservationVAE to encode obs_state + images
            #       and get mu (mean of latent distribution) as the target z_tgt
            z_tgt = obs_vae.encode(obs_state, obs_image=obs_image, obs_wrist_image=obs_wimg)  # replace None with obs_vae.encode(...)

            # ---- Step Y: Predict latent embedding from BC policy ----
            # TODO: Feed obs_state + images to your policy to get z_pred
            z_pred = policy(obs_state, obs_img, obs_wimg)  # replace None with policy(...)

            # ---- Step Z: Compute MSE loss between predicted and target latent ----
            # TODO: Use nn.MSELoss or functional.mse_loss to compute loss between z_pred and z_tgt
            loss = nn.MSELoss(z_pred, z_tgt)  # replace None with computed loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr += loss.item() * obs_state.size(0)
            n += obs_state.size(0)

        tr /= max(n, 1)

        policy.eval()
        te = 0.0; n = 0
        with torch.no_grad():
            for b in test_loader:
                obs_state = b["obs_state"].to(device)
                obs_img = b.get("obs_image", None)
                obs_wimg = b.get("obs_wrist_image", None)
                if obs_img is not None: obs_img = obs_img.to(device)
                if obs_wimg is not None: obs_wimg = obs_wimg.to(device)

                # ---- Step X: Compute target latent embedding ----
                # TODO: Use the frozen ObservationVAE to encode obs_state + images
                #       and get mu (mean of latent distribution) as the target z_tgt
                z_tgt = obs_vae.encode(obs_state, obs_image=obs_image, obs_wrist_image=obs_wimg)  # replace None with obs_vae.encode(...)

                # ---- Step Y: Predict latent embedding from BC policy ----
                # TODO: Feed obs_state + images to your policy to get z_pred
                z_pred = policy(obs_state, obs_img, obs_wimg)  # replace None with policy(...)

                # ---- Step Z: Compute MSE loss between predicted and target latent ----
                # TODO: Use nn.MSELoss or functional.mse_loss to compute loss between z_pred and z_tgt
                loss = nn.MSELoss(z_pred, z_tgt)  # replace None with computed loss

                te += loss.item() * obs_state.size(0)
                n += obs_state.size(0)

        te /= max(n, 1)
        print(f"[BC->latent {ep:03d}] train_mse={tr:.6f}  test_mse={te:.6f}")


# -------------------------
# Saving / loading both
# -------------------------
def save_latent_policy(path, policy, action_ae, stats, policy_kwargs, action_ae_kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "action_ae_state_dict": action_ae.state_dict(),
        "stats": stats,
        "policy_kwargs": policy_kwargs,
        "action_ae_kwargs": action_ae_kwargs,
    }, path)

def main():

    p = argparse.ArgumentParser("Latent action BC (AE/VAE) runner")
    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["vae", "bc", "inference"],
        help="train_vae: train action encoder | train_bc: train BC to predict latent | inference: run forward demo",
    )

    p.add_argument("--data_dir", type=str, default="/home/robot-lab/Downloads/xarm_lift_data")
    p.add_argument("--obs_h", type=int, default=2)
    p.add_argument("--pred_h", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--include_images", action="store_true", default=True)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--z_dim", type=int, default=6)
    p.add_argument("--action_dim", type=int, default=8)
    p.add_argument("--obs_dim", type=int, default=8)
    p.add_argument("--image_type", type=str, default="both", choices=["both", "none"])

    p.add_argument("--ae_hidden", type=int, default=256)
    p.add_argument("--ae_epochs", type=int, default=5)
    p.add_argument("--bc_epochs", type=int, default=2)

    p.add_argument("--vae_beta", type=float, default=1e-3)
    p.add_argument("--vae_lr", type=float, default=1e-3)
    p.add_argument("--vae_wd", type=float, default=1e-6)

    p.add_argument("--ckpt_path", type=str, default="asset/checkpoints/bcconv_latent_final.pt")

    args = p.parse_args()

    # -----------------------------
    # data + device
    # -----------------------------
    train_loader, test_loader, stats = make_loaders(
        args.data_dir,
        obs_h=args.obs_h,
        pred_h=args.pred_h,
        batch_size=args.batch_size,
        include_images=args.include_images,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # build action encoder (AE or VAE)
    # -----------------------------
    action_ae_kwargs = dict(action_dim=args.action_dim, z_dim=args.z_dim, hidden=args.ae_hidden)
    action_ae = ObservationVAE(**action_ae_kwargs).to(device)

    # -----------------------------
    # mode dispatch
    # -----------------------------
    if args.mode == "vae":
        train_obs_vae(obs_vae, train_loader, test_loader,
                    device=device,
                    lr=args.vae_lr,
                    wd=args.vae_wd,
                    epochs=args.ae_epochs,
                    beta=args.vae_beta)

    elif args.mode == "bc":
        # Load VAE checkpoint first
        ckpt = torch.load(args.ckpt_path, map_location=device) if os.path.exists(args.ckpt_path) else None
        if ckpt:
            obs_vae.load_state_dict(ckpt["action_ae_state_dict"])
            obs_vae.eval()
            stats = ckpt.get("stats", stats)

        policy_kwargs = dict(
            z_dim=args.z_dim,
            obs_dim=args.obs_dim,
            obs_horizon=args.obs_h,
            pred_horizon=args.pred_h,
            image_type=args.image_type,
        )
        policy = BCConvMLPPolicyLatent(**policy_kwargs).to(device)

        train_bc_to_latent(policy, obs_vae, train_loader, test_loader, device=device, epochs=args.bc_epochs)


    elif args.mode == "inference":
        if not os.path.exists(args.ckpt_path):
            raise RuntimeError(f"Checkpoint not found: {args.ckpt_path}")

        ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("action_ae_state_dict", None) is None or ckpt.get("policy_state_dict", None) is None:
            raise RuntimeError("Need combined checkpoint with BOTH action_ae_state_dict and policy_state_dict. Run --mode train_bc first.")

        # rebuild + load
        action_ae = ObservationVAE(**ckpt["action_ae_kwargs"]).to(device)
        action_ae.load_state_dict(ckpt["action_ae_state_dict"])
        action_ae.eval()

        policy = BCConvMLPPolicyLatent(**ckpt["policy_kwargs"]).to(device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()

        stats = ckpt["stats"]

        # run one batch
        b = next(iter(test_loader))
        obs_state = b["obs_state"].to(device)
        obs_img = b.get("obs_image", None)
        obs_wimg = b.get("obs_wrist_image", None)
        if obs_img is not None:
            obs_img = obs_img.to(device)
        if obs_wimg is not None:
            obs_wimg = obs_wimg.to(device)

        with torch.no_grad():
            z_hat = policy(obs_state, obs_img, obs_wimg)   # (B,Hpred,Z)
            a_hat = action_ae.decode(z_hat)                # (B,Hpred,A)

        print("Inference OK")
        print("  z_hat:", tuple(z_hat.shape))
        print("  a_hat:", tuple(a_hat.shape))

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()