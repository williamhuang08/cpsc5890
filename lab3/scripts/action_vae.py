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

class ActionVAE(nn.Module):
    """
    Per-timestep VAE for action sequences.

    Input:
      a: (B, Hpred, action_dim)

    Output:
      z:      (B, Hpred, z_dim)    sampled latent
      a_hat:  (B, Hpred, action_dim)
      mu:     (B, Hpred, z_dim)
      logvar: (B, Hpred, z_dim)
    """
    def __init__(self, action_dim=8, z_dim=16, hidden=256):
        super().__init__()
        self.action_dim = action_dim
        self.z_dim = z_dim

        self.enc = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def encode(self, a):
        """
        TODO:
        - flatten (B, H, A) -> (B*H, A)
        - run encoder
        - produce mu and logvar
        - reshape back to (B, H, z_dim)
        """
        B, H, A = a.shape
        enc_a = self.enc(a.reshape(B*H, A))
        mu = self.mu(enc_a).reshape(B, H, self.z_dim)
        logvar = self.logvar(enc_a).reshape(B, H, self.z_dim)
        return mu, logvar

    @staticmethod
    def reparam(mu, logvar):
        """
        TODO:
        - sample epsilon ~ N(0, I)
        - return reparameterized z
        """
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5*logvar)

    def decode(self, z):
        """
        TODO:
        - flatten (B, H, Z)
        - run decoder
        - reshape back to (B, H, action_dim)
        """
        B, H, Z = z.shape
        a_hat = self.dec(z.reshape(B*H, Z)).reshape(B, H, self.action_dim)
        return a_hat

    def forward(self, a):
        mu, logvar = self.encode(a)
        z = self.reparam(mu, logvar)
        a_hat = self.decode(z)
        return z, a_hat, mu, logvar


def train_action_vae(action_vae, train_loader, test_loader, device="cuda",
                     lr=1e-3, wd=1e-6, epochs=20, beta=1e-3):
    """
    Loss = recon + beta * KL
    recon: MSE(a_hat, a)
    KL:    KL(q(z|a) || N(0, I))
    """
    opt = torch.optim.AdamW(action_vae.parameters(), lr=lr, weight_decay=wd)

    for ep in range(1, epochs + 1):
        action_vae.train()
        trL = trR = trK = 0.0
        n = 0

        for b in tqdm(train_loader, desc=f"VAE train {ep}/{epochs}"):
            a = b["pred_action"].to(device)

            _, a_hat, mu, logvar = action_vae(a)
            
            """
            TODO:
            - compute reconstruction loss (MSE)
            - compute KL divergence to N(0, I)
            - combine into total loss
            """
            # recon = nn.MSELoss(a, a_hat)
            # kl = torch.functional.kl_div(
            #     torch.distributions.normal(mean=mu, variance=torch.exp(logvar)),
            #     torch.distributions.normal(mean=0, variance=1)
            # )

            recon = nn.functional.mse_loss(a_hat, a)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

            loss = recon + beta * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = a.size(0)
            trL += loss.item() * bs
            trR += recon.item() * bs
            trK += kl.item() * bs
            n += bs

        trL /= max(n, 1); trR /= max(n, 1); trK /= max(n, 1)

        action_vae.eval()
        teL = teR = teK = 0.0
        n = 0
        with torch.no_grad():
            for b in test_loader:
                a = b["pred_action"].to(device)
                _, a_hat, mu, logvar = action_vae(a)
                
                """
                TODO:
                - compute reconstruction loss (MSE)
                - compute KL divergence to N(0, I)
                - combine into total loss
                """
                recon = nn.functional.mse_loss(a_hat, a)
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

                loss = recon + beta * kl

                bs = a.size(0)
                teL += loss.item() * bs
                teR += recon.item() * bs
                teK += kl.item() * bs
                n += bs

        teL /= max(n, 1); teR /= max(n, 1); teK /= max(n, 1)
        print(f"[VAE {ep:03d}] train loss={trL:.6f} recon={trR:.6f} kl={trK:.6f} | "
              f"test loss={teL:.6f} recon={teR:.6f} kl={teK:.6f}")


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
        """
        TODO:
        - encode images (if used), use "time_distributed" function from robomimic
        - concatenate state + visual features
        - apply temporal conv
        - predict latent sequence
        """

        feats = [obs_state]
        if self.image_type == "both":
            feat_img = time_distributed(obs_image, self.obs_encoder.obs_nets["external"], inputs_as_kwargs=False)
            feat_wimg = time_distributed(obs_wrist_image, self.obs_encoder.obs_nets["wrist"], inputs_as_kwargs=False)
            feats += [feat_img, feat_wimg]   

        x = torch.cat(feats, dim=-1)
        x = x.transpose(1,2)

        x = self.temporal(x)
        x = x.mean(dim=-1)
        y = self.head(x)
        return y.view(x.shape[0], self.pred_horizon, self.z_dim)


def train_bc_to_latent(policy, action_ae, train_loader, test_loader, device="cuda", lr=1e-4, wd=1e-6, epochs=30):
    """
    Freeze ActionVAE. Train policy to predict z = encode(action).
    """
    action_ae.eval()
    for p in action_ae.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        policy.train()
        tr = 0.0; n = 0

        for b in tqdm(train_loader, desc=f"BC->z train {ep}/{epochs}"):
            obs_state = b["obs_state"].to(device)
            a = b["pred_action"].to(device)

            obs_img = b.get("obs_image", None)
            obs_wimg = b.get("obs_wrist_image", None)
            if obs_img is not None:  obs_img = obs_img.to(device)
            if obs_wimg is not None: obs_wimg = obs_wimg.to(device)

            with torch.no_grad():
                """
                TODO:
                - encode actions using action_ae
                - choose appropriate latent target (mu or sampled z)
                """
                _, _, mu, logvar = action_ae(a)
                z_tgt = mu
                

            z_pred = policy(obs_state, obs_img, obs_wimg)
            loss = loss_fn(z_pred, z_tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = obs_state.size(0)
            tr += loss.item() * bs
            n += bs

        tr /= max(n, 1)

        policy.eval()
        te = 0.0; n = 0
        with torch.no_grad():
            for b in test_loader:
                obs_state = b["obs_state"].to(device)
                a = b["pred_action"].to(device)

                obs_img = b.get("obs_image", None)
                obs_wimg = b.get("obs_wrist_image", None)
                if obs_img is not None:  obs_img = obs_img.to(device)
                if obs_wimg is not None: obs_wimg = obs_wimg.to(device)

                mu, logvar = action_ae.encode(a)
                z_tgt = mu
                z_pred = policy(obs_state, obs_img, obs_wimg)
                loss = loss_fn(z_pred, z_tgt)

                bs = obs_state.size(0)
                te += loss.item() * bs
                n += bs

        te /= max(n, 1)
        print(f"[BC->z {ep:03d}] train_mse={tr:.6f}  test_mse={te:.6f}")


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

    p.add_argument("--data_dir", type=str, default="/home/williamhuang/Downloads/xarm_lift_data")
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
    p.add_argument("--ae_epochs", type=int, default=15)
    p.add_argument("--bc_epochs", type=int, default=10)

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
    action_ae = ActionVAE(**action_ae_kwargs).to(device)

    # -----------------------------
    # mode dispatch
    # -----------------------------
    if args.mode == "vae":
        # trains encoder/decoder to reconstruct pred_action
        train_action_vae(action_ae, train_loader, test_loader,
                 device=device,
                 lr=args.vae_lr,
                 wd=args.vae_wd,
                 epochs=args.ae_epochs,
                 beta=args.vae_beta)

        # save AE-only checkpoint (policy is None)
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
        torch.save(
            {
                "policy_state_dict": None,
                "action_ae_state_dict": action_ae.state_dict(),
                "stats": stats,
                "policy_kwargs": None,
                "action_ae_kwargs": action_ae_kwargs,
            },
            args.ckpt_path,
        )
        print(f"Saved AE-only checkpoint to: {args.ckpt_path}")

    elif args.mode == "bc":
        # load AE weights if checkpoint exists (recommended)
        if os.path.exists(args.ckpt_path):
            ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
            if ckpt.get("action_ae_state_dict", None) is None:
                raise RuntimeError("Checkpoint exists but has no action_ae_state_dict. Run --mode train_vae first.")
            action_ae.load_state_dict(ckpt["action_ae_state_dict"])
            action_ae.eval()
            if ckpt.get("stats", None) is not None:
                stats = ckpt["stats"]
        else:
            raise RuntimeError(f"Checkpoint not found: {args.ckpt_path}. Run --mode train_vae first to create it.")

        # build policy
        policy_kwargs = dict(
            z_dim=args.z_dim,
            action_dim=args.action_dim,
            obs_dim=args.obs_dim,
            obs_horizon=args.obs_h,
            pred_horizon=args.pred_h,
            image_type=args.image_type,
        )
        policy = BCConvMLPPolicyLatent(**policy_kwargs).to(device)

        # train BC to predict latent actions
        train_bc_to_latent(policy, action_ae, train_loader, test_loader, device=device, epochs=args.bc_epochs)

        # save combined checkpoint
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "action_ae_state_dict": action_ae.state_dict(),
                "stats": stats,
                "policy_kwargs": policy_kwargs,
                "action_ae_kwargs": action_ae_kwargs,
            },
            args.ckpt_path,
        )
        print(f"Saved combined checkpoint to: {args.ckpt_path}")

    elif args.mode == "inference":
        if not os.path.exists(args.ckpt_path):
            raise RuntimeError(f"Checkpoint not found: {args.ckpt_path}")

        ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("action_ae_state_dict", None) is None or ckpt.get("policy_state_dict", None) is None:
            raise RuntimeError("Need combined checkpoint with BOTH action_ae_state_dict and policy_state_dict. Run --mode train_bc first.")

        # rebuild + load
        action_ae = ActionVAE(**ckpt["action_ae_kwargs"]).to(device)
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
