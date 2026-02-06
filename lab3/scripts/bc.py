import os, numpy as np, torch
import torch.nn as nn
from tqdm import tqdm
from robomimic.models.obs_nets import ObservationEncoder
from robomimic.utils.tensor_utils import time_distributed

from robomimic.models.obs_core import VisualCore
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax

from scripts.dataset import make_loaders


class Conv1dBlock(nn.Module):
    def __init__(self, inp, out, k=3, n_groups=8):
        super().__init__()
        """
        TODO:
        - 1D convolution with padding
        - normalization (GroupNorm)
        - nonlinearity
        """
        
        self.net = nn.Sequential(
            nn.Conv1d(inp, out, k, padding=1),
            nn.GroupNorm(n_groups, out),
            nn.ReLU()
        )

    def forward(self, x):
        """
        TODO:
        - apply the block to input x
        """
        return self.net(x)

def swap_bn_to_gn(module: nn.Module, max_groups: int = 32):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            C = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups=min(max_groups, C), num_channels=C))
        else:
            swap_bn_to_gn(child, max_groups)
    return module


class BCConvMLPPolicy(nn.Module):
    """
    Inputs:
      obs_state:       (B, Hobs, obs_dim) normalized
      obs_image:       (B, Hobs, 3, H, W) normalized (train stats)
      obs_wrist_image: (B, Hobs, 3, H, W) normalized

    Output:
      pred_action:     (B, Hpred, action_dim) normalized (same space as loader target)
    """
    def __init__(
        self,
        action_dim=8,
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
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.image_type = image_type

        if img_backbone_kwargs is None:
            """
            TODO:
            - define img_backbone_kwargs for VisualCore, read robomimic documentation: https://robomimic.github.io/docs/modules/models.html
            - must specify input_shape, backbone, pooling, and feature dimension
            """
            img_backbone_kwargs = dict(
                input_shape=(3, 96, 96),
                backbone_class="ResNet18Conv",  # use ResNet18 as the visualcore backbone
                backbone_kwargs={"pretrained": False, "input_coord_conv": False},  # kwargs for the ResNet18Conv class
                pool_class="SpatialSoftmax",  # use spatial softmax to regularize the model output
                pool_kwargs={"num_kp": 32},  # kwargs for the SpatialSoftmax --- use 32 keypoints
                flatten=True,  # flatten the output of the spatial softmax layer
                feature_dimension=img_feat_dim,  # project the flattened feature into a 64-dim vector through a linear layer
            )
            
            
        img_feat_dim = img_backbone_kwargs["feature_dimension"]

        # --- encoders (robomimic) ---
        self.obs_encoder = ObservationEncoder(feature_activation=nn.ReLU)
        if image_type == "both":
            self.obs_encoder.register_obs_key("external", img_backbone_kwargs["input_shape"],
                                              net_class="VisualCore", net_kwargs=img_backbone_kwargs)
            self.obs_encoder.register_obs_key("wrist", img_backbone_kwargs["input_shape"],
                                              net_class="VisualCore", net_kwargs=img_backbone_kwargs)
        self.obs_encoder.make()

        if image_type == "both":
            """
            TODO:
            - decide whether to replace BatchNorm with GroupNorm in visual backbones
            - justify your choice for small-batch BC
            """
            swap_bn_to_gn(self.obs_encoder, 8)
            # we would like to replace BN w/ GN since GN is preferred for smaller batch sizes

        # per-timestep feature dim
        per_t = obs_dim + (2 * img_feat_dim if image_type == "both" else 0)

        # --- 1D conv over time (BC-Conv) ---
        layers = [Conv1dBlock(per_t, conv_channels, k=kernel_size)]
        for _ in range(conv_layers - 1):
            layers.append(Conv1dBlock(conv_channels, conv_channels, k=kernel_size))
        self.temporal = nn.Sequential(*layers)

        # --- MLP head (MLP) ---
        self.head = nn.Sequential(
            nn.Linear(conv_channels, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, pred_horizon * action_dim),
        )

    def forward(self, obs_state, obs_image=None, obs_wrist_image=None):
        """
        TODO:
        - encode visual observations (if enabled), use "time_distributed" function from robomimic
        - concatenate state + image features per timestep
        """
        feats=[obs_state]

        # print(obs_image.shape)
        # print(obs_wrist_image.shape)

        if self.image_type == "both":
            ext = time_distributed(obs_image, self.obs_encoder.obs_nets["external"], inputs_as_kwargs=False)
            wst = time_distributed(obs_wrist_image, self.obs_encoder.obs_nets["wrist"], inputs_as_kwargs=False)
            feats+=[ext,wst]
        
        x = torch.cat(feats, dim=-1)
        x = x.transpose(1,2)

        x = self.temporal(x)
        x = x.mean(dim=-1)
        y = self.head(x)
        return y.view(x.shape[0], self.pred_horizon, self.action_dim)

        

def train_bc(model, train_loader, test_loader, device="cuda", lr=1e-4, wd=1e-6, epochs=30):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        tr = 0.0; n = 0
        for b in tqdm(train_loader):
            obs_state = b["obs_state"].to(device)
            tgt = b["pred_action"].to(device)

            obs_img = b.get("obs_image", None)
            obs_wimg = b.get("obs_wrist_image", None)
            if obs_img is not None:  obs_img = obs_img.to(device)
            if obs_wimg is not None: obs_wimg = obs_wimg.to(device)
                
            """
            TODO:
            - run policy forward pass
            - compute behavior cloning loss
            """
            action = model.forward(obs_state, obs_img, obs_wimg)
            loss = loss_fn(action, tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = obs_state.size(0)
            tr += loss.item() * bs
            n += bs

        tr /= max(n, 1)

        model.eval()
        te = 0.0; n = 0
        with torch.no_grad():
            for b in test_loader:
                obs_state = b["obs_state"].to(device)
                tgt = b["pred_action"].to(device)

                obs_img = b.get("obs_image", None)
                obs_wimg = b.get("obs_wrist_image", None)
                if obs_img is not None:  obs_img = obs_img.to(device)
                if obs_wimg is not None: obs_wimg = obs_wimg.to(device)
                    
                """
                TODO:
                - forward pass without gradients
                - accumulate validation loss
                """
                action = model.forward(obs_state, obs_img, obs_wimg)
                loss_fn(action, tgt)

                bs = obs_state.size(0)
                te += loss.item() * bs
                n += bs

        te /= max(n, 1)
        print(f"[{ep:03d}] train_mse={tr:.6f}  test_mse={te:.6f}")


def save_model(path, model, stats, model_kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "stats": stats,
        "model_kwargs": model_kwargs,
    }, path)

if __name__ == "__main__":

    train_loader, test_loader, stats = make_loaders(
        "/home/williamhuang/Downloads/xarm_lift_data",
        obs_h=1,
        pred_h=16,
        batch_size=64,
        include_images=True,
        test_ratio=0.1,
        num_workers=0
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_kwargs = dict(
        action_dim=8,
        obs_dim=8,
        obs_horizon=2,
        pred_horizon=16,
        image_type="both",
    )

    model = BCConvMLPPolicy(**model_kwargs).to(device)

    train_bc(model, train_loader, test_loader, device=device, epochs=15)

    save_model(
        "asset/checkpoints/bcconv_final.pt",
        model,
        stats,
        model_kwargs
    )

    ckpt = torch.load("asset/checkpoints/bcconv_final.pt", map_location="cpu", weights_only=False)
    model = BCConvMLPPolicy(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    stats = ckpt["stats"]
    print(stats)
