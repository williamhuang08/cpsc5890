import torch
import torch.nn as nn
import math
from typing import Union
from robomimic.models.obs_nets import ObservationEncoder
from robomimic.utils.tensor_utils import time_distributed

from scripts.crop import _random_crop_bhchw, _center_crop_bhchw

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class DiffusionPolicyUNet1D(nn.Module):
    def __init__(self,
                 action_dim,
                 obs_low_dim: int,
                 action_horizon=16,
                 obs_horizon=2,
                 diffusion_step_embed_dim=256,
                 down_dims=[256,512,1024],
                 kernel_size=5,
                 n_groups=8,
                 img_backbone_kwargs=None,
                 image_type = "both"
                 ):
        """
        UNet‑based diffusion policy with image encoding via robomimic.
        
        Args:
            action_dim: dimensionality of actions
            obs_low_dim: dimensionality of low‑dim (non‐image) observations (if any)
            action_horizon: horizon length for action sequence
            obs_horizon: number of past observation timesteps (for low dim obs)
            diffusion_step_embed_dim: embedding size for diffusion time step
            down_dims: channels for UNet down path
            kernel_size, n_groups: as before
            img_backbone_kwargs: dict of kwargs passed to VisualCore for both image streams
        """
        super().__init__()
        self.action_dim = action_dim
        self.obs_low_dim = obs_low_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        
        # Time‐step encoder
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        
        # Define image encoders via robomimic VisualCore
        # Let's assume both external and wrist camera have same backbone config
        if img_backbone_kwargs is None:
            img_backbone_kwargs = {
                "input_shape": [3, 96, 96],
                "backbone_class": "ResNet18Conv",
                "backbone_kwargs": {"pretrained": True, "input_coord_conv": False},
                "pool_class": "SpatialSoftmax",
                "pool_kwargs": {"num_kp": 32},
                "feature_dimension": 64,
            }
        # two image feature dimensions
        img_feat_dim = img_backbone_kwargs["feature_dimension"]
        
        self.obs_encoder = ObservationEncoder(feature_activation=nn.ReLU)
        # register both cameras
        self.obs_encoder.register_obs_key(
            name="external",
            shape=img_backbone_kwargs["input_shape"],
            net_class="VisualCore",
            net_kwargs=img_backbone_kwargs
        )
        self.obs_encoder.register_obs_key(
            name="wrist",
            shape=img_backbone_kwargs["input_shape"],
            net_class="VisualCore",
            net_kwargs=img_backbone_kwargs
        )

        # also register low‐dim observation, if any
        if self.obs_low_dim > 0:
            from robomimic.models.base_nets import MLP
            self.obs_encoder.register_obs_key(
                name="low_dim_obs",
                shape=[self.obs_low_dim * self.obs_horizon],
                net=MLP(input_dim=self.obs_low_dim * self.obs_horizon,
                        output_dim=img_feat_dim,
                        layer_dims=(128,),
                        output_activation=None)
            )

        self.obs_encoder.make()

        # === 2. Replace all BatchNorm with GroupNorm ===
        def swap_bn_to_gn(module: nn.Module, max_groups: int = 32):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    C = child.num_features
                    setattr(module, name, nn.GroupNorm(num_groups=min(max_groups, C), num_channels=C))
                else:
                    swap_bn_to_gn(child, max_groups)
            return module

        swap_bn_to_gn(self.obs_encoder.obs_nets["external"])
        swap_bn_to_gn(self.obs_encoder.obs_nets["wrist"])

        # # Compute the total global conditioning dimension
        # self.global_cond_dim = img_feat_dim * 2 + (img_feat_dim if self.obs_low_dim > 0 else 0)
        # self.cond_dim = dsed + self.global_cond_dim

        # total cond_dim

        if image_type == "both":
            self.cond_dim = dsed + img_feat_dim * self.obs_horizon * 2 + obs_low_dim * self.obs_horizon if obs_low_dim > 0 else 0
        elif image_type == "none":
            self.cond_dim = dsed + obs_low_dim * self.obs_horizon if obs_low_dim > 0 else 0
        
        # Setup UNet as before
        all_dims = [action_dim] + list(down_dims)
        start_dim = down_dims[0]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = down_dims[-1]
        
        # Mid modules
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=self.cond_dim,
                                       kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=self.cond_dim,
                                       kernel_size=kernel_size, n_groups=n_groups),
        ])
        
        # Down modules
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = (ind >= len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=self.cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=self.cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if (not is_last) else nn.Identity()
            ]))
        
        # Up modules (mirror)
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = (ind >= len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=self.cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=self.cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if (not is_last) else nn.Identity()
            ]))
        
        # Final conv
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, action_dim, kernel_size=1),
        )
        
        print("UNet with RoboMimic image encoder — params:", sum(p.numel() for p in self.parameters()))
    
    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: Union[torch.Tensor, float, int],
        observations: dict
    ):
        """
        Forward pass of a conditional 1D-UNet diffusion model.

        See assignment write-up for full description.
        """

        # ============================================================
        # TODO 0: Basic bookkeeping
        # ============================================================
        B = noisy_actions.shape[0]

        # ============================================================
        # TODO 1: Reorder action tensor for 1D UNet
        # ============================================================
        # Convert:
        #   [B, action_horizon, action_dim]
        # into:
        #   [B, action_dim, action_horizon]
        #
        # Why:
        #   1D conv expects channels-first format.
        #
        # YOUR CODE HERE
        raise NotImplementedError


        # ============================================================
        # TODO 2: Standardize timesteps
        # ============================================================
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0).to(sample.device)
        timesteps = timesteps.expand(B)

        # ============================================================
        # TODO 3: Encode diffusion timestep
        # ============================================================
        # Map timesteps [B] → t_feat [B, t_dim]
        #
        # YOUR CODE HERE
        raise NotImplementedError


        # ============================================================
        # TODO 4: Encode observation streams
        # ============================================================
        # Build a list of per-time-step features:
        #   each entry: [B, T, feat_dim_i]
        #
        # Steps:
        #   - If "external" in observations:
        #         encode via time_distributed
        #   - If "wrist" in observations:
        #         encode via time_distributed
        #   - If "low_dim_obs":
        #         expand to [B, T, D] if needed
        #
        # Concatenate along last dim.
        #
        # YOUR CODE HERE
        raise NotImplementedError


        # ============================================================
        # TODO 5: Flatten observation encoding
        # ============================================================
        # obs_enc: [B, T, total_feat_dim]
        # → [B, T * total_feat_dim]
        obs_enc = obs_enc.reshape(B, -1)

        # ============================================================
        # TODO 6: Combine timestep + observation conditioning
        # ============================================================
        cond = torch.cat([t_feat, obs_enc], dim=-1)

        # ============================================================
        # TODO 7: Initialize UNet state
        # ============================================================
        x = sample
        h = []

        # ============================================================
        # TODO 8: UNet Downsampling Path
        # ============================================================
        # For each (res1, res2, down):
        #   x = res1(x, cond)
        #   x = res2(x, cond)
        #   store skip
        #   x = down(x)
        #
        # YOUR CODE HERE
        raise NotImplementedError


        # ============================================================
        # TODO 9: UNet Middle Blocks
        # ============================================================
        # For each mid block:
        #   x = mid(x, cond)
        #
        # YOUR CODE HERE
        raise NotImplementedError


        # ============================================================
        # TODO 10: UNet Upsampling Path
        # ============================================================
        # For each (res1, res2, up):
        #   pop skip
        #   concatenate along channel dim
        #   apply res1, res2
        #   upsample
        #
        # YOUR CODE HERE
        raise NotImplementedError


        # ============================================================
        # TODO 11: Final projection
        # ============================================================
        # Map UNet channels → action_dim channels
        #
        # YOUR CODE HERE
        raise NotImplementedError


        # ============================================================
        # TODO 12: Restore original tensor layout
        # ============================================================
        predicted_noise = x.moveaxis(-1, -2)
        return predicted_noise

# Wrapper for compatibility:
class DiffusionPolicyUNet(nn.Module):
    def __init__(self, obs_low_dim: int, action_dim: int, action_horizon: int = 16,
                 obs_horizon: int = 2, image_type: str = "none", **kwargs):
        super().__init__()
        self.obs_low_dim = obs_low_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.image_type = image_type
        
        self.unet = DiffusionPolicyUNet1D(
            action_dim=action_dim,
            obs_low_dim=obs_low_dim,
            action_horizon=action_horizon,
            obs_horizon=obs_horizon,
            image_type = image_type,    
            **kwargs
        )
            
    def get_optimizer(
        self,
        learning_rate: float = 1e-4,
        unet_weight_decay: float = 1e-6,
        obs_encoder_weight_decay: float = 1e-6,
        betas=(0.9, 0.95),
    ) -> torch.optim.AdamW:
        """
        TODO: Create an AdamW optimizer with separate parameter groups.

        Goal:
            - Use AdamW.
            - Apply DIFFERENT weight decay values to:
                (1) UNet parameters
                (2) Observation encoder parameters

        Why separate them?
            - Vision backbones (obs_encoder) are often pretrained and may
            require smaller weight decay.
            - The UNet may benefit from different regularization strength.
            - Fine-grained control improves training stability.

        Steps:

        1) Iterate through self.named_parameters()
        - Skip parameters where requires_grad == False.
        - If parameter name contains "obs_encoder":
                → add to obs_encoder_params
            Else:
                → add to unet_params

        2) Construct torch.optim.AdamW with TWO parameter groups:
            [
                {"params": unet_params, "weight_decay": unet_weight_decay},
                {"params": obs_encoder_params, "weight_decay": obs_encoder_weight_decay},
            ]

        3) Use:
            lr = learning_rate
            betas = betas

        Return:
            The constructed AdamW optimizer.

        Notes:
            - Make sure each parameter appears in exactly one group.
            - Do NOT include frozen parameters.
            - Keep function signature unchanged.
        """

        # YOUR CODE HERE
        raise NotImplementedError

    def forward(self,
                noisy_actions: torch.Tensor,
                timesteps: torch.Tensor,
                observations: torch.Tensor = None,
                img_ext: torch.Tensor = None,
                img_wst: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            noisy_actions: [B, action_horizon, action_dim]
            timesteps: [B] or scalar
            observations: [B, obs_horizon, obs_low_dim] or [B, obs_low_dim]
            img_ext: [B, 3, H, W] external camera images
            img_wst: [B, 3, H, W] wrist camera images
        Returns:
            predicted_noise: [B, action_horizon, action_dim]
        """
        """
        TODO: Apply spatial cropping to image inputs.

        Goal:
            Standardize all image inputs to spatial size (96, 96).

        Behavior:
            - If the model is in training mode (self.training == True):
                → apply RANDOM crop for data augmentation.
            - If the model is in evaluation mode:
                → apply CENTER crop for deterministic behavior.

        Why:
            - Random crop improves robustness and reduces overfitting.
            - Center crop ensures stable evaluation results.
            - UNet expects fixed-size images.

        Inputs:
            img_ext: [B, 3, H, W]  (external camera)
            img_wst: [B, 3, H, W]  (wrist camera)

        Outputs:
            Cropped tensors of shape [B, 3, 96, 96].

        Implementation hints:
            - Only crop if the tensor is not None.
            - Use _random_crop_bhchw(...) during training.
            - Use _center_crop_bhchw(...) during evaluation.
            - Do NOT change dtype or device.
        """

        # YOUR CODE HERE
        raise NotImplementedError

        obs_dict = {}
        if img_ext is not None:
            obs_dict["external"] = img_ext
        if img_wst is not None:
            obs_dict["wrist"] = img_wst
        if observations is not None and self.obs_low_dim > 0:
            if observations.dim() == 2:
                observations = observations.unsqueeze(1).expand(-1, self.obs_horizon, -1)
            obs_dict["low_dim_obs"] = observations
        
        return self.unet(noisy_actions, timesteps, obs_dict)