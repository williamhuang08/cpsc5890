import torch

def _random_crop_bhchw(x, crop_hw=(96, 96)):
    """
    x: (B, T, 3, H, W)
    Returns: (B, T, 3, 96, 96)
    """
    B, T, C, H, W = x.shape
    device = x.device
    ch, cw = crop_hw

    if H < ch or W < cw:
        raise ValueError(f"Image too small for crop {crop_hw}, got {(H,W)}")

    max_y = H - ch
    max_x = W - cw

    # random top-left per sample
    y0 = torch.randint(0, max_y + 1, (B,), device=device)
    x0 = torch.randint(0, max_x + 1, (B,), device=device)

    crops = []
    for b in range(B):
        crops.append(
            x[b, :, :, y0[b]:y0[b] + ch, x0[b]:x0[b] + cw]
        )  # (T,3,96,96)

    return torch.stack(crops, dim=0)

def _center_crop_bhchw(x, crop_hw=(96, 96)):
    """
    x: (B, T, 3, H, W)
    Returns: (B, T, 3, 96, 96)
    """
    B, T, C, H, W = x.shape
    ch, cw = crop_hw

    if H < ch or W < cw:
        raise ValueError(f"Image too small for crop {crop_hw}, got {(H,W)}")

    y0 = (H - ch) // 2
    x0 = (W - cw) // 2

    return x[:, :, :, y0:y0+ch, x0:x0+cw]