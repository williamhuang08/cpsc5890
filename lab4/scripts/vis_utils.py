import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

ArrayLike = Union[np.ndarray, torch.Tensor]


# ----------------------------
# basic helpers
# ----------------------------
def _to_np(x: ArrayLike) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def subsample_list(items: List, n: int) -> List:
    """Keep n evenly spaced items from list; if n<=0 or n>=len(items), return items."""
    if n <= 0 or n >= len(items):
        return items
    idxs = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idxs]


# ----------------------------
# xArm7 forward kinematics (DH / modified DH)
# ----------------------------
def _rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]], dtype=np.float64)


def _rotx(alpha: float) -> np.ndarray:
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1, 0,  0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0,  0, 1]], dtype=np.float64)


def _transz(d: float) -> np.ndarray:
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, d],
                     [0, 0, 0, 1]], dtype=np.float64)


def _transx(a: float) -> np.ndarray:
    return np.array([[1, 0, 0, a],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float64)


def dh_standard(theta: float, d: float, a: float, alpha: float, offset: float = 0.0) -> np.ndarray:
    """
    Standard DH: Rz(theta+offset) * Tz(d) * Tx(a) * Rx(alpha)
    """
    return _rotz(theta + offset) @ _transz(d) @ _transx(a) @ _rotx(alpha)


def dh_modified(theta: float, d: float, a: float, alpha: float, offset: float = 0.0) -> np.ndarray:
    """
    Modified DH (common robotics convention): Rz(theta+offset) * Tx(a) * Rx(alpha) * Tz(d)
    """
    return _rotz(theta + offset) @ _transx(a) @ _rotx(alpha) @ _transz(d)


def xarm7_dh_params(which: str = "modified") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays (d, a, alpha, offset) for 7 joints, in meters/radians.

    Values taken from UFACTORY xArm7 manual tables:
      - Modified D-H Parameters (xArm7)  [oai_citation:1‡UFACTORY Official Website](https://www.ufactory.cc/wp-content/uploads/2023/05/xArm-User-Manual-V2.0.0.pdf)
      - Standard D-H Parameters (xArm7)  [oai_citation:2‡UFACTORY Official Website](https://www.ufactory.cc/wp-content/uploads/2023/05/xArm-User-Manual-V2.0.0.pdf)
    """
    which = which.lower()
    if which not in ("modified", "standard"):
        raise ValueError("which must be 'modified' or 'standard'")

    if which == "modified":
        # Joint i: theta=0, d(mm), alpha(rad), a(mm), offset(rad)
        # [d, a, alpha, offset]
        table = [
            (267.0,   0.0,   0.0,        0.0),
            (  0.0,   0.0,  -np.pi/2,    0.0),
            (293.0,   0.0,   np.pi/2,    0.0),
            (  0.0,  52.5,   np.pi/2,    0.0),
            (342.5,  77.5,   np.pi/2,    0.0),
            (  0.0,   0.0,   np.pi/2,    0.0),
            ( 97.0,  76.0,  -np.pi/2,    0.0),
        ]
    else:
        # Standard DH table from manual  [oai_citation:3‡UFACTORY Official Website](https://www.ufactory.cc/wp-content/uploads/2023/05/xArm-User-Manual-V2.0.0.pdf)
        table = [
            (267.0,   0.0,  -np.pi/2,    0.0),
            (  0.0,   0.0,   np.pi/2,    0.0),
            (293.0,  52.5,   np.pi/2,    0.0),
            (  0.0,  77.5,   np.pi/2,    0.0),
            (342.5,   0.0,   np.pi/2,    0.0),
            (  0.0,  76.0,  -np.pi/2,    0.0),
            ( 97.0,   0.0,   0.0,        0.0),
        ]

    d = np.array([r[0] for r in table], dtype=np.float64) / 1000.0  # m
    a = np.array([r[1] for r in table], dtype=np.float64) / 1000.0  # m
    alpha = np.array([r[2] for r in table], dtype=np.float64)
    offset = np.array([r[3] for r in table], dtype=np.float64)
    return d, a, alpha, offset


def xarm7_fk(
    q: Sequence[float],
    which: str = "modified",
) -> np.ndarray:
    """
    Forward kinematics for xArm7.
    q: 7 joint angles in radians
    Returns 4x4 T_0_ee
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] != 7:
        raise ValueError(f"Expected q shape (7,), got {q.shape}")

    d, a, alpha, offset = xarm7_dh_params(which=which)

    T = np.eye(4, dtype=np.float64)
    for i in range(7):
        if which == "modified":
            Ti = dh_modified(q[i], d[i], a[i], alpha[i], offset[i])
        else:
            Ti = dh_standard(q[i], d[i], a[i], alpha[i], offset[i])
        T = T @ Ti
    return T


def xarm7_fk_xyz(
    q: Sequence[float],
    which: str = "modified",
) -> np.ndarray:
    T = xarm7_fk(q, which=which)
    return T[:3, 3].astype(np.float32)


# ----------------------------
# build EE trajectory from joint actions
# ----------------------------
def get_start_q_from_obs(
    obs: ArrayLike,
    obs_q_idx: Sequence[int],
    use_last: bool = True,
) -> np.ndarray:
    """
    obs: (obs_horizon, obs_dim) or (obs_dim,)
    obs_q_idx: indices (len=7) of joint angles inside obs vector
    """
    o = _to_np(obs)
    row = o if o.ndim == 1 else (o[-1] if use_last else o[0])
    q0 = row[list(obs_q_idx)].astype(np.float32)
    if q0.shape[0] != 7:
        raise ValueError("obs_q_idx must specify 7 indices for xArm7")
    return q0


def joint_actions_to_q_seq(
    actions: ArrayLike,
    q0: ArrayLike,
    action_mode: str = "delta",   # "delta" or "absolute"
    action_q_idx: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    actions: (H, D) joint actions
    q0: (7,)
    action_mode:
      - "delta": q_{t+1} = q_t + action[t]
      - "absolute": q_t = action[t]  (q0 only used to prepend)
    action_q_idx: if actions has extra dims, pick 7 joint dims here; else None uses first 7.
    Returns q_seq: (H+1, 7) including initial q0
    """
    a = _to_np(actions).astype(np.float32)
    if a.ndim != 2:
        raise ValueError(f"actions must be (H,D), got {a.shape}")

    if action_q_idx is None:
        if a.shape[1] < 7:
            raise ValueError(f"actions dim D={a.shape[1]} < 7; provide action_q_idx")
        u = a[:, :7]
    else:
        u = a[:, list(action_q_idx)]

    q0 = _to_np(q0).astype(np.float32).reshape(7)
    H = u.shape[0]

    q_seq = np.zeros((H + 1, 7), dtype=np.float32)
    q_seq[0] = q0

    if action_mode == "absolute":
        q_seq[1:] = u
    elif action_mode == "delta":
        q_seq[1:] = q0 + np.cumsum(u, axis=0)
    else:
        raise ValueError("action_mode must be 'delta' or 'absolute'")

    return q_seq


def q_seq_to_ee_traj_xyz(
    q_seq: ArrayLike,
    which: str = "modified",
) -> np.ndarray:
    """
    q_seq: (T,7) -> xyz trajectory (T,3)
    """
    qs = _to_np(q_seq).astype(np.float64)
    if qs.ndim != 2 or qs.shape[1] != 7:
        raise ValueError(f"q_seq must be (T,7), got {qs.shape}")

    traj = np.zeros((qs.shape[0], 3), dtype=np.float32)
    for i in range(qs.shape[0]):
        traj[i] = xarm7_fk_xyz(qs[i], which=which)
    return traj


# ----------------------------
# gif rendering
# ----------------------------
def render_denoise_gif_3d(
    trajs: List[np.ndarray],
    out_path: str,
    gt_traj: Optional[np.ndarray] = None,
    fps: int = 6,
    elev: float = 25,
    azim: float = -60,
    title: str = "DDIM denoise (EE XYZ)",
):
    """
    trajs: list of (T,3) arrays, one per denoise step (including initial noisy)
    Saves a GIF to out_path.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    all_pts = np.concatenate(trajs + ([gt_traj] if gt_traj is not None else []), axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    pad = 0.05 * (maxs - mins + 1e-6)
    mins -= pad
    maxs += pad

    frames = []
    for k, traj in enumerate(trajs):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)

        if gt_traj is not None:
            ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], linewidth=2, label="GT")

        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2, label="Sample")
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], s=30, marker="o", label="Start")
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], s=30, marker="x", label="End")

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{title} | step {k}/{len(trajs) - 1}")
        ax.legend(loc="upper left")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        # backend-agnostic RGB extraction
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4)  # RGBA
        img = buf[:, :, :3].copy()  # drop alpha, keep RGB

        frames.append(img)
        plt.close(fig)

    try:
        import imageio.v2 as imageio
        imageio.mimsave(out_path, frames, fps=fps)
    except Exception:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        duration_ms = int(1000 / max(1, fps))
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)


def denoise_actions_to_ee_trajs(
    denoise_actions: List[ArrayLike],
    obs_np: ArrayLike,
    *,
    obs_q_idx: Sequence[int],
    action_mode: str = "delta",
    action_q_idx: Optional[Sequence[int]] = None,
    fk_which: str = "modified",
) -> List[np.ndarray]:
    """
    denoise_actions: list of (H,D) actions, one per DDIM denoise step
    obs_np: (obs_horizon, obs_dim) raw obs used to extract q0
    Returns list of EE xyz trajs: each is (H+1,3)
    """
    q0 = get_start_q_from_obs(obs_np, obs_q_idx=obs_q_idx, use_last=True)

    trajs = []
    for a in denoise_actions:
        q_seq = joint_actions_to_q_seq(a, q0=q0, action_mode=action_mode, action_q_idx=action_q_idx)
        ee = q_seq_to_ee_traj_xyz(q_seq, which=fk_which)
        trajs.append(ee)
    return trajs