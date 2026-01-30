"""
DAgger (Student Version)

Students will fill in:
- normalization + training pieces inside train_bc_on_arrays (or call into their BC implementation)
- DAgger rollout collection loop details (expert labeling, mixture execution)
- aggregation + retraining loop in main
- saving/loading artifacts

NOTE: This file imports from scripts.bc. In your lab repo, make sure to have:
  - scripts/bc.py (BC TODOs)
  - scripts/dagger.py (this file)
and update the import accordingly if needed.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

from xarm_lab.arm_utils import (
    connect_arm, disconnect_arm, ArmConfig,
    get_joint_angles, get_tcp_pose, get_gripper_position
)
from xarm_lab.safety import enable_basic_safety, clear_faults
from xarm_lab.kinematics import ik_from_pose
from utils.plot import plot_3d_positions
from utils.collect_demo_high_freq import policy

# You can point this import at the student BC file if desired:
from scripts.bc import *  # contains: load_data_by_episode, compute_norm_stats, normalize, BCPolicy, evaluate


def train_bc_on_arrays(
    X_raw_train, Y_raw_train, X_raw_test, Y_raw_test,
    device,
    epochs=200,
    batch_size=256,
    lr=1e-3,
):
    """
    (Re)train a BC policy on provided (already-flattened) arrays.

    TODO:
    - compute normalization stats on TRAIN only
    - normalize train/test
    - create TensorDatasets + DataLoaders
    - instantiate BCPolicy + optimizer + MSE loss
    - training loop
    - return model and (X_mean, X_std, Y_mean, Y_std)
    """

    # TODO: compute normalization stats on training arrays
    X_mean, X_std = None, None  # TODO
    Y_mean, Y_std = None, None  # TODO

    # TODO: normalize (X_raw_train, X_raw_test, Y_raw_train, Y_raw_test)
    Xtr = None  # TODO
    Xte = None  # TODO
    Ytr = None  # TODO
    Yte = None  # TODO

    # TODO: wrap in datasets + loaders
    train_ds = None  # TODO
    test_ds = None   # TODO
    train_loader = None  # TODO
    test_loader = None   # TODO

    # TODO: create model, optimizer, loss
    model = None      # TODO
    optimizer = None  # TODO
    loss_fn = None    # TODO

    # TODO: training loop
    # for ep in range(1, epochs+1):
    #   model.train()
    #   for x,y in train_loader:
    #     ...
    #   optionally print evaluate() every few epochs
    raise NotImplementedError

    return model, (X_mean, X_std, Y_mean, Y_std)


def rollout_dagger_collect(
    arm,
    model,
    norm_stats,
    device,
    obs_horizon,
    episodes,
    beta,
):
    """
    Runs rollouts on robot and collects (obs_stack, expert_action) pairs.
    Executes mixture policy: expert w.p. beta else learned.
    Returns: X_new_raw, Y_new_raw

    TODO:
    - Build obs history buffer
    - For visited states: ALWAYS label with expert action (DAgger)
    - For execution: mixture of expert vs learned action
    """
    X_mean, X_std, Y_mean, Y_std = norm_stats

    X_new, Y_new = [], []

    model.eval()

    for ep in range(episodes):
        # Home & randomize start (kept as-is)
        code, initial_joints = arm.get_initial_point()
        arm.set_servo_angle(angle=initial_joints, speed=20.0, wait=True, is_radian=False)

        pose = get_tcp_pose(arm)
        pose[:3] += np.random.uniform(-5, 5, size=3)
        goal_pose = pose.copy()

        goal_q = ik_from_pose(arm, goal_pose)

        arm.set_servo_angle(angle=goal_q, speed=20.0, wait=True, is_radian=True)
        arm.set_gripper_position(600, wait=True, speed=0.1)

        obs_buffer = deque(maxlen=obs_horizon)

        # Reset expert policy internal state (kept as-is)
        policy.stage = "PICK_HOVER"
        policy.seg_t0 = None
        policy.seg_p0 = None
        policy.seg_p1 = None
        policy.seg_T  = None

        for t in range(200):
            # --- read state ---
            # TODO:
            # q = get_joint_angles(arm)
            # g = float(get_gripper_position(arm))
            # state = np.concatenate([q, [g]]).astype(np.float32)
            q = None       # TODO
            g = None       # TODO
            state = None   # TODO

            # TODO: obs_buffer.append(state)
            # if len(obs_buffer) < obs_horizon: continue
            raise NotImplementedError

            # TODO: obs_stack = np.concatenate(list(obs_buffer), axis=0).astype(np.float32)
            obs_stack = None  # TODO

            # --- expert label (DAgger) ---
            # IMPORTANT: label with expert for visited states
            # NOTE: policy(arm) returns (a_exp, done) in your code
            # TODO: call expert policy
            a_exp, done = None, None  # TODO

            # TODO: store training pair
            # X_new.append(obs_stack)
            # Y_new.append(a_exp)
            raise NotImplementedError

            # --- mixture execution ---
            # TODO:
            # With prob beta: execute expert action (a_exp)
            # Else: execute learned policy action (predicted from model)
            #
            # Steps for learned action:
            #   x = (obs_stack - X_mean) / X_std
            #   x = torch.tensor(x, dtype=torch.float32, device=device)
            #   with torch.no_grad(): a_norm = model(x).cpu().numpy()
            #   a_learned = a_norm * Y_std + Y_mean
            #
            # Then choose:
            #   action_exec = a_exp if rng < beta else a_learned
            raise NotImplementedError

            # --- execute (skeleton kept, students fill action_exec + dq) ---
            dq = action_exec[:7]
            arm.set_servo_angle(
                angle=(q + dq).tolist(),
                speed=0.5,
                wait=False,
                is_radian=True,
            )

            if len(action_exec) >= 8:
                arm.set_gripper_position(float(action_exec[7]), wait=False, speed=0.1)

            if done:
                break

        print(f"DAgger rollout episode {ep+1}/{episodes} done.")

    return np.asarray(X_new, dtype=np.float32), np.asarray(Y_new, dtype=np.float32)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference", "dagger"], default="train")
    parser.add_argument("--data", default="asset/demo_high_freq.npz")

    # BC training params
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # robot / inference params
    parser.add_argument("--ip", required=True)
    parser.add_argument("--out", default="asset/inf.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--obs_horizon", type=int, default=1)
    parser.add_argument("--inf_steps", type=int, default=200)

    # dagger params
    parser.add_argument("--dagger-iters", type=int, default=1)
    parser.add_argument("--dagger-rollout-episodes", type=int, default=5)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--beta-decay", type=float, default=0.8)  # beta_k = beta0 * decay^k

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load initial dataset (expert demos)
    Xtr0, Ytr0, Xte0, Yte0 = load_data_by_episode(
        args.data,
        H=args.obs_horizon,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    if args.mode == "train":
        print(f"Train samples: {len(Xtr0)} | Test samples: {len(Xte0)}")

        # TODO: train initial BC on demonstrations using train_bc_on_arrays
        model, (X_mean, X_std, Y_mean, Y_std) = None, (None, None, None, None)  # TODO
        raise NotImplementedError

        # TODO (optional): save model and normalization
        # torch.save(...)
        # np.savez(...)
        raise NotImplementedError

    elif args.mode == "dagger":
        # Aggregated dataset starts with original demos
        Xtr_agg = Xtr0.copy()
        Ytr_agg = Ytr0.copy()
        Xte = Xte0
        Yte = Yte0

        # Initial train
        print("[DAgger] Initial BC training on demonstrations...")
        model, (X_mean, X_std, Y_mean, Y_std) = train_bc_on_arrays(
            Xtr_agg, Ytr_agg, Xte, Yte,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        arm = connect_arm(ArmConfig(ip=args.ip))
        try:
            clear_faults(arm)
            enable_basic_safety(arm)

            arm.set_gripper_mode(0)
            arm.set_gripper_enable(True)
            arm.set_gripper_speed(5000)

            for k in range(args.dagger_iters):
                beta = args.beta0 * (args.beta_decay ** k)
                print(f"\n[DAgger] Iter {k+1}/{args.dagger_iters} | beta={beta:.4f}")

                # TODO:
                # - collect on-policy states using rollout_dagger_collect
                # - aggregate X_new/Y_new into Xtr_agg/Ytr_agg
                # - retrain using train_bc_on_arrays on aggregated dataset
                # - save artifacts each iteration (optional)
                raise NotImplementedError

        finally:
            disconnect_arm(arm)

    elif args.mode == "inference":
        # Load model
        model = BCPolicy(obs_dim=Xtr0.shape[1], act_dim=Ytr0.shape[1]).to(device)

        # TODO: load dagger policy weights (asset/dagger_policy.pt)
        raise NotImplementedError

        model.eval()

        # TODO: load dagger normalization (asset/dagger_norm.npz)
        raise NotImplementedError
        X_mean, X_std, Y_mean, Y_std = None, None, None, None  # TODO

        arm = connect_arm(ArmConfig(ip=args.ip))

        ep_states_list = []
        ep_actions_list = []

        try:
            clear_faults(arm)
            enable_basic_safety(arm)

            arm.set_gripper_mode(0)
            arm.set_gripper_enable(True)
            arm.set_gripper_speed(5000)

            print("\n=== Inference ===")

            for ep in range(args.episodes):
                # Home / randomize start (kept as-is)
                code, initial_joints = arm.get_initial_point()
                arm.set_servo_angle(angle=initial_joints, speed=20.0, wait=True, is_radian=False)

                pose = get_tcp_pose(arm)
                pose[:3] += np.random.uniform(-5, 5, size=3)
                joint_angles = ik_from_pose(arm, pose)
                arm.set_servo_angle(angle=joint_angles, speed=20.0, wait=True, is_radian=True)
                arm.set_gripper_position(600, wait=True, speed=0.1)

                states, actions, eefs = [], [], []
                obs_buffer = deque(maxlen=args.obs_horizon)

                for t in range(args.inf_steps):
                    # TODO: read state, update buffer, stack, normalize, predict, unnormalize, execute
                    raise NotImplementedError

                ep_states_list.append(np.asarray(states, dtype=np.float32))
                ep_actions_list.append(np.asarray(actions, dtype=np.float32))

                plot_3d_positions(np.array(eefs)[:, :3])

            np.savez(
                args.out,
                states=np.array(ep_states_list, dtype=object),
                actions=np.array(ep_actions_list, dtype=object),
                action_type="delta_joint_angles",
                unit="radians",
            )
            print(f"\nDataset saved to {args.out}")

        finally:
            disconnect_arm(arm)


if __name__ == "__main__":
    main()
