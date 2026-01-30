import argparse
import numpy as np
import json
import time

from xarm_lab.arm_utils import connect_arm, disconnect_arm, ArmConfig, get_joint_angles, get_tcp_pose, get_gripper_position
from xarm_lab.safety import enable_basic_safety, clear_faults
from xarm_lab.kinematics import ik_from_pose
import time
import numpy as np


def _angle_wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def policy(
    arm,
    pick_pose=[200, 0, 200, 3.1415, 0, 0],
    place_pose=[300, 300, 300, 3.1415, 0, 0],
    hover_height=100,
):
    """
    Open-loop FSM pick-and-place controller.
    - Streaming (wait=False)
    - Proportional joint-space control
    - Error-tolerant state transitions
    """

    # ---------------- parameters ----------------
    Kp = 0.2            # proportional gain
    dq_max = 0.05       # rad / step clamp
    q_tol = 5        # rad ≈ 1 deg
    grip_hold_time = 1.0  # seconds

    # ---------------- persistent FSM state ----------------
    if not hasattr(policy, "stage"):
        policy.stage = "PICK_HOVER"
        policy.stage_t0 = None

    # ---------------- read state ----------------
    q = np.array(get_joint_angles(arm), dtype=float)
    pose = np.array(get_tcp_pose(arm), dtype=float)
    gripper_pos = np.array(get_gripper_position(arm), dtype=float)

    # ---------------- waypoints ----------------
    pick_hover  = np.array(pick_pose, dtype=float)
    pick_hover[2] += hover_height
    pick_down = np.array(pick_pose, dtype=float)

    place_hover = np.array(place_pose, dtype=float)
    place_hover[2] += hover_height
    place_down = np.array(place_pose, dtype=float)

    # ---------------- helpers ----------------
    def joint_step_to(target_pose):
        q_target = np.array(ik_from_pose(arm, target_pose.tolist()), dtype=float)
        err = _angle_wrap_pi(q_target - q)
        dq = Kp * err
        dq = np.clip(dq, -dq_max, dq_max)
        return dq, np.linalg.norm(err)

    # ---------------- FSM logic ----------------

    # ---- PICK HOVER ----
    if policy.stage == "PICK_HOVER":
        dq, err = joint_step_to(pick_hover)
        if np.linalg.norm(pose[:3] - pick_hover[:3]) < q_tol:
            policy.stage = "PICK_DOWN"
        return np.hstack([dq, 600]), False

    # ---- PICK DOWN ----
    if policy.stage == "PICK_DOWN":
        dq, err = joint_step_to(pick_down)
        if np.linalg.norm(pose[:3] - pick_down[:3]) < q_tol:
            policy.stage = "GRASP"
            policy.stage_t0 = time.time()
        return np.hstack([dq, 600]), False


    # ---- GRASP ----
    if policy.stage == "GRASP":
        if time.time() - policy.stage_t0 >= grip_hold_time:
            policy.stage = "PICK_UP"
        return np.hstack([np.zeros(7), 300]), False
    
    # ---- PICK UP ----
    if policy.stage == "PICK_UP":
        dq, err = joint_step_to(pick_hover)
        if np.linalg.norm(pose[:3] - pick_hover[:3]) < q_tol:
            policy.stage = "PLACE_HOVER"
            policy.stage_t0 = time.time()
        return np.hstack([dq, 300]), False

    # ---- PLACE HOVER ----
    if policy.stage == "PLACE_HOVER":
        dq, err = joint_step_to(place_hover)
        if np.linalg.norm(pose[:3] - place_hover[:3]) < q_tol:
            policy.stage = "PLACE_DOWN"
        return np.hstack([dq, 300]), False

    # ---- PLACE DOWN ----
    if policy.stage == "PLACE_DOWN":
        dq, err = joint_step_to(place_down)
        if np.linalg.norm(pose[:3] - place_down[:3]) < q_tol:
            policy.stage = "RELEASE"
            policy.stage_t0 = time.time()
        return np.hstack([dq, 300]), False

    # ---- RELEASE ----
    if policy.stage == "RELEASE":
        if time.time() - policy.stage_t0 >= grip_hold_time:
            return np.hstack([np.zeros(7), 600]), True
        return np.hstack([np.zeros(7), 600]), False


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--out", default="asset/demo_high_freq.npz")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    arm = connect_arm(ArmConfig(ip=args.ip))

    ep_states_list = []
    ep_actions_list = []

    try:
        clear_faults(arm)
        enable_basic_safety(arm)

        code = arm.set_gripper_mode(0)
        code = arm.set_gripper_enable(True)
        code = arm.set_gripper_speed(5000)

        print("\n=== Pick-and-Place Demonstration (IK → Δq) ===")

        for ep in range(args.episodes):

            code, initial_joints = arm.get_initial_point()
            arm.set_servo_angle(
                angle=initial_joints,
                speed=20.0,
                wait=True,
                is_radian=False
            )
            pose = get_tcp_pose(arm)
            pose[:3] += np.random.uniform(-5, 5, size=3)
            joint_angles = ik_from_pose(arm, pose)
            arm.set_servo_angle(
                angle=joint_angles,
                speed=20.0,
                wait=True,
                is_radian=True
            )
            arm.set_gripper_position(600, wait=True, speed=0.1)

            print(f"Episode {ep+1}: robot homed to {np.asarray(pose, dtype=float)}")

            states = []
            actions = []

            policy.stage = "PICK_HOVER"
            policy.seg_t0 = None
            policy.seg_p0 = None
            policy.seg_p1 = None
            policy.seg_T  = None
            
            while True:

                current_joint = get_joint_angles(arm)
                current_gripper_pos = get_gripper_position(arm)

                action, done = policy(arm)

                # Execute motion
                arm.set_servo_angle(
                    angle=(action[:7] + current_joint).tolist(),
                    speed=0.5,
                    wait=False,
                    is_radian=True
                )
                arm.set_gripper_position(action[-1], wait=False, speed=0.1)

                # Record transition: [q1..q7, gripper]
                state = np.concatenate([current_joint, [current_gripper_pos]])
                states.append(state)
                actions.append(action)

                if done:
                    break

            states = np.asarray(states, dtype=np.float32)
            ep_states_list.append(states)
            actions = np.asarray(actions, dtype=np.float32)
            ep_actions_list.append(actions)

        np.savez(
            args.out,
            states=np.array(ep_states_list, dtype=object),   # (E,) each item (Ti,7)
            actions=np.array(ep_actions_list, dtype=object), # (E,) each item (Ti,8)
            action_type="delta_joint_angles",
            unit="radians"
        )

        print(f"\nDataset saved to {args.out}")

    finally:
        disconnect_arm(arm)


if __name__ == "__main__":
    main()
