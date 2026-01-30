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
    One-shot FSM pick-and-place controller.
    Each stage is visited exactly once.
    No interpolation: each stage outputs a single target.
    """

    # ---------------- persistent FSM state ----------------
    if not hasattr(policy, "stage"):
        policy.stage = "PICK_HOVER"

    # ---------------- read state ----------------
    q = np.array(get_joint_angles(arm), float)
    pose = np.array(get_tcp_pose(arm), float)

    # ---------------- waypoints ----------------
    pick_hover  = np.array(pick_pose, float);  pick_hover[2]  += hover_height
    pick_down   = np.array(pick_pose, float)

    place_hover = np.array(place_pose, float); place_hover[2] += hover_height
    place_down  = np.array(place_pose, float)

    # ---------------- FSM logic ----------------
    if policy.stage == "PICK_HOVER":
        target_pose = pick_hover
        gripper_cmd = 600
        policy.stage = "PICK_DOWN"

    elif policy.stage == "PICK_DOWN":
        target_pose = pick_down
        gripper_cmd = 600
        policy.stage = "GRASP"

    elif policy.stage == "GRASP":
        # close gripper, no motion
        policy.stage = "PLACE_HOVER"
        return np.hstack([np.zeros(7), 300]), False

    elif policy.stage == "PLACE_HOVER":
        target_pose = place_hover
        gripper_cmd = 300
        policy.stage = "PLACE_DOWN"

    elif policy.stage == "PLACE_DOWN":
        target_pose = place_down
        gripper_cmd = 300
        policy.stage = "RELEASE"

    elif policy.stage == "RELEASE":
        # open gripper, no motion
        return np.hstack([np.zeros(7), 600]), True

    # ---------------- IK → joint delta ----------------
    q_next = np.array(ik_from_pose(arm, target_pose.tolist()), float)
    dq = _angle_wrap_pi(q_next - q)

    return np.hstack([dq, gripper_cmd]), False


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--out", default="asset/demo.npz")
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
                    wait=True,
                    is_radian=True
                )
                arm.set_gripper_position(action[-1], wait=True, speed=0.1)

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
