"""
Solve IK for a desired pose and move robot safely.
"""

import argparse
from xarm_lab.arm_utils import connect_arm, disconnect_arm, ArmConfig
from xarm_lab.kinematics import ik_from_pose
from xarm_lab.safety import enable_basic_safety, clear_faults


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--z", type=float, required=True)
    ap.add_argument("--roll", type=float, required=True)
    ap.add_argument("--pitch", type=float, required=True)
    ap.add_argument("--yaw", type=float, required=True)
    args = ap.parse_args()

    arm = connect_arm(ArmConfig(ip=args.ip))

    try:
        # TODO: clear faults
        # TODO: enable safety

        target_pose = [args.x, args.y, args.z, args.roll, args.pitch, args.yaw]

        # TODO: compute IK
        q_sol = None

        # TODO: move robot using joint command
        # IMPORTANT: keep speed LOW

    finally:
        disconnect_arm(arm)


if __name__ == "__main__":
    main()