"""
Compare FK computed in code vs TCP pose reported by GUI.
"""

import argparse
import numpy as np

from xarm_lab.arm_utils import connect_arm, disconnect_arm, get_joint_angles, get_tcp_pose, ArmConfig
from xarm_lab.kinematics import fk_from_joints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    args = ap.parse_args()

    arm = connect_arm(ArmConfig(ip=args.ip))

    try:
        # TODO: read joint angles
        q = None

        # TODO: get GUI TCP pose
        gui_pose = None

        # TODO: compute FK pose
        fk_pose = None

        # TODO: compute difference
        diff = None

        print("Joint angles:", q)
        print("GUI pose:", gui_pose)
        print("FK pose:", fk_pose)
        print("Position error (mm):", np.linalg.norm(diff[:3]))
        print("Orientation error (rad):", np.linalg.norm(diff[3:]))

    finally:
        disconnect_arm(arm)


if __name__ == "__main__":
    main()