import argparse
from xarm_lab.arm_utils import connect_arm, disconnect_arm, get_joint_angles, get_tcp_pose, ArmConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    args = ap.parse_args()

    arm = connect_arm(ArmConfig(ip=args.ip))

    try:
        # TODO: print joint angles
        # TODO: print TCP pose
        pass
    finally:
        disconnect_arm(arm)

if __name__ == "__main__":
    main()