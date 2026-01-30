"""
Replay a recorded xArm trajectory file (.traj) using the SDK trajectory playback APIs.

Workflow:
  1) connect
  2) enable motion
  3) set normal mode (Mode 0)
  4) load_trajectory(<traj>)
  5) playback_trajectory()

IMPORTANT:
- Stand clear
- Keep speeds conservative (trajectory playback is controller-defined)
- Be ready to hit E-stop
"""

import argparse
from xarm.wrapper import XArmAPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--traj", required=True, help="Name of the .traj file recorded in teach mode")
    args = ap.parse_args()

    # TODO: initialize XArmAPI
    arm = None

    # TODO: connect

    try:
        # TODO: enable motion
        # TODO: set normal mode (Mode 0)
        # TODO: set state ready

        # TODO: load_trajectory(args.traj)
        # TODO: playback_trajectory()

        print("[OK] Playback command sent.")

    finally:
        # TODO: disconnect
        pass


if __name__ == "__main__":
    main()
