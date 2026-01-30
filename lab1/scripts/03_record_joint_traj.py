"""
Record a trajectory by using Joint Teaching (compliant / zero-gravity) mode.

Workflow (high level):
  1) connect to robot
  2) enable motion + ready state
  3) set mode to Joint Teaching (Mode 2)
  4) start_record_trajectory()
  5) user physically guides the robot for N seconds
  6) stop_record_trajectory()
  7) save_record_trajectory(<out>)
  8) set mode back to normal (Mode 0)

IMPORTANT:
- Keep workspace clear
- Be ready to hit E-stop
- After recording, ALWAYS return to Mode 0
"""

import argparse
import time
from pathlib import Path

from xarm.wrapper import XArmAPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True, help="Robot IP, e.g. 192.168.1.123")
    ap.add_argument("--out", required=True, help="Output name .traj, e.g. demo.traj")
    ap.add_argument("--seconds", type=float, default=8.0, help="How long to record while guiding the robot")
    args = ap.parse_args()

    # TODO: initialize XArmAPI (recommend is_radian=True)
    arm = None

    # TODO: connect

    try:
        # TODO: enable motion
        # TODO: set state ready

        # TODO: set mode to joint teaching (Mode 2)

        # TODO: start_record_trajectory()

        print(f"[INFO] Recording for {args.seconds:.1f}s. Guide the robot NOW.")
        time.sleep(args.seconds)

        # TODO: stop_record_trajectory()

        # TODO: save_record_trajectory(str(out_path))

        print(f"[OK] Saved trajectory to: {out_path}")

    finally:
        # TODO: set mode back to normal (Mode 0)
        # TODO: set state ready
        # TODO: disconnect
        pass


if __name__ == "__main__":
    main()
