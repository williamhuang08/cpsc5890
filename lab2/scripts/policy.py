import argparse
import numpy as np
import json
import time

from xarm_lab.arm_utils import connect_arm, disconnect_arm, ArmConfig, get_joint_angles, get_tcp_pose, get_gripper_position
from xarm_lab.safety import enable_basic_safety, clear_faults
from xarm_lab.kinematics import ik_from_pose
import time
import numpy as np

def policy(arm):
    """
    Simple hand-coded policy for generating joint-space motions.

    The policy should return an action of shape (8,):
      - action[:7]  : delta joint angles (radians)
      - action[-1]  : gripper position

    Suggested motions (choose ONE):
      - Move a single joint back and forth (sinusoidal)
      - Move two joints in a coordinated pattern (circle-like motion)
      - Piecewise motion that traces a square in joint space

    Keep delta magnitudes SMALL to ensure safety.
    """

    # TODO 1: Read current robot state if needed
    #   - current joint angles
    #   - current gripper position
    # Hint: use get_joint_angles(arm), get_gripper_position(arm)

    # TODO 2: Define a time variable or step counter
    # Hint: time.time() or a global counter

    # TODO 3: Initialize a 7D delta joint vector (Δq)
    # Example: dq = np.zeros(7)

    # TODO 4: Modify one or more joint deltas to create motion
    # Examples:
    #   - sinusoidal motion for smooth back-and-forth movement
    #   - piecewise constant deltas for square-like motion
    #   - linear ramp forward then backward

    # TODO 5: Choose a gripper command
    #   - keep it fixed
    #   - or periodically open/close

    # TODO 6: Concatenate joint deltas and gripper command
    # Return a numpy array of shape (8,)

    raise NotImplementedError("Implement a simple policy for demonstration")


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    args = ap.parse_args()

    arm = connect_arm(ArmConfig(ip=args.ip))

    try:
        clear_faults(arm)
        enable_basic_safety(arm)

        code = arm.set_gripper_mode(0)
        code = arm.set_gripper_enable(True)
        code = arm.set_gripper_speed(5000)

        print("Sending to Init Position")

        code, initial_joints = arm.get_initial_point()
        arm.set_servo_angle(
            angle=initial_joints,
            speed=20.0,
            wait=True,
            is_radian=False
        )

        arm.set_gripper_position(600, wait=True, speed=0.1)
        
        while True:

            current_joint = get_joint_angles(arm)
            current_gripper_pos = get_gripper_position(arm)

            action = policy(arm)

            # Execute motion
            arm.set_servo_angle(
                angle=(action[:7] + current_joint).tolist(),
                speed=0.5,
                wait=True,
                is_radian=True
            )
            arm.set_gripper_position(action[-1], wait=True, speed=0.1)

    finally:
        disconnect_arm(arm)


if __name__ == "__main__":
    main()
