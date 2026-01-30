"""
Forward and inverse kinematics helpers.
"""

from typing import List
from xarm.wrapper import XArmAPI


def fk_from_joints(arm: XArmAPI, joints: List[float]) -> List[float]:
    """
    Compute forward kinematics from joint angles.

    Hint:
    - Look for a function with 'forward_kinematics' in the SDK
    """
    # TODO
    raise NotImplementedError


def ik_from_pose(arm: XArmAPI, pose: List[float]) -> List[float]:
    """
    Compute inverse kinematics for a desired TCP pose.

    pose = [x, y, z, roll, pitch, yaw]
    """
    # TODO
    raise NotImplementedError