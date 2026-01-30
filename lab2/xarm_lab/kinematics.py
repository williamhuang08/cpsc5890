from typing import List
from xarm.wrapper import XArmAPI


def fk_from_joints(arm: XArmAPI, joints: List[float]) -> List[float]:
    code, pose = arm.get_forward_kinematics(
        joints,
        input_is_radian=arm._is_radian,
        return_is_radian=arm._is_radian
    )
    if code != 0:
        raise RuntimeError(f"FK failed: {code}")
    return pose


def ik_from_pose(arm: XArmAPI, pose: List[float]) -> List[float]:
    code, joints = arm.get_inverse_kinematics(
        pose,
        input_is_radian=arm._is_radian,
        return_is_radian=arm._is_radian
    )
    if code != 0:
        raise RuntimeError(f"IK failed: {code}")
    return joints

