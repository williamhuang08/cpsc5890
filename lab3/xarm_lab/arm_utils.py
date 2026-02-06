from dataclasses import dataclass
from typing import List
from xarm.wrapper import XArmAPI


@dataclass
class ArmConfig:
    ip: str
    is_radian: bool = True


def connect_arm(cfg: ArmConfig) -> XArmAPI:
    arm = XArmAPI(cfg.ip, is_radian=cfg.is_radian)
    arm.connect()

    try:
        arm.clean_warn()
    except Exception:
        pass

    try:
        arm.clean_error()
    except Exception:
        pass

    arm.motion_enable(True)

    try:
        arm.set_mode(0)
        arm.set_state(0)
    except Exception:
        pass

    return arm


def get_joint_angles(arm: XArmAPI) -> List[float]:
    code, angles = arm.get_servo_angle(is_radian=arm._is_radian)
    if code != 0:
        raise RuntimeError(f"get_servo_angle failed: {code}")
    return angles


def get_tcp_pose(arm: XArmAPI) -> List[float]:
    code, pose = arm.get_position(is_radian=arm._is_radian)
    if code != 0:
        raise RuntimeError(f"get_position failed: {code}")
    return pose


def disconnect_arm(arm: XArmAPI) -> None:
    try:
        arm.disconnect()
    except Exception:
        pass

def get_gripper_position(arm: XArmAPI) -> float:
    code, pos = arm.get_gripper_position()
    if code != 0:
        raise RuntimeError(f"get_gripper_position failed: {code}")
    return pos


