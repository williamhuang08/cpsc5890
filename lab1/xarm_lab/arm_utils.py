"""
Utilities for connecting to and querying the xArm.

You should NOT modify function signatures.
Fill in TODOs only.
"""

from dataclasses import dataclass
from typing import List
from xarm.wrapper import XArmAPI


@dataclass
class ArmConfig:
    ip: str
    is_radian: bool = True


def connect_arm(cfg: ArmConfig) -> XArmAPI:
    """
    Connect to the robot and put it into a ready-to-move state.

    Steps you likely need:
    - create XArmAPI object
    - connect()
    - clear warnings / errors
    - enable motion
    - set mode / state

    Return:
        XArmAPI instance
    """
    # TODO: initialize arm
    arm = None

    # TODO: connect to robot

    # TODO: clear warnings / errors

    # TODO: enable motion

    # TODO: set mode/state if needed

    return arm


def get_joint_angles(arm: XArmAPI) -> List[float]:
    """
    Return current joint angles.
    """
    # TODO: call SDK API
    raise NotImplementedError


def get_tcp_pose(arm: XArmAPI) -> List[float]:
    """
    Return TCP pose as [x, y, z, roll, pitch, yaw].
    """
    # TODO: call SDK API
    raise NotImplementedError


def disconnect_arm(arm: XArmAPI) -> None:
    """
    Cleanly disconnect from robot.
    """
    # TODO: disconnect safely
    pass