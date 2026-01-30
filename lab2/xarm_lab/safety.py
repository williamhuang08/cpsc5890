from xarm.wrapper import XArmAPI


def enable_basic_safety(arm: XArmAPI):
    try:
        arm.set_collision_sensitivity(3)
    except Exception:
        pass

    try:
        arm.set_self_collision_detection(True)
    except Exception:
        pass


def clear_faults(arm: XArmAPI):
    try:
        arm.clean_warn()
    except Exception:
        pass

    try:
        arm.clean_error()
    except Exception:
        pass