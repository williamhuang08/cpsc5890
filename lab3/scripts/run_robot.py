# run_policy_on_robot.py
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro

from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera

from policy import UniversalPolicy


@dataclass
class Args:
    # robot + camera nodes are running on THIS machine
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "gadget.stdusr.yale.internal"
    hz: int = 30

    # debug / dev
    mock: bool = False
    max_steps: int = 10_000
    print_every: int = 30  # steps


def main(args: Args):
    # --- env wiring ---
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        camera_clients = {
            "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    # --- policy ---
    policy = UniversalPolicy()
    policy.reset()

    dt = 1.0 / args.hz

    # RL-style loop
    obs = env.get_obs()
    obs = env.step([0,-0.8,0,0.8,0,1.0,0,0])
    time.sleep(5)
    t0 = time.time()
    print("HELLO")

    try:
        for step in range(args.max_steps):
            out = policy.step(obs)
            action = np.asarray(out.action, dtype=np.float32)
            for i in range(4):
                curr_action_grip = action[i][-1]
                if curr_action_grip >= 0.25:
                    action[i][-1] = 1
                else:
                    action[i][-1] = 0 
                obs = env.step(action[i])
                # time.sleep(0.5)

            if args.print_every > 0 and (step % args.print_every == 0):
                elapsed = time.time() - t0
                jp = obs["joint_positions"]
                print(f"[step {step:05d} | {elapsed:6.1f}s] joints[0:3]={np.array(jp)[:3]}")

            # keep approximate rate (RobotEnv may already sleep; this is just extra)
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")


if __name__ == "__main__":
    main(tyro.cli(Args))