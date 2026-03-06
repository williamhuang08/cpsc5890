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

from scripts.policy import UniversalPolicy



@dataclass
class Args:
    # robot + camera nodes are running on THIS machine
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "gadget.stdusr.yale.internal"
    hz: int = 10

    # debug / dev
    mock: bool = False
    max_steps: int = 10_000
    print_every: int = 10  # steps


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
    t0 = time.time()

    try:
        for step in range(args.max_steps):
            action_seq = policy.step(obs)                 # (K,8) or (1,8)
            action_seq = np.asarray(action_seq, dtype=np.float32)

            if action_seq.ndim == 1:                      # (8,) -> (1,8)
                action_seq = action_seq[None, :]

            # execute the whole chunk
            for a in action_seq:
                a = np.asarray(a, dtype=np.float32).reshape(-1)  # ensure (8,)
                a = np.asarray([ 0.024528, -0.885107, -0.065961,  0.704097, -0.001534,  0.909635,  0.027596, -0.      ])
                obs = env.step(a)

                if args.print_every > 0 and (step % args.print_every == 0):
                    elapsed = time.time() - t0
                    jp = obs["joint_positions"]
                    print(f"[step {step:05d} | {elapsed:6.1f}s] joints[0:3]={np.array(jp)[:3]}")

                time.sleep(dt)

    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")



if __name__ == "__main__":
    main(tyro.cli(Args))