# example.py
import argparse
import os
from pathlib import Path

import gymnasium as gym
import gym_xarm  # registers envs on import

import time


ENVS = [
    "gym_xarm/XarmLift-v0",
    "gym_xarm/XarmReach-v0",
    "gym_xarm/XarmPickPlaceDense-v0",
    "gym_xarm/XarmPickPlaceSemi-v0",
    "gym_xarm/XarmPickPlaceSparse-v0",
]

def run_env(env_id: str, render_mode: str, steps: int, episodes: int, out_dir: str | None):
    print(f"\n=== {env_id} | render_mode={render_mode} | steps={steps} | episodes={episodes} ===")

    env = gym.make(env_id, render_mode=render_mode)
    frames = []

    obs, info = env.reset()
    # Force first render
    if render_mode in ("human", "rgb_array"):
        first = env.render()
        if render_mode == "rgb_array" and first is not None:
            frames.append(first)

    ep = 0
    t = 0
    while ep < episodes and t < steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if render_mode in ("human", "rgb_array"):
            frame = env.render()
            if render_mode == "rgb_array" and frame is not None:
                frames.append(frame)
            if render_mode == "human":
                time.sleep(1 / 60.0)  # let the window update

        if terminated or truncated:
            ep += 1
            obs, info = env.reset()

        t += 1

    env.close()
    # (video saving code unchanged)
    print(f"Done: {env_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        choices=["human", "rgb_array", "none"],
        default="human",
        help="Render mode. Use rgb_array for headless + video saving.",
    )
    parser.add_argument("--steps", type=int, default=500, help="Max steps per env test")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes to run per env")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="If set and render=rgb_array, save an MP4 per env to this directory.",
    )
    args = parser.parse_args()

    render_mode = None if args.render == "none" else args.render

    print("Testing environments:")
    for e in ENVS:
        print(" -", e)

    for env_id in ENVS:
        run_env(
            env_id=env_id,
            render_mode=render_mode if render_mode is not None else "rgb_array",
            steps=args.steps,
            episodes=args.episodes,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()