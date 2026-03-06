"""
Generic RL training script using Stable-Baselines3.
Supports PPO and SAC on any Gym-compatible environment.

Designed for:
- Comparing PPO vs SAC
- Testing reward functions
- Plotting reward + success curves
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import torch
# import matplotlib.pyplot as plt

# RL algorithms
from stable_baselines3 import PPO, SAC

# Utilities
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import gym_xarm


# ============================================================
# Callback for Logging Episode Metrics
# ============================================================
class MetricsCallback(BaseCallback):
    """
    Custom callback to record:
    - Episode rewards
    - Success rates (if environment provides 'is_success')

    Called automatically during training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_success = []

    def _on_step(self):
        """
        This function runs at every environment step.
        We check if an episode has finished (done=True),
        then log metrics.
        """
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if done:
                ep_info = infos[i]

                # Stable-Baselines3 Monitor wrapper
                # stores episode reward under "episode"
                if "episode" in ep_info:
                    self.episode_rewards.append(ep_info["episode"]["r"])

                # Optional success metric
                if "is_success" in ep_info:
                    self.episode_success.append(float(ep_info["is_success"]))

        return True

def make_env(env_id, rank=0):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=rank)
        return env
    return _init

# ============================================================
# Model Factory
# ============================================================
def make_model(algo, env, args):
    """
    Creates PPO or SAC model with user-defined hyperparameters.
    """

    # Define neural network architecture for policy
    policy_kwargs = dict(
        net_arch=args.policy_arch,   # e.g., [256, 256]
        activation_fn=torch.nn.ReLU  # activation function
    )

    if algo == "ppo":
        # PPO is on-policy
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            gamma=args.gamma,            # discount factor
            clip_range=args.clip_range,  # PPO clipping parameter
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

    elif algo == "sac":
        # Convert ent_coef if numeric
        ent_coef = args.ent_coef
        if ent_coef not in ["auto"] and not ent_coef.startswith("auto"):
            ent_coef = float(ent_coef)

        # SAC is off-policy with entropy regularization
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            gamma=args.gamma,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_starts=10_000,
            tau=0.05,
            batch_size=256,
        )

    else:
        raise ValueError("Unsupported algorithm")

    return model

# ============================================================
# Evaluation Function
# ============================================================
def evaluate(model, env, n_rollouts=10):
    """
    Runs deterministic rollouts for evaluation.
    Returns:
        mean_reward
        mean_success (if available)
    """

    rewards = []
    success = []
    max_steps = 500
    for _ in range(n_rollouts):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done and steps < max_steps:
            steps += 1
            # Deterministic=True → no exploration noise
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        rewards.append(ep_reward)

        # Optional success flag
        if "is_success" in info:
            success.append(float(info["is_success"]))

    mean_reward = np.mean(rewards)
    mean_success = np.mean(success) if success else None

    return mean_reward, mean_success


# ============================================================
# Main Training Function
# ============================================================
def main(args):

    os.makedirs(args.log_dir, exist_ok=True)

    # --------------------------------------------------------
    # Create Gym environment
    # --------------------------------------------------------
    # This initializes the simulation environment specified by
    # --env (e.g., "Pendulum-v1", "HalfCheetah-v4", etc.).
    # The environment defines:
    #   - Observation space (state representation)
    #   - Action space (control inputs)
    #   - Reward function
    #   - Termination conditions
    # The RL algorithm will interact with this environment
    # during training to collect experience.
    env = DummyVecEnv([make_env(args.env, i) for i in range(args.n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------
    model = make_model(args.algo, env, args)

    callback = MetricsCallback()

    print(f"Training {args.algo.upper()} on {args.env}")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback
    )

    # Save trained model
    model.save(os.path.join(args.log_dir, f"{args.algo}_{args.env}"))

    # --------------------------------------------------------
    # Evaluate after training
    # --------------------------------------------------------
    env = gym.make(args.env)
    mean_reward, std_reward = evaluate(
        model,
        env,
        10,
    )

    print("Mean reward:", mean_reward)
    print("Std reward:", std_reward)
    print("Mean success: not computed in evaluate_policy")

    # --------------------------------------------------------
    # Plot Reward Curve
    # --------------------------------------------------------
    plt.figure()
    plt.plot(callback.episode_rewards)
    plt.title("Episode Reward vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(args.log_dir, "reward_curve.png"))

    # --------------------------------------------------------
    # Plot Success Curve (if available)
    # --------------------------------------------------------
    if callback.episode_success:
        plt.figure()
        plt.plot(callback.episode_success)
        plt.title("Success Rate vs Episode")
        plt.xlabel("Episode")
        plt.ylabel("Success")
        plt.savefig(os.path.join(args.log_dir, "success_curve.png"))

    print("Plots saved to:", args.log_dir)


# ============================================================
# Argument Parser
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Number of envs
    parser.add_argument("--n_envs", type=int, default=4)

    # Environment name (Gym registry)
    parser.add_argument("--env", type=str, required=True)

    # Algorithm choice
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], required=True)

    # Training length
    parser.add_argument("--timesteps", type=int, default=2_000_000)

    # Learning rate
    parser.add_argument("--lr", type=float, default=3e-4)

    # Discount factor
    parser.add_argument("--gamma", type=float, default=0.95)

    # PPO clip parameter: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    parser.add_argument("--clip_range", type=float, default=0.2)

    # SAC entropy regularization: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
    parser.add_argument(
        "--ent_coef",
        type=str,
        default="auto",
        help="Entropy coefficient for SAC. Examples: 'auto', 'auto_0.1', '0.2'"
    )

    # Policy network architecture: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    parser.add_argument(
        "--policy_arch",
        nargs="+",
        type=int,
        default=[256, 256],
        help="Hidden layer sizes"
    )

    # Output directory
    parser.add_argument("--log_dir", type=str, default="asset/")

    args = parser.parse_args()

    main(args)