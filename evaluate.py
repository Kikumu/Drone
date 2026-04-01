"""
Evaluate a trained drone agent with visual rendering.
Run this after training to watch your agent fly.

Usage:
    python training/evaluate.py                        # load best model
    python training/evaluate.py --model models/drone_ppo_final
    python training/evaluate.py --episodes 5
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from envs.drone_env import DroneEnv

MODELS_DIR      = os.path.join(os.path.dirname(__file__), "..", "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model", "best_model")


def evaluate(model_path: str = None, episodes: int = 3, target_pos=None):
    model_path = model_path or BEST_MODEL_PATH

    if not os.path.exists(model_path + ".zip"):
        print(f"No model found at {model_path}.zip")
        print("Train first: python training/train.py")
        return

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    target = target_pos or [0.0, 0.0, 1.0]
    env    = DroneEnv(render_mode="human", target_pos=target)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps        = 0
        done         = False

        print(f"\nEpisode {ep + 1}/{episodes}")
        print(f"Target: {target}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps        += 1
            done          = terminated or truncated

            # Print position every 50 steps
            if steps % 50 == 0:
                pos  = obs[0:3]
                dist = obs[17]
                print(f"  Step {steps:4d} | pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                      f" | dist_to_target={dist:.3f}m | reward={total_reward:.1f}")

        status = "CRASHED/FLIPPED" if terminated else "TIME LIMIT"
        print(f"  Episode ended: {status} | Steps: {steps} | Total reward: {total_reward:.1f}")

    env.close()
    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained drone agent")
    parser.add_argument("--model",    type=str, default=None,
                        help="Path to model (without .zip)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--target",   type=float, nargs=3, default=[0.0, 0.0, 1.0],
                        metavar=("X", "Y", "Z"),
                        help="Target position (default: 0 0 1)")
    args = parser.parse_args()

    evaluate(
        model_path = args.model,
        episodes   = args.episodes,
        target_pos = args.target,
    )
