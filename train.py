"""
Training Script — Drone Hover Agent
=====================================
Uses PPO (Proximal Policy Optimisation) from Stable-Baselines3.

PPO is a good default for drone control because:
  - Works well in continuous action spaces (motor throttles)
  - Stable training — won't diverge as easily as DDPG/TD3
  - Handles the noisy, high-frequency nature of flight control

Usage:
    python training/train.py                  # train from scratch
    python training/train.py --resume         # continue from last checkpoint
    python training/train.py --timesteps 1e6  # custom training length
"""

import os
import sys
import argparse
import numpy as np

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from envs.drone_env import DroneEnv

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR      = os.path.join(os.path.dirname(__file__), "..", "models")
LOGS_DIR        = os.path.join(os.path.dirname(__file__), "..", "logs")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "drone_ppo_checkpoint")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)


def make_env(target_pos=None):
    """Factory for creating monitored drone environments."""
    def _init():
        env = DroneEnv(render_mode=None, target_pos=target_pos)
        env = Monitor(env)
        return env
    return _init


def train(timesteps: int = 500_000, resume: bool = False, n_envs: int = 4):
    """
    Train the hover agent.

    Args:
        timesteps: Total environment steps to train for.
                   500k = ~15 mins on a modern CPU
                   1M+  = better hover stability
        resume:    Load existing model and continue training.
        n_envs:    Parallel environments (speeds up data collection).
                   Reduce to 1 if you get memory errors.
    """
    print(f"\n{'='*50}")
    print(f"  Drone RL Training")
    print(f"  Timesteps : {timesteps:,}")
    print(f"  Envs      : {n_envs}")
    print(f"  Resume    : {resume}")
    print(f"{'='*50}\n")

    # ── Create vectorised training environments ───────────────────────────────
    # Each env targets the same hover point: 1m above origin
    train_env = make_vec_env(
        make_env(target_pos=[0.0, 0.0, 1.0]),
        n_envs=n_envs,
    )

    # Single env for evaluation (no parallelism needed)
    eval_env = Monitor(DroneEnv(render_mode=None, target_pos=[0.0, 0.0, 1.0]))

    # ── PPO Hyperparameters ───────────────────────────────────────────────────
    # These are tuned for drone control — feel free to experiment
    ppo_kwargs = dict(
        policy         = "MlpPolicy",   # standard MLP neural network policy
        env            = train_env,
        learning_rate  = 3e-4,          # Adam LR — reduce if unstable
        n_steps        = 1024,          # steps per env before update
        batch_size     = 64,            # minibatch size
        n_epochs       = 10,            # PPO update epochs per rollout
        gamma          = 0.99,          # discount factor
        gae_lambda     = 0.95,          # GAE smoothing
        clip_range     = 0.2,           # PPO clip parameter
        ent_coef       = 0.01,          # entropy bonus (encourages exploration)
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        verbose        = 1,
        tensorboard_log= LOGS_DIR,
        device         = "cpu",         # change to "cuda" if you have a GPU
        policy_kwargs  = dict(
            net_arch = [256, 256],      # 2 hidden layers, 256 units each
        ),
    )

    # ── Load or create model ──────────────────────────────────────────────────
    checkpoint_file = CHECKPOINT_PATH + ".zip"
    if resume and os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        model = PPO.load(CHECKPOINT_PATH, env=train_env, **{
            k: v for k, v in ppo_kwargs.items()
            if k not in ["policy", "env"]
        })
    else:
        print("Starting fresh training run...")
        model = PPO(**ppo_kwargs)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,           # save every 50k steps
        save_path   = MODELS_DIR,
        name_prefix = "drone_ppo",
        verbose     = 1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = BEST_MODEL_PATH,
        log_path             = LOGS_DIR,
        eval_freq            = 25_000,  # evaluate every 25k steps
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = timesteps,
        callback            = callbacks,
        reset_num_timesteps = not resume,
    )

    # Save final model
    final_path = os.path.join(MODELS_DIR, "drone_ppo_final")
    model.save(final_path)
    print(f"\nTraining complete. Model saved to: {final_path}")

    train_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train drone hover agent")
    parser.add_argument("--timesteps", type=float, default=500_000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    args = parser.parse_args()

    train(
        timesteps = int(args.timesteps),
        resume    = args.resume,
        n_envs    = args.envs,
    )
