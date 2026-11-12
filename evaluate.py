"""Evaluate a trained model and record a demo video.

Usage:
    python evaluate.py

Needs the files saved by train.py (MODEL_NAME.zip and VECNORM_NAME).
"""

import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

from config import (
    MODEL_NAME,
    N_EVAL_EPISODES,
    VECNORM_NAME,
    VIDEO_FOLDER,
    VIDEO_LENGTH,
)
from pusher_env import make_env


def load_eval_env(venv):
    """Wrap an environment with the normalization statistics saved during training."""
    venv = VecNormalize.load(VECNORM_NAME, venv)
    venv.training = False        # freeze running statistics
    venv.norm_reward = False     # report real rewards
    return venv


def evaluate(n_episodes=N_EVAL_EPISODES):
    """Run the deterministic policy and report success rate, distance and length."""
    env = load_eval_env(DummyVecEnv([make_env]))
    model = PPO.load(MODEL_NAME, env=env)

    successes, final_dists, lengths = [], [], []
    obs = env.reset()
    steps = 0
    while len(successes) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        steps += 1
        if done[0]:
            successes.append(float(infos[0]["is_success"]))
            final_dists.append(infos[0]["distance_to_goal"])
            lengths.append(steps)
            steps = 0
    env.close()

    print(f"\nEvaluation over {n_episodes} episodes (deterministic policy)")
    print(f"  success rate        : {np.mean(successes):.2%}")
    print(f"  mean final distance : {np.mean(final_dists):.4f} m")
    print(f"  mean episode length : {np.mean(lengths):.1f} steps")


def record_video():
    """Record VIDEO_LENGTH steps of the trained policy (several episodes back to back)."""
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    env = load_eval_env(DummyVecEnv([lambda: make_env("rgb_array")]))
    env = VecVideoRecorder(
        env, VIDEO_FOLDER,
        record_video_trigger=lambda step: step == 0,
        video_length=VIDEO_LENGTH,
        name_prefix=MODEL_NAME,
    )
    model = PPO.load(MODEL_NAME, env=env)

    obs = env.reset()
    for _ in range(VIDEO_LENGTH):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)   # the vec env resets itself after each episode
    env.close()
    print(f"Video saved inside: {os.path.abspath(VIDEO_FOLDER)}")


if __name__ == "__main__":
    print("--- EVALUATION ---")
    evaluate()

    print("\n--- VIDEO ---")
    record_video()
