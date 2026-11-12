"""Train PPO on the Pusher environment with the custom reward.

Usage:
    python train.py

Saves the model (MODEL_NAME.zip), the observation/reward statistics
(VECNORM_NAME) and the training curves (PLOT_PATH).
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from config import (
    BATCH_SIZE,
    CLIP_RANGE,
    GAE_LAMBDA,
    GAMMA,
    LEARNING_RATE,
    MODEL_NAME,
    N_ENVS,
    N_EPOCHS,
    N_STEPS,
    SEED,
    TOTAL_TIMESTEPS,
    VECNORM_NAME,
)
from pusher_env import make_env, sanity_check
from utils import EpisodeLogger, plot_training


def train():
    """Train the agent, save it, and return the episode logger."""
    venv = make_vec_env(make_env, n_envs=N_ENVS, seed=SEED)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        seed=SEED,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
    )
    logger = EpisodeLogger()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger)

    model.save(MODEL_NAME)
    venv.save(VECNORM_NAME)
    venv.close()
    return logger


if __name__ == "__main__":
    print("--- 0. SANITY CHECK ---")
    sanity_check()

    print("\n--- 1. TRAINING ---")
    logger = train()
    print("--- TRAINING COMPLETE & SAVED ---")

    print("\n--- 2. PLOTS ---")
    plot_training(logger)

    print("\nModel        :", MODEL_NAME + ".zip")
    print("Normalization:", VECNORM_NAME)
    print("Next         : python evaluate.py")
