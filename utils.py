"""Training utilities: per-episode logging callback and plotting."""

import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from config import PLOT_PATH, WINDOW


# ------------------------------------------------------------
# Callback: raw episode reward, length and success
# ------------------------------------------------------------

class EpisodeLogger(BaseCallback):
    """Records the real (un-normalized) return, length and success of every episode."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self._ret = None
        self._len = None

    def _on_training_start(self):
        n = self.training_env.num_envs
        self._ret = [0.0] * n
        self._len = [0] * n

    def _on_step(self):
        # Original (un-normalized) rewards, from underneath VecNormalize
        rewards = self.training_env.get_original_reward()
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i in range(self.training_env.num_envs):
            self._ret[i] += float(rewards[i])
            self._len[i] += 1
            if dones[i]:
                self.episode_rewards.append(self._ret[i])
                self.episode_lengths.append(self._len[i])
                self.episode_success.append(float(infos[i].get("is_success", False)))
                self._ret[i] = 0.0
                self._len[i] = 0
        return True


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    ma = np.convolve(x, np.ones(window) / window, mode="valid")
    return np.arange(window - 1, len(x)), ma


def plot_training(logger, window=WINDOW, path=PLOT_PATH):
    """Plot reward, success rate and episode length against episode number."""
    panels = [
        (logger.episode_rewards, "Reward", "Reward vs Episode"),
        (logger.episode_success, "Success rate", "Success vs Episode"),
        (logger.episode_lengths, "Steps", "Episode length vs Episode"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (data, ylabel, title) in zip(axes, panels):
        ax.plot(data, alpha=0.25, label="per episode")
        if len(data) >= window:
            x, ma = moving_average(data, window)
            ax.plot(x, ma, linewidth=2, label=f"{window}-episode average")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved training plot to {os.path.abspath(path)}")
    plt.show()
