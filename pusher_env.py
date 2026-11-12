"""Pusher-v5 environment with the custom reward and early termination on success."""

import gymnasium as gym
import numpy as np

from config import ENV_ID, SEED, SUCCESS_RADIUS
from reward import calculate_reward, planar_dist


# ------------------------------------------------------------
# Position helpers (work whether MuJoCo names are bodies or sites)
# ------------------------------------------------------------

def _named_pos(u, name):
    """World position of a body, falling back to a site with the same name."""
    try:
        return u.data.body(name).xpos.copy()
    except KeyError:
        return u.data.site(name).xpos.copy()


def get_tip(u):
    """Midpoint of the two fork prongs; falls back to the 'tips_arm' body."""
    try:
        left = u.data.geom("tip_arml").xpos
        right = u.data.geom("tip_armr").xpos
        return ((left + right) / 2.0).copy()
    except KeyError:
        return _named_pos(u, "tips_arm")


def get_positions(u):
    """Return (tip, object, goal) world positions."""
    return get_tip(u), _named_pos(u, "object"), _named_pos(u, "goal")


# ------------------------------------------------------------
# Wrapper
# ------------------------------------------------------------

class PusherWrapper(gym.Wrapper):
    """Replaces the default reward and ends the episode when the object reaches the goal."""

    def __init__(self, env):
        super().__init__(env)
        self.prev_tip = None
        self.prev_obj = None
        self.prev_goal = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_tip, self.prev_obj, self.prev_goal = get_positions(self.env.unwrapped)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        u = self.env.unwrapped

        tip, obj, goal = get_positions(u)
        q = u.data.qpos[:7].copy()

        reward = calculate_reward(
            tip, obj, goal, action,
            self.prev_tip, self.prev_obj, self.prev_goal, q,
        )

        distance = planar_dist(obj, goal)
        success = distance < SUCCESS_RADIUS
        if success:
            terminated = True            # early termination on success

        info["is_success"] = bool(success)
        info["distance_to_goal"] = distance

        self.prev_tip, self.prev_obj, self.prev_goal = tip, obj, goal
        return obs, reward, terminated, truncated, info


def make_env(render_mode=None):
    """Create one wrapped Pusher environment."""
    return PusherWrapper(gym.make(ENV_ID, render_mode=render_mode))


# ------------------------------------------------------------
# Sanity check (run once before training)
# ------------------------------------------------------------

def sanity_check():
    """Print which tip source is used, the joint names and the start positions."""
    env = make_env()
    env.reset(seed=SEED)
    u = env.unwrapped
    tip, obj, goal = get_positions(u)
    try:
        u.data.geom("tip_arml")
        tip_source = "midpoint of fork geoms tip_arml / tip_armr"
    except KeyError:
        tip_source = "'tips_arm' body/site"
    print("Tip source :", tip_source)
    print("Joints 0-6 :", [u.model.joint(i).name for i in range(7)])
    print("tip, obj, goal:", np.round(tip, 3), np.round(obj, 3), np.round(goal, 3))
    env.close()
