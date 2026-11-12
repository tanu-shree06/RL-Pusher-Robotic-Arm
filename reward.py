"""Custom reward function for the Pusher task.

reward = W_PROGRESS * (progress of object toward goal)
       + W_TIP      * (progress of tip toward the pre-push point)
       - W_CTRL     * ||action||^2
       - (W_FLEX * q5^2 + W_ROLL * q6^2)
       + SUCCESS_BONUS   (when the object is within SUCCESS_RADIUS of the goal)
"""

import numpy as np

from config import (
    PRE_PUSH_OFFSET,
    SUCCESS_BONUS,
    SUCCESS_RADIUS,
    W_CTRL,
    W_FLEX,
    W_PROGRESS,
    W_ROLL,
    W_TIP,
)


def planar_dist(a, b):
    """Distance in the table plane (x, y) only."""
    return float(np.linalg.norm(a[:2] - b[:2]))


def pre_push_point(obj, goal):
    """Point PRE_PUSH_OFFSET behind the object, opposite the goal (table plane)."""
    direction = np.zeros(3)
    direction[:2] = (goal[:2] - obj[:2]) / (planar_dist(goal, obj) + 1e-8)
    return obj - direction * PRE_PUSH_OFFSET


def calculate_reward(tip, obj, goal, action, prev_tip, prev_obj, prev_goal, q):
    """Reward for one step, computed from the current and previous positions.

    Args:
        tip, obj, goal: current positions of the fork tip, object and goal.
        action: the 7 joint torques applied this step.
        prev_tip, prev_obj, prev_goal: positions at the previous step.
        q: the 7 joint angles.
    """
    # Object -> goal progress (planar: the goal marker sits ~4.8 cm below the
    # object's centre, so a 3D distance can never drop below ~0.048)
    d_now = planar_dist(obj, goal)
    d_prev = planar_dist(prev_obj, prev_goal)
    r_progress = W_PROGRESS * (d_prev - d_now)

    # Tip -> pre-push point
    d_tip_now = np.linalg.norm(tip - pre_push_point(obj, goal))
    d_tip_prev = np.linalg.norm(prev_tip - pre_push_point(prev_obj, prev_goal))
    r_tip = W_TIP * (d_tip_prev - d_tip_now)

    # Control cost and arm posture penalty
    r_ctrl = -W_CTRL * np.sum(np.square(action))
    r_arm = -(W_FLEX * q[5] ** 2 + W_ROLL * q[6] ** 2)

    # Success bonus
    r_success = SUCCESS_BONUS if d_now < SUCCESS_RADIUS else 0.0

    return r_progress + r_tip + r_ctrl + r_arm + r_success
