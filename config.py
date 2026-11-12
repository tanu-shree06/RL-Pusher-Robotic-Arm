"""Central configuration for the RL Pusher project.

Every tunable value lives here, so an experiment only needs edits in one file.
"""

# ------------------------------------------------------------
# Run settings
# ------------------------------------------------------------
ENV_ID = "Pusher-v5"
SEED = 0
N_ENVS = 4                      # parallel environments
TOTAL_TIMESTEPS = 1_500_000

# ------------------------------------------------------------
# Output paths
# ------------------------------------------------------------
MODEL_NAME = "pusher_ppo_1500k"
VECNORM_NAME = "vecnorm_1500k.pkl"
VIDEO_FOLDER = "recorded_videos"
PLOT_PATH = "training_curves.png"

# ------------------------------------------------------------
# PPO hyperparameters (Stable-Baselines3)
# ------------------------------------------------------------
LEARNING_RATE = 3e-4
N_STEPS = 2048                  # rollout length per environment
BATCH_SIZE = 64
N_EPOCHS = 10                   # epochs per update
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

# ------------------------------------------------------------
# Reward design
# ------------------------------------------------------------
SUCCESS_RADIUS = 0.05           # object within 5 cm of the goal = success
PRE_PUSH_OFFSET = 0.06          # tip should get this far behind the object
W_PROGRESS = 20.0               # weight on object -> goal progress
W_TIP = 2.0                     # weight on tip -> pre-push point progress
W_CTRL = 0.001                  # control (torque) cost
W_FLEX = 0.02                   # posture penalty on q[5], wrist flex
W_ROLL = 0.01                   # posture penalty on q[6], wrist roll
SUCCESS_BONUS = 50.0

# ------------------------------------------------------------
# Logging and evaluation
# ------------------------------------------------------------
WINDOW = 50                     # moving-average window (episodes)
N_EVAL_EPISODES = 50
VIDEO_LENGTH = 1000             # steps recorded in the demo video
