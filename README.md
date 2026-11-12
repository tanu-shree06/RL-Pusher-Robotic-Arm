# RL Pusher Robotic Arm

Training a 7-DoF robotic arm to push an object to a goal using **Proximal Policy Optimization (PPO)** in the Gymnasium `Pusher-v5` environment, with a custom reward.

---

## Objective
The objective of this project is to train a 7-DoF robotic arm to push an object towards a specified target position using Reinforcement Learning.

The arm learns an optimal pushing strategy through trial and error using the Proximal Policy Optimization (PPO) algorithm. A custom reward function is designed to encourage the arm to approach the object, make effective contact, and push it towards the goal

---

## What is RL?

Reinforcement Learning (RL) is how a computer learns to do a task **by practicing**, the same way you learn a video game or train a pet.

- **Trial and error:** the learner (called the agent) tries something, sees what happens, and tries again. Nobody shows it the correct move.
- **Only a score as feedback:** after each action it gets one number, the **reward**. A high number means "good", a low or negative number means "bad". It never hears *why*, so it has to work out for itself which actions led to the high scores.
- **Goal:** over many attempts, learn to pick the actions that give the highest total reward.

```
          action (7 torques)
   ┌────────────────────────────▶┐
 Policy                      Environment
 (the learner) ◀─────────────────┘
   what it sees (23 numbers) + reward
```

**In this project:** the learner is a neural network that looks at 23 numbers describing the arm and the object, and decides how hard to turn each of the 7 joints. After every step it gets a reward. We never program the arm's movements. We only design the reward, and the arm figures out how to push the cylinder to the goal by itself.

---
## APPROACH


## Environment and observation space

We use the Gymnasium **`Pusher-v5`** environment (MuJoCo): a 7-DoF arm with a forked tip pushes a cylinder to a red goal disc.

| Property | Value |
| --- | --- |
| Degrees of freedom | 7 |
| Observations | 23 |
| Steps per episode | 100 |
| Success radius | 5 cm |

| Observation indices | Meaning |
| --- | --- |
| 0 – 6 | Joint angles (7) |
| 7 – 13 | Joint angular velocities (7) |
| 14 – 16 | Fork tip position (x, y, z) |
| 17 – 19 | Object position (x, y, z) |
| 20 – 22 | Goal position (x, y, z) |

---

## Action space

Each action is a vector of **7 joint torques**, one per joint, each in the range **[-2, 2]**. The PPO actor outputs `a0 … a6`, which are applied to joints `q0 … q6`.

| Joint | Motion |
| --- | --- |
| q0 | Shoulder pan |
| q1 | Shoulder lift |
| q2 | Upper-arm roll |
| q3 | Elbow flex |
| q4 | Forearm roll |
| q5 | Wrist flex |
| q6 | Wrist roll |

---

## Reward function

### Default Gymnasium reward

```
reward = −0.5·d(tip, obj) − 1.0·d(obj, goal) − 0.1·‖a‖²
```

By default, `Pusher-v5` has no success condition, so every episode runs for the full 100 steps.

### Our reward

```
reward = F + 20·Δd(obj, goal) − 0.001·‖a‖² − (0.02·q5² + 0.01·q6²) + 50·success
```

### What each term means

- **`d(tip, obj)`**: distance between the fork tip and the object. Smaller is better, hence the minus sign.
- **`d(obj, goal)`**: distance between the object and the goal. Smaller is better.
- **`‖a‖²`**: the sum of the squares of the 7 torques, i.e. how much effort the arm uses. It is penalized so the arm doesn't waste energy. We keep this penalty tiny (0.001) in our reward, because a larger one (-0.05) made the arm stand still.
- **`F = Φ(t) − Φ(t−1)`**: the reward for **progress**. `Φ` is a score for the current situation, `Φ = −5·d(obj, goal) − 2·d(tip, push point)`, where the push point is the spot near the object from which the fork can push it. `F` is positive when the arm moves closer to the push point or the object moves closer to the goal. Because it only rewards *change*, it guides the arm without changing which behavior is best.
- **`Δd(obj, goal)`**: how much closer the object moved to the goal during this step. Multiplied by 20, it is the **push bonus**.
- **`q5`, `q6`**: the wrist flex and wrist roll joint angles. Penalizing them **keeps the fork level**.
- **`success`**: equals 1 when the object is within 5 cm of the goal, and the episode ends. The `+50` is the **success bonus**.

---

## What is Deep RL?

**Deep Reinforcement Learning (DRL)** is Reinforcement Learning that uses **deep neural networks** to make decisions.

In basic RL, the agent keeps a table of what is good in each situation. That works only for small problems. Real problems have far too many possible situations (camera images, many continuous sensor readings, robot joints), so a table is impossible. DRL replaces the table with a neural network that takes the situation as input and learns to recognize patterns, so it can make good decisions even in situations it has never seen exactly before.

---

## What is PPO?

**Proximal Policy Optimization (PPO)** is a reinforcement learning algorithm that improves the policy in **small, safe steps**, so that one bad update can't wreck training.

It uses two neural networks:

- **Actor:** looks at the 23 observations and chooses the 7 torques.
- **Critic:** estimates how much reward to expect from a given state.


**How it learns**

1. The arm runs for a while with the current policy, and we collect what happened (observations, torques, rewards).
2. The critic judges how good each state was, so we can tell whether an action turned out **better or worse than expected** (this is called the advantage).
3. Actions that did better than expected are made more likely. Actions that did worse are made less likely.

Each update uses 8192 samples (4 environments × 2048 steps), reused for 10 epochs in batches of 64.

### The clipping idea

Think of clipping as a **speed limit on learning**. After each round of practice, the arm's behavior is only allowed to change by a small amount (about 20%).

- If an action has already become 20% more likely, PPO stops pushing it further, even if it looked really good.
- If a bad action has already become 20% less likely, PPO stops reducing it further.
Without this limit, one lucky or unlucky run could make the arm suddenly behave completely differently and undo everything it had learned. With it, the arm improves steadily.

---


## Training hyperparameters

| Setting | Value |
| --- | --- |
| Algorithm | PPO (Stable-Baselines3, `MlpPolicy`) |
| Total timesteps | 1,500,000 |
| Parallel environments | 4 |
| Learning rate | 3e-4 |
| Rollout length | 2048 per environment |
| Batch size | 64 |
| Epochs per update | 10 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| Clip range | 0.2 |
| Normalization | `VecNormalize` (observations, reward) |

---

## Results

| Success rate | Mean final distance | Mean episode length |
| :---: | :---: | :---: |
| **90%** | **0.058 m** | **61.1 steps** |

---

## Training graphs

<img width="1500" height="750" alt="training_curves" src="https://github.com/user-attachments/assets/32d00901-a634-4e70-91c0-608a193162c3" />

Reward rises sharply after about 4,000 episodes, then stabilizes in the high-40s to low-50s.

---

## Training video

<img width="580" height="400" alt="pusher_success" src="https://github.com/user-attachments/assets/7ebcb325-b5dc-42fd-a96b-dbc0fedf616c" />

---

## Team

- Tanushree Sundar
- Nanda Kishor M
