# Lab 5 — RL Policy Optimization

## Objectives

By the end of this lab, you will:

- Train policy-gradient algorithms using **Stable-Baselines3**
- Compare **PPO** and **SAC** on simple control tasks
- Study the effect of **hyperparameters on convergence**
- Design **sparse vs dense reward functions**
- Evaluate **convergence speed and stability**
- Generate **rollout videos**
- Deploy a trained policy to a **real robot**
- Analyze **sim-to-real performance differences**

---

## Setup

Set up your environment

```
conda env create -f lab5.yml
```

Please pull the gym_xarm repo inside the lab 5 folder and install via
```bash
git clone git@github.com:iqr-lab/gym-xarm.git
cd gym_xarm
pip install -e .
```

run the test script to see all the environments
```bash
MUJOCO_GL=glfw python scripts/test_gym_xarm.py --render human --steps 500 --episodes 2
```

Hit tab once when the sim appears if the camer is not aligned to switch the camera

# Running the Training Script

Training is performed using the provided `train.py` script.

### Example Usage

Train PPO:

```bash
python train.py \
    --algo ppo \
    --env reach \
    --timesteps 200000 \
    --eval-freq 5000 \
    --seed 0
```

Train SAC:

```bash
python train.py \
    --algo sac \
    --env reach \
    --timesteps 200000 \
    --eval-freq 5000 \
    --seed 0
```

### Output

The script will generate:

```
logs/
models/
videos/
plots/
```

Example structure:

```
logs/
    ppo_reach/
    sac_reach/

models/
    ppo_reach.zip
    sac_reach.zip

videos/
    rollout_ppo.mp4
    rollout_sac.mp4
```

---

# Part 1 — Simple Reaching Task (Simulation)

In this section, you will train RL policies on a simple **end-effector reaching task**.

The environment is built using:

- **MuJoCo via Gym**
- or **GELLO xArm simulation**

### Reward

The reward is predefined:

```
reward = - distance(end_effector, target)
```

### Performance Metrics

- Episode reward
- Success rate (distance threshold)

---

## 1. Train PPO and SAC

Train both algorithms using:

- PPO
- SAC

Use:

- Default reward function
- Default hyperparameters

### Plots

Generate:

- **Epochs vs Success Rate**
- **Epochs vs Reward**

⚠ Average over **multiple rollouts per evaluation epoch**.

---

## 2. Modify Key Hyperparameters

You will study how parameter choices affect learning.

### PPO — Modify

- Clip range
- Discount factor (γ)
- Learning rate
- Policy architecture

### SAC — Modify

- Exploration noise (`ent_coef`)
- Discount factor (γ)
- Learning rate
- Policy architecture

---

## Reflection Questions

- How does **clip range** affect PPO stability?
- How does **entropy (`ent_coef`)** affect SAC exploration?
- Which algorithm converges faster?
- Which is more stable?
- How sensitive are results to learning rate?
- Does larger **γ** improve performance?

---

## Deliverable — Reaching Task

Submit:

- **2 plots**
  - epochs vs success rate
  - epochs vs reward
- **Short written analysis (1–2 paragraphs)**

---

# Part 2 — Pick and Place Task

Now you will design **your own reward functions**.

---

## 1. Define Three Reward Functions

Design reward functions with **different sparsity levels**.

### Reward 1 — Dense

Example components:

- Distance to block
- Distance to goal
- Gripper alignment bonus
- Small shaping terms

---

### Reward 2 — Semi-Sparse

Example components:

- Bonus for lifting
- Bonus for placing near goal
- Small penalties for time

---

### Reward 3 — Sparse

```
+1 if block placed within tolerance
0 otherwise
```

---

## 2. Justify Each Reward

In your README explain:

For each reward:

- Why did you design it this way?
- What do you expect to happen?
- Pros?
- Cons?

---

## 3. Train PPO and SAC

Use **default hyperparameters**.

For each reward function train:

- PPO
- SAC

This results in:

```
6 trained models
```

---

## Deliverable — Reward Comparison

### Plot 1 (3 plots total)

For each reward function:

```
Epochs vs Reward
```

Both PPO and SAC on the same graph.

Total: **3 plots**

---

### Plot 2 (Combined Plot)

```
Epochs vs Success Rate
```

All **6 models** on one graph.

Lines:

- PPO (Dense)
- SAC (Dense)
- PPO (Semi)
- SAC (Semi)
- PPO (Sparse)
- SAC (Sparse)

---

## Reflection Questions

- Do both algorithms converge on all reward functions?
- Which reward function leads to **fastest convergence**?
- Which is **most stable**?
- Does SAC handle **sparse rewards better**?
- Which reward function would you deploy?

---

# Rollout Videos

Render simulation rollouts for all **6 models**.

Environment must match:

- Real robot block start position
- Real robot goal position

Submit:

```
videos/

ppo_dense.mp4
sac_dense.mp4
ppo_semi.mp4
sac_semi.mp4
ppo_sparse.mp4
sac_sparse.mp4
```

---

## Reflection

- Do rollouts look **safe**?
- Is behavior **smooth**?
- Is performance **consistent across rollouts**?
- Are there **oscillations or instability**?
- Which model appears **most reliable**?

---

# Part 3 — Fine-Tuning

Choose:

- One reward function
- One algorithm

Explain:

```
Why did you choose this pair?
```

---

## Improve the Model

Modify:

- Learning rate
- Network size
- Entropy (SAC)
- Clip range (PPO)
- Discount factor
- Training time
- Batch size

---

## Deliverable

Plot:

```
Baseline vs Improved
Epochs vs Success Rate
```

Render:

```
videos/improved_model.mp4
```

---

## Reflection

- What modifications helped?
- Why?
- Did convergence speed improve?
- Did stability improve?
- Any tradeoffs?

---

# Part 4 — Real Robot Deployment

After verifying simulation rollouts:

Deploy the chosen model to the **real robot**.

---

## Safety Requirements

- Low speed
- Workspace limits
- Emergency stop ready
- Supervised operation

---

## Deliverable

Record:

```
videos/real_robot_test.mp4
```

---

## Reflection Questions

- How did real performance compare to simulation?
- Were motions slower or less accurate?
- Did failures occur?

Possible causes:

- Dynamics mismatch
- Friction
- Sensor noise
- Delay

How would you reduce the **sim-to-real gap**?

---

# Pre-Lab Quiz

Before lab, answer:

- Match **Stable-Baselines3 parameters** to algorithm theory
- How does **entropy affect SAC**?
- How does **clip range affect PPO**?
- How does **discount factor affect long-term reward**?
- How do you define **policy network architecture**?

---

# TODO (Instructor Setup)

- Add objects to MuJoCo via **GELLO or Gym xArm**

Randomize:

- Robot start pose
- Block position
- Goal position

Match simulation to real robot:

- Base height
- Base frame orientation

Provide working:

- PPO training script
- SAC training script
- Predefined reaching reward

Provide:

- Evaluation loop
- Averaging across rollouts
- Plotting utilities
- Example policy architectures
- Code for rendering and saving videos

---

# Submission Checklist

Submit:

- 2 reaching plots
- 3 reward comparison plots
- 1 combined success-rate plot (6 lines)
- 6 simulation rollout videos
- Fine-tuning comparison plot
- Improved rollout video
- Real robot video
- Written reflections
- GitHub repo link
