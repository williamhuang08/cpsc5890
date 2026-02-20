# Lab 4 — Diffusion Policy

## Objectives

By the end of this lab, you will:

- Implement the forward diffusion process
- Design and compare multiple noise schedules
- Visualize trajectory corruption over time
- Implement a noise prediction model
- Train diffusion models with different schedules and step counts
- Compare performance across configurations
- Generate rollouts using DDPM sampling
- Deploy diffusion-based control on the robot

---


# Part 0 — Installation
```
conda env create -f lab4.yml
```

# Part 1 — Forward Process

In this section, you will study how trajectories are progressively corrupted by noise.

This builds intuition about:
- stochastic processes
- noise schedules
- trajectory distributions
- how diffusion transforms structured motion into randomness

---

## 1. Visualize Noising Iterations
Add Noise to a Trajectory

```
python -m scripts.ddpm --mode visual --config config/lift_cube.yaml
```

## 2. Define a Simple Noise Schedule

Start with a basic cosine beta schedule:

## 3. Implement Two Additional Noise Schedules

In addition to cosine, implement two more schedules.

Examples:
- linear schedule
- Quadratic schedule
- Sigmoid schedule
- Exponential schedule

---

## Reflection Questions

Answer in your writeup:

1. How many steps are necessary to turn a trajectory into a random distribution?
2. How does the noise schedule affect the corruption rate?
3. What parameters did you change?
4. What was their effect on:
   - Speed of corruption?
   - Smoothness of corruption?
   - Stability?

---

## Deliverable — Forward Process

Choose 3 different noise schedules.

Using the same initial trajectory, generate:

- A visualization of the forward sequence
- Save a screenshot for each schedule

Example structure:

figs/
linear_schedule.png
cosine_schedule.png
schedule_3.png


---

# Part 2 — Backward Process (Learning to Denoise)

Now we learn to invert the corruption.

---

## 1. Implement Noise Prediction Model

Implement:

ε_θ(xₜ, t)

This can be:
- MLP
- 1D CNN
- UNet (recommended)

Train with loss:

L = ||ε − ε_θ(xₜ, t)||²

---

## 2. Visualize One Noising + Denoising Example

For debugging:

- Take one trajectory
- Noise it
- Run reverse process
- Compare:
  - original
  - noised
  - reconstructed

Check:
- Does denoising improve over iterations?
- Does loss correlate with visual quality?

---

## 3. Train With Different Configurations

```
python -m scripts.ddpm --mode train --config config/lift_cube.yaml
```

Vary:

- Number of denoising steps
- Noise schedule

Use:

- 3 noise schedules
- 3 different denoising step counts

Example:
- 50 steps
- 100 steps
- 200 steps

---

## Deliverable — Loss Table

Report final validation loss:

| Noise Schedule | 50 Steps | 100 Steps | 200 Steps |
|---------------|----------|-----------|-----------|
| Linear        |          |           |           |
| Cosine        |          |           |           |
| Schedule 3    |          |           |           |

Discuss:
- Does more denoising always help?
- Which schedule works best?

---

# Part 3 — Sampling

Now generate trajectories.

```
python -m scripts.ddpm --mode inf --config config/lift_cube.yaml
```

---

## 1. Call DDPM Sampling

Implement reverse sampling loop:

xₜ₋₁ = DDPM_step(xₜ)

Generate rollouts from:

x_T ~ N(0, I)

---

## 2. Visualize 5 Rollouts

Using the same trained model:

- Sample 5 trajectories
- Plot them together

Observe:
- Diversity
- Smoothness
- Stability
- Failure cases

---

## Deliverable — Sampling Visualization

Save:

figs/
rollout_1.png
rollout_2.png
rollout_3.png
rollout_4.png
rollout_5.png



Or one combined figure.

---

# Part 4 — Deploy on the Robot

Now integrate diffusion into the control loop.

---

## 1. Incorporate Sampling Into Control

Modify control loop:

1. Observe current state
2. Sample action sequence from diffusion model
3. Execute first action
4. Repeat

Use:
- Low speeds
- Safety limits
- Collision detection

---

## 2. Run on the Robot

Test:
- Stability
- Smoothness
- Safety behavior
- Recovery from near-singular poses

---

## Deliverable — Robot Videos

Record:

- 3 videos of robot executing diffusion-generated trajectories

Save:

videos/
run1.mp4
run2.mp4
run3.mp4



---

# Debugging

If you run out of RAM, try closing down Windows (browser, VS code) to save memory. Try set DataLoader ```num_workers``` to 0 and gradually increase it.

You can slice this list ```episode_files = _list_episode_files(data_dir)``` in `scripts/dataset.py` to enable faster data loading when testing code. 



---

# Checkers (Written Reflection)

Include answers in README:

1. How does the noise schedule affect learning stability?
2. Why might cosine schedules outperform linear ones?
3. Why does increasing denoising steps not always improve performance?
4. Why is diffusion more stable than autoregressive action models?
5. How does trajectory length affect diffusion difficulty?
6. Why does sampling produce diverse rollouts?
7. What failure modes did you observe on the robot?
8. Why is safety critical when deploying stochastic policies?
9. How does diffusion compare to behavior cloning for this task?
10. What tradeoffs exist between speed and sample quality?

---

# Submission Checklist

- [ ] 3 forward-process visualizations
- [ ] Loss table (3 schedules × 3 step counts)
- [ ] 5 rollout visualizations
- [ ] 3 robot execution videos
- [ ] Written reflection answers
- [ ] GitHub repo link
