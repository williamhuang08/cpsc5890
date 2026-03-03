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
*Answer*: ~ 100 steps
3. How does the noise schedule affect the corruption rate?
*Answer*: The corruption rate for all noise schedulers is approximately identical.
5. What parameters did you change?
*Answer*: We changed the number of diffusion steps (50, 100, 200).
7. What was their effect on:
   - Speed of corruption?
   - Smoothness of corruption?
   - Stability?
*Answer*: We changed the number of diffusion steps (50, 100, 200). Fewer diffusion steps result in faster corruption but also lead to corruption that is not as smooth. The stabiltiy of training was very similar.
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
*Answer*: Yes, denoising improves significantly over iterations, often converging as the number of diffusion steps approaches the maximum.

- Does loss correlate with visual quality?
*Answer*: No. For example, linear scheduler w/ 200 diffusion steps had the best visual quality, but the cosine scheduler w/ 200 diffusion steps had the lowest loss.
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
| Linear        |   0.0513249       |   0.039398        |    0.023872       |
| Cosine        |   0.0214275       |   0.021293        |    0.018970       |
| Schedule 3    |   0.067531        |   0.043657        |    0.037537       |

Discuss:
- Does more denoising always help?
*Answer*: Yes, in general, more denoising steps result in lower validation loss.
- Which schedule works best?
*Answer*: The cosine scheduler has the lowest validation loss for all amounts of denoising steps.
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

*Answer*: For each of the noise schedulers, the validation loss decreases over time, indicating that learning is stable.

3. Why might cosine schedules outperform linear ones?

*Answer*: A cosine scheduler will reduce the amount of noise over time in a smoother way than a linear scheduler.

5. Why does increasing denoising steps not always improve performance?

*Answer*: In general, for us, increasing the number of denoising steps reduced the validation loss. However, I can imagine that if the aggregate amount of noise introduced to the image is not large, then a large number of denoising steps may be unnecessary.

7. Why is diffusion more stable than autoregressive action models?
   
*Answer*: Diffusion is more stable because it has a Markov assumption, while an autoregressive action model can be dependent on several previous timesteps.

9. How does trajectory length affect diffusion difficulty?

*Answer*: As the length of the trajectory increases, it becomes more difficult for diffusion to noise/denoise as there are more states/actions to noise/denoise.

11. Why does sampling produce diverse rollouts?
    
*Answer*: Sampling produces diverse rollouts because, for each diffusion step, we are sampling from a distribution to extract the level of noise that we want to remove from the previous noisy trajectory.

13. What failure modes did you observe on the robot?

*Answer*: The robot was close to succeeding. It seems like the robot arm begins to extend its end-effector toward the red block. However, the gripper either closes too early or the gripper does not reach the block. In some cases, we believe it could be because the robot was in a ood state. 

15. Why is safety critical when deploying stochastic policies?

*Answer*: When deploying stochastic policies, we need to ensure that all of the possible denoised rollouts from a given noised trajectory meet the safety requirement. In contrast, for a deterministic policy, we simply need to ensure a single rollout is safe. 

17. How does diffusion compare to behavior cloning for this task?

*Answer*: Diffusion performs much better than behavior cloning for this task. In this case, the robot arm is able to move towards the red block, but fails to pick up the block. For behavior cloning, the robot arm was unable to even begin a trajectory towards the block. This is because behavior cloning struggles in learning multimodal behavior, unlike a diffusion policy, which can exhibit multimodal behavior. However, behavior cloning is much faster to train than diffusion, especially as the number of diffusion steps increases. Both diffusion and behavior cloning suffer from action collapse in ood states.

19. What tradeoffs exist between speed and sample quality?



---

# Submission Checklist

- [ ] 3 forward-process visualizations
- [ ] Loss table (3 schedules × 3 step counts)
- [ ] 5 rollout visualizations
- [ ] 3 robot execution videos
- [ ] Written reflection answers
- [ ] GitHub repo link
