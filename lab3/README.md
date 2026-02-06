# Lab 3 — Visuomotor Policies, Action Embeddings, Autoencoders, and VAEs


## Names: Farhan Baig, William Huang, Vini Rupchandani

## Objectives

By the end of this lab, you will:

- Incorporate **observations (images)** into robot policies  
- Turn a **state-based BC policy** into a **visuomotor policy**
- Understand how **visual + state features** are fused inside policies
- Implement and train **Autoencoders (AE)** and **Variational Autoencoders (VAE)** for **action embeddings**
- Study how **model complexity** and **latent dimensionality** affect performance
- Use learned embeddings inside **Behavior Cloning (BC)**
- Evaluate performance on:
  - in-distribution (ID) start states  
  - out-of-distribution (OOD) start states  
- Deploy best models on the **real robot**

---

## Setup

Install GELLO software: https://github.com/wuphilipp/gello_software

Download training dataset of lifting task: https://drive.google.com/file/d/1wXDLTP4kNBk1rIlRg8erWKcrAW5BFSfq/view?usp=drive_link

Connect to Yale Secure: https://yale.service-now.com/it?id=kb_article&sysparm_article=KB0025500

## Part 1 — Visuomotor Behavior Cloning

We extend BC from:
state → action
to
(image, state) → action

### Policy Architecture

| Component | Role |
|----------|------|
| CNN encoder | Extract visual features |
| MLP encoder | Encode robot state |
| Fusion | Concatenate vision + state features |
| Policy head | Predict action |

### Tasks

1. Inspect the starter architectures
2. Match code to architecture diagrams
3. Train visuomotor BC
4. Compare:
   - training loss  
   - validation loss  
5. Evaluate policy on:
   - ID start states  
   - OOD start states  

### Reflection Questions

When do images actually help Behavior Cloning?

Why is the image encoder applied independently at each timestep?

Why can BatchNorm be problematic in image-based BC?

What are the risks of training an image encoder from scratch in BC?

How can you tell whether the policy is actually using visual information?

---

## Part 2 — Action Autoencoders

We compress actions using:
action → encoder → latent z → decoder → reconstructed action

VAEs learn a **distribution** over latent actions.

Loss:

\[
L = \text{reconstruction} + \beta \cdot KL(q(z|a) || N(0,1))
\]

| AE | VAE |
|----|----|
| Deterministic | Probabilistic |
| Can overfit | Regularized |
| Weak OOD behavior | Better structured latent space |

### VAE → AE

If you:
- remove KL term
- make encoder deterministic  

then a VAE becomes a regular AE.

### Two Axes to Explore

| Axis | Meaning |
|------|---------|
| Model complexity | depth / width of encoder and decoder |
| Latent dimension | size of bottleneck |

### Deliverable 1 — VAE Performance Grid

Create a **3×3 table**:

| Latent Dim ↓ / Model Size → | Small | Medium | Large |
|-----------------------------|-------|--------|-------|
| Low dim                     |       |        |       |
| Mid dim                     |       |        |       |
| High dim                    |       |        |       |

Report:
- Training reconstruction loss
- Validation reconstruction loss

### Questions

- When does increasing latent size stop helping?
- Which model overfits?
- What is the best latent dimension for reconstruction?

Now BC predicts **latent actions** instead of raw actions:

BC: (obs) → z
Decoder(z) → action

### Deliverable 2 — BC Performance Grid

Same 3×3 table, but report:

- BC training loss  
- BC validation loss  
- (Optional) task success rate  

### Question

Does lower VAE reconstruction loss → better BC performance?  
Often **no** — reconstruction quality does not always align with control usefulness.

### Required Trials

Inference the best performing model on the real robot:

| Type | # Trials |
|------|----------|
| ID start | 2 |
| OOD start | 2 |

Submit videos.

## Part 6 — Repeat for State Embeddings

Repeat AE and VAE experiments, but encode:
