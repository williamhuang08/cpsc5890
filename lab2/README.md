# Lab 2 — Behavior Cloning (BC) and DAgger

William Huang, Farhan Baig

## Objectives
- Understand how demonstrations are represented as data
- Collect demonstrations using GELLO or kinesthetic teaching
- Replay and analyze demonstrations
- Implement simple policies and Behavior Cloning (BC)
- Train, evaluate, and visualize BC models
- Implement, train, and evaluate DAgger (robot-led and human-led)
- Compare vanilla BC and DAgger approaches
- Understand evaluation metrics and intervention strategies


## Part 1: Understanding Demonstrations

### Step 1: Demonstrations

GELLO walk-through.

## Part 2: Behavior Cloning

### Goal

In this part, you will train and evaluate a Behavior Cloning (BC) policy that learns to imitate robot behavior from demonstration data. Your objective is to verify that the trained BC model can reproduce actions similar to the recorded demonstrations and to understand its limitations.

Behavior Cloning treats imitation learning as a supervised learning problem: given a robot state, the model predicts the action taken by the demonstrator in that state.

### Task

### Step 1: Train the BC model

Use a pre-recorded demonstration dataset ```asset/demo.npz``` to train a BC model:

```bash

python -m scripts.bc --mode train --ip <robot_ip> --epochs <epochs> --batch_size <batch_size> --lr <lr>
```
* Experiment with hyperparameters (epochs, batch_size, lr)

* Observe the training and test loss over epochs

### Step 2: Run inference on the robot
Use the trained BC model to generate actions on the robot:

```bash

python -m scripts.bc --ip <robot_ip> --mode inference
```
The robot will replay actions predicted by the BC model
Observe smoothness, accuracy, and timing relative to the original demonstration
Record the EEF state trajectory using the provided visualization code.

### Step 3: Training and Inference Using High-Frequency Data

Repeat Step 1 and 2 using ```asset/demo_high_freq.npz``` instead.

```bash
python -m scripts.bc --mode train --ip <robot_ip> \
    --data asset/demo_high_freq.npz \
    --epochs <epochs> --batch_size <batch_size> --lr <lr>
```

Tune hyperparameters as needed (epochs, batch size, learning rate).

Monitor training and test loss curves.

Compare convergence behavior against the low-frequency dataset.

Run inference using the BC model trained on high-frequency data. Be sure to set ```wait=False``` in ```set_servo_angle``` (line 355) and ```set_gripper_position``` (line 361) to enable high-frequency control. Leave ```wait=True``` at anywhere else.

```bash
python -m scripts.bc --mode inference --ip <robot_ip>
```
Observe the resulting robot motion.

Pay attention to smoothness, responsiveness, and stability.

Compare execution timing and trajectory fidelity against the original demonstrations.

Record the EEF state trajectory using the provided visualization code.

### What to Record and Report
* Are the movements smooth?
* Do actions closely follow the demonstration?
* Are there any large deviations, jerks, or unexpected behavior?
* Start the robot from slightly different initial poses. Observe whether the BC model still produces reasonable behavior
* Training hyperparameters: epochs, batch_size, lr
* Loss over time. Final training and test loss
* Visualization of visited EEF states
* Record a video of successes and failures

### Reflection Questions
* How closely does the BC model reproduce the original demonstrations?

*Answer*: We did not have time to fully replicate the original demonstrations. The model we trained was able to move the arm from the start state to a later point in the trajectory, but then the robot end-effector collides with the table.

* Where does the model fail or deviate most significantly? Why might that happen?

*Answer*: The arm encoutered a set_servo_angle on the third and fourth joint. We hypothesize that this could be because the model we trained was not expressive enough (need to increase the dimension of the hidden layer) and/or experiment more with the batch size/learning rate.

* What could go wrong if the robot starts from a pose outside the demonstration distribution?

*Answer*: If the robot starts from a pose outside the demonstration, then the robot has entered a state that is out-of-distribution, leading to the distributional shift problem mentioned in class. Thus, the action predicted by the BC policy in this state will likely be meaningless.


* Why is normalization of states and actions important for BC performance? Is this the only way to pre-process data?

*Answer*: Normalization of states and actions is important for BC performance because it leads to more stable training as larger inputs could lead to larger gradient updates. Alternative ways to pre-process the data could be to smoothen the trajectories such that the motions are less jerky.

## Part 3: DAgger

### Goal

In this part, you will iteratively improve the Behavior Cloning (BC) policy by collecting on-policy states visited by the learned policy and labeling them with the expert (human or scripted). The goal is to reduce compounding errors that occur when the BC policy encounters states not present in the original demonstrations.

#### Key Concept:

Vanilla BC only learns from offline demonstrations. DAgger collects states visited by the learned policy and adds expert labels, reducing distributional shift.

### Step 1: Initialize DAgger

Use the high-frequency demonstration dataset ```asset/demo_high_freq.npz``` to train an initial BC policy:

```bash

python -m scripts.dagger \
    --mode train \
    --data asset/demo_high_freq.npz \
    --epochs <epochs> \
    --batch-size <batch-size> \
    --lr <lr> \
    --ip <robot_ip>
```
This is your starting BC model.

### Step 2: Run DAgger Iterations
Run DAgger to collect new on-policy states and aggregate them with the original dataset. Finish any TODO before running the following script.

```bash
python -m scripts.dagger \
    --mode dagger \
    --ip <robot_ip>
```
Add and adjust prarameter as needed.

Explanation of parameters:
```bash
--dagger-iters: Number of DAgger iterations (retrain with aggregated data).

--dagger-rollout-episodes: Number of episodes to collect per iteration.

--beta0 / --beta-decay: Probability of following the expert vs learned policy during rollouts.

```

During each iteration:

The robot executes a mixture policy (expert with probability β, learned BC policy otherwise).

States visited by the robot are labeled with the expert action.

The dataset is aggregated and used to retrain the BC model.

After each iteration, the following are saved automatically:
```bash
asset/bc_policy.pt       # retrained model
asset/bc_norm.npz        # normalization stats
asset/dagger_agg.npz     # aggregated dataset
```

### Step 3: Run Inference Using the DAgger-Trained Policy
Repeat Step 1 and 2 using ```asset/demo_high_freq.npz``` instead,

Train the BC model using the high-frequency dataset:

```bash
python -m lab2.scripts.dagger \
    --mode inference \
    --ip <robot_ip> \
    --episodes <episodes> \
    --out asset/inf_dagger.npz
```

This runs the DAgger-trained policy on the robot.

Observe trajectory smoothness, accuracy, and response compared to vanilla BC.

Record EEF state trajectories using plot_3d_positions.

### What to Record and Report
* Aggregated dataset size after each DAgger iteration.
* Training and test loss curves across iterations.
* Comparison of robot motion: BC vs DAgger.
* Are movements smoother?
* Does DAgger reduce deviations in states outside the original demonstration distribution?
* Videos of robot performing DAgger policy. Label successes and failures.
* EEF trajectories visualized in 3D.

### Reflection Questions
* How does DAgger improve performance compared to vanilla BC?

*Answer*: DAgger improves performance compared to vanilla BC because the algorithm will query the expert for all states, including states that are out-of-distribution.
* Which states benefit most from expert relabeling?

*Answer*: States that benefit most from expert relabeling are states that are out-of-distribution for the current trained policy.
* How does high-frequency data affect DAgger’s stability and learning?
*Answer*: High-frequency data helps DAgger's stability because the learned policy will be able to create fine-grained actions, which will lead to less jerky motions.

* What are potential risks if β decays too quickly or too slowly?

* How could you extend this approach to handle dynamic tasks or obstacles?

## Final Questions

* How are demonstrations represented in the dataset? What do the observation and action arrays correspond to?

* How does the sampling frequency (low vs high frequency) affect the recorded data?

* Did you notice any noise or irregularities in the demonstrations? How might these affect imitation learning?

* How closely did the BC model reproduce the original demonstrations? Provide examples.

* In which situations did the BC model fail or deviate from the demonstrations? Why might this happen?

* How does the model behave when the robot starts from a state outside the demonstration distribution?

* How do hyperparameters (epochs, batch size, learning rate) affect the training and test loss?

* How smooth and responsive were the robot’s actions during BC inference? Were there any jerks or unexpected movements?

* Why is normalization of states and actions crucial for BC? Can you think of other preprocessing methods that might help?

* How does DAgger address the compounding error problem seen in vanilla BC?

* What effect did aggregating on-policy states have on model performance?

* How did the choice of beta (expert probability) affect the policy rollout? What happened when beta decayed too quickly or too slowly?

* If DAgger did not improve policy by much, why?
