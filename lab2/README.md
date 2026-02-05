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

*Answer*: All of these are not observable since we were unable to get to this point in the lab.

### Reflection Questions
* How does DAgger improve performance compared to vanilla BC?

*Answer*: DAgger improves performance compared to vanilla BC because the algorithm will query the expert for all states, including states that are out-of-distribution.
* Which states benefit most from expert relabeling?

*Answer*: States that benefit most from expert relabeling are states that are out-of-distribution for the current trained policy.

* How does high-frequency data affect DAgger’s stability and learning?

*Answer*: High-frequency data helps DAgger's stability because the learned policy will be able to create fine-grained actions, which will lead to less jerky motions.

* What are potential risks if β decays too quickly or too slowly?

*Answer*: If beta decays too quickly, we do not query the expert enough early on, which might lead to us aggregating unsage and far from intended states in our dataset. This would mean worse performance later on. If it decays too slowly, then the policy becomes too reliant on the expert, leading the agent to frequently reach states that the agent has already seen.

* How could you extend this approach to handle dynamic tasks or obstacles?

*Answer*: We could add more information to the model such as obstacle locations (potentially through RGB or some other encoding). Since DAgger does well at handling slight variation we can add more dynamism such that the model doesn't overfit and is comfortable with varition in its environment.

## Final Questions

* How are demonstrations represented in the dataset? What do the observation and action arrays correspond to?

*Answer*: They are represented as observations and actions. The observation array corresponds to joint angle and gripper position pairs, and the action array corresponds to a list of joint angle changes for each joint on the robot arm that is predicted by the model according to the policy.

* How does the sampling frequency (low vs high frequency) affect the recorded data?

*Answer*: If we sample more frequently (high frequency), we have more data, which also means that the actions we predict are smoother since dq is small due to the difference in consecutive states being smaller. Similarly, the opposite is true for low-frequency sampling. 

* Did you notice any noise or irregularities in the demonstrations? How might these affect imitation learning?

*Answer*: Cannot say as we were unable to get to this point in lab.

* How closely did the BC model reproduce the original demonstrations? Provide examples.

*Answer*: Although we were not able to reproduce the task of picking up a block from the table, we were able to move the arm to nearly reach the table before we got a set_servo_angle error.

* In which situations did the BC model fail or deviate from the demonstrations? Why might this happen?

*Answer*: Cannot say as we were unable to get to this point in lab. However, I hypothesize that this would happen when the BC policy encounters states that are out-of-distribution.

* How does the model behave when the robot starts from a state outside the demonstration distribution?

*Answer*: Cannot say as we were unable to get to this point in lab. However, I would predict that the predicted actions by the policy would be meaningless.

* How do hyperparameters (epochs, batch size, learning rate) affect the training and test loss?

*Answer*: We saw that raising the number of epochs decreased our loss. Learning rate and batch size we were unable to test thoroughly as we were fighting through the bugs in lab.

* How smooth and responsive were the robot’s actions during BC inference? Were there any jerks or unexpected movements?

*Answer*: The robot's actions seemed to be jerky at times, especially when the robot arm started to move down to pick up the object. We hypothesize that this may be because the kinesthetic demonstrations were not that smooth.

* Why is normalization of states and actions crucial for BC? Can you think of other preprocessing methods that might help?

*Answer*: Normalization is crucial since we have to equate the states and actions somehow as joint angles and gripper positions take on different scales. Addtiioanlyl as mentioned earlier its important to stabilize gradients and improve training. Other preprocessing might be things like low pass filters in order to improve smoothness.

* How does DAgger address the compounding error problem seen in vanilla BC?

*Answer*: We were unable to observe this empirically. However, theoretically speaking, it addresses the compounding error by augmenting the dataset through calls to the expert and the policy (which we then retrain on). Generally, we decay beta as well, so we tend to query the expert less over time, allowing our model not to overfit to the expert and avoid the compounding issue.

* What effect did aggregating on-policy states have on model performance?

*Answer*: Cannot say as we were unable to get to this point in lab.

* How did the choice of beta (expert probability) affect the policy rollout? What happened when beta decayed too quickly or too slowly?

*Answer*: Cannot say as we were unable to get to this point in lab.

* If DAgger did not improve policy by much, why?

*Answer*: Cannot say as we were unable to get to this point in lab.
