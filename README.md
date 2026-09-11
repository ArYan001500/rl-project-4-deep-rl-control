# RL project 4: Deep reinforcement learning control

This project uses deep reinforcement-learning methods for Mountain Car and Pendulum control.

## Contents

- Double DQN for the custom AIUT Mountain Car environment.
- Gymnasium MountainCar demo using trained Double DQN weights.
- PPO for `Pendulum-v1` with Gaussian continuous-action policies.
- SAC for `Pendulum-v1` with twin critics, target critics, replay buffer, soft target updates, and automatic entropy tuning.
- Training logs, reward plots, and selected `.h5` weights for the final demonstrations.

## Results from the project notes

- Double DQN on AIUT Mountain Car: deterministic 20-episode test mean reward `-104.4`, best `-82`, worst `-116`.
- Gymnasium MountainCar demo: reported 5-episode mean reward `-102.6`.
- PPO V3 on Pendulum-v1: deterministic 20-episode test mean reward `-195.06`, best `-1.96`, worst `-375.96`.
- SAC on Pendulum-v1: deterministic 20-episode test mean reward `-149.86`, best `-0.18`, worst `-298.74`.

SAC performed better than PPO V3 in the reported Pendulum-v1 comparison by mean test reward.
