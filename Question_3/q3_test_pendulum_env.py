import gymnasium as gym
import numpy as np


env = gym.make("Pendulum-v1", render_mode="human")

state, info = env.reset()

print("Environment: Pendulum-v1")
print("Initial state:", state)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Action low:", env.action_space.low)
print("Action high:", env.action_space.high)

for step in range(10):
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {step + 1} | "
        f"Action: {action} | "
        f"Reward: {reward:.3f} | "
        f"Next state: {next_state}"
    )

env.close()