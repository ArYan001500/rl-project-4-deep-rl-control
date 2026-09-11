import gymnasium as gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
state, info = env.reset()

print("State:", state)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("TensorFlow version:", tf.__version__)

env.close()