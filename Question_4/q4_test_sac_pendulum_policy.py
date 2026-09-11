import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import tensorflow as tf


ENV_NAME = "Pendulum-v1"

STATE_DIM = 3
ACTION_DIM = 1
ACTION_LOW = -2.0
ACTION_HIGH = 2.0

TEST_EPISODES = 20
RENDER_EPISODES = 3

BASE_DIR = Path(__file__).resolve().parent.parent
ACTOR_WEIGHTS_FILE = BASE_DIR / "sac_pendulum_actor_best.weights.h5"


def normalize_state(state):
    cos_theta = state[0]
    sin_theta = state[1]
    angular_velocity = state[2] / 8.0

    return np.array([cos_theta, sin_theta, angular_velocity], dtype=np.float32)


class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(256, activation="relu")

        self.mean_layer = tf.keras.layers.Dense(action_dim, activation="linear")
        self.log_std_layer = tf.keras.layers.Dense(action_dim, activation="linear")

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        return mean, log_std

    def deterministic_action(self, state):
        mean, _ = self(state)
        action_norm = tf.tanh(mean)
        return action_norm


def scale_action_to_env(action_norm):
    action_env = action_norm * ACTION_HIGH
    action_env = np.clip(action_env, ACTION_LOW, ACTION_HIGH)
    return action_env.astype(np.float32)


def select_deterministic_action(state, actor):
    state_array = np.array([state], dtype=np.float32)
    state_tf = tf.convert_to_tensor(state_array, dtype=tf.float32)

    action_norm_tf = actor.deterministic_action(state_tf)
    action_norm = action_norm_tf[0].numpy()

    action_env = scale_action_to_env(action_norm)

    return action_env


def run_test(actor, episodes, render=False):
    if render:
        env = gym.make(ENV_NAME, render_mode="human")
    else:
        env = gym.make(ENV_NAME)

    rewards = []

    for episode in range(1, episodes + 1):
        raw_state, info = env.reset()
        state = normalize_state(raw_state)

        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = select_deterministic_action(state, actor)

            raw_next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            state = normalize_state(raw_next_state)

            total_reward += reward
            steps += 1

            if render:
                time.sleep(0.01)

        rewards.append(total_reward)

        print(
            f"Test Episode {episode:02d} | "
            f"Reward: {total_reward:8.2f} | "
            f"Steps: {steps:3d}"
        )

    env.close()

    return rewards


if not ACTOR_WEIGHTS_FILE.exists():
    raise FileNotFoundError(
        f"Actor weight file not found: {ACTOR_WEIGHTS_FILE}\n"
        "Make sure sac_pendulum_actor_best.weights.h5 is inside D:\\Projects\\RlProject"
    )


actor = Actor(STATE_DIM, ACTION_DIM)

dummy_state = np.zeros((1, STATE_DIM), dtype=np.float32)
actor(dummy_state)

actor.load_weights(str(ACTOR_WEIGHTS_FILE))

print("======================================")
print("Question 4: SAC Pendulum-v1 Final Test")
print("Loaded actor weights from:", ACTOR_WEIGHTS_FILE)
print("Policy: deterministic actor action")
print("======================================")


print("\nTesting without rendering...")
test_rewards = run_test(actor, TEST_EPISODES, render=False)

print("\n======================================")
print("Final SAC test finished.")
print("Test episodes:", TEST_EPISODES)
print("Average reward:", np.mean(test_rewards))
print("Best reward:", np.max(test_rewards))
print("Worst reward:", np.min(test_rewards))
print("Standard deviation:", np.std(test_rewards))
print("======================================")


print("\nNow running graphical demo...")
render_rewards = run_test(actor, RENDER_EPISODES, render=True)

print("\n======================================")
print("Graphical SAC demo finished.")
print("Render episodes:", RENDER_EPISODES)
print("Average render reward:", np.mean(render_rewards))
print("======================================")