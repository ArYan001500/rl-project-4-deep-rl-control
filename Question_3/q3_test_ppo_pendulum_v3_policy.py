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
MODEL_WEIGHTS_FILE = BASE_DIR / "ppo_pendulum_v3_best.weights.h5"


def normalize_state(state):
    cos_theta = state[0]
    sin_theta = state[1]
    angular_velocity = state[2] / 8.0

    return np.array([cos_theta, sin_theta, angular_velocity], dtype=np.float32)


class ActorCritic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared1 = tf.keras.layers.Dense(128, activation="tanh")
        self.shared2 = tf.keras.layers.Dense(128, activation="tanh")

        self.mean_layer = tf.keras.layers.Dense(action_dim, activation="tanh")
        self.value_layer = tf.keras.layers.Dense(1, activation="linear")

        self.log_std = self.add_weight(
            name="log_std",
            shape=(action_dim,),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True
        )

    def call(self, state):
        x = self.shared1(state)
        x = self.shared2(x)

        mean_normalized = self.mean_layer(x)
        value = self.value_layer(x)

        return mean_normalized, value


def select_deterministic_action(state, model):
    state_array = np.array([state], dtype=np.float32)

    mean_normalized, _ = model(state_array)

    action_normalized = mean_normalized[0].numpy()
    action_env = action_normalized * ACTION_HIGH
    action_env = np.clip(action_env, ACTION_LOW, ACTION_HIGH)

    return action_env.astype(np.float32)


def run_test(model, episodes, render=False):
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
            action = select_deterministic_action(state, model)

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


if not MODEL_WEIGHTS_FILE.exists():
    raise FileNotFoundError(
        f"Weight file not found: {MODEL_WEIGHTS_FILE}\n"
        "Make sure ppo_pendulum_v3_best.weights.h5 is inside D:\\Projects\\RlProject"
    )


model = ActorCritic(STATE_DIM, ACTION_DIM)

dummy_state = np.zeros((1, STATE_DIM), dtype=np.float32)
model(dummy_state)

model.load_weights(str(MODEL_WEIGHTS_FILE))

print("======================================")
print("Question 3: PPO Pendulum-v1 V3 Final Test")
print("Loaded weights from:", MODEL_WEIGHTS_FILE)
print("Policy: deterministic mean action")
print("======================================")


print("\nTesting without rendering...")
test_rewards = run_test(model, TEST_EPISODES, render=False)

print("\n======================================")
print("Final PPO V3 test finished.")
print("Test episodes:", TEST_EPISODES)
print("Average reward:", np.mean(test_rewards))
print("Best reward:", np.max(test_rewards))
print("Worst reward:", np.min(test_rewards))
print("Standard deviation:", np.std(test_rewards))
print("======================================")


print("\nNow running graphical demo...")
render_rewards = run_test(model, RENDER_EPISODES, render=True)

print("\n======================================")
print("Graphical demo finished.")
print("Render episodes:", RENDER_EPISODES)
print("Average render reward:", np.mean(render_rewards))
print("======================================")