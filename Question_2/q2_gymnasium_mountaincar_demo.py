import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import gymnasium as gym


STATE_DIM = 2
ACTION_DIM = 3


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_WEIGHTS_FILE = BASE_DIR / "double_dqn_aiut_mountaincar_tuned_best.weights.h5"

DEMO_EPISODES = 5


def normalize_state(state):
    position = state[0]
    velocity = state[1]

    position_norm = (position + 1.2) / 1.7
    velocity_norm = (velocity + 0.07) / 0.14

    return np.array([position_norm, velocity_norm], dtype=np.float32)


class QNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.output_layer = tf.keras.layers.Dense(action_dim, activation="linear")

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        q_values = self.output_layer(x)
        return q_values


def select_greedy_action(state, model):
    state_array = np.array([state], dtype=np.float32)
    q_values = model(state_array)

    # خروجی شبکه 0, 1, 2 است
    # در Gymnasium MountainCar نیز اکشن‌ها 0, 1, 2 هستند
    action = int(np.argmax(q_values[0].numpy()))

    return action


print("Question 2: Gymnasium MountainCar graphical demo")
print("Looking for weights at:")
print(MODEL_WEIGHTS_FILE)

if not MODEL_WEIGHTS_FILE.exists():
    raise FileNotFoundError(
        f"Weight file not found: {MODEL_WEIGHTS_FILE}\n"
        "Make sure double_dqn_aiut_mountaincar_tuned_best.weights.h5 is in D:\\Projects\\RlProject"
    )


env = gym.make("MountainCar-v0", render_mode="human")

model = QNetwork(STATE_DIM, ACTION_DIM)

dummy_state = np.zeros((1, STATE_DIM), dtype=np.float32)
model(dummy_state)

model.load_weights(str(MODEL_WEIGHTS_FILE))

print("Loaded weights successfully.")
print("Policy: greedy, epsilon = 0")
print("======================================")


success_count = 0
episode_rewards = []
episode_steps = []

for episode in range(1, DEMO_EPISODES + 1):
    raw_state, info = env.reset()
    state = normalize_state(raw_state)

    total_reward = 0
    steps = 0
    done = False
    terminated_final = False

    while not done:
        action = select_greedy_action(state, model)

        raw_next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        terminated_final = terminated

        state = normalize_state(raw_next_state)

        total_reward += reward
        steps += 1

        time.sleep(0.01)

    if terminated_final:
        success_count += 1

    episode_rewards.append(total_reward)
    episode_steps.append(steps)

    print(
        f"Demo Episode {episode:02d} | "
        f"Reward: {total_reward:6.1f} | "
        f"Steps: {steps:3d} | "
        f"Success: {terminated_final}"
    )

env.close()

print("\n======================================")
print("Gymnasium demo finished.")
print("Success count:", success_count, "/", DEMO_EPISODES)
print("Average reward:", np.mean(episode_rewards))
print("Average steps:", np.mean(episode_steps))
print("======================================")