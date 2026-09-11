import numpy as np
import tensorflow as tf

from AIUT_Environments.AIUT_Environments import MountainCar


STATE_DIM = 2
ACTION_DIM = 3
ACTION_MAP = [-1, 0, 1]

MODEL_WEIGHTS_FILE = "double_dqn_aiut_mountaincar_tuned_best.weights.h5"
TEST_EPISODES = 20


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
    action_index = int(np.argmax(q_values[0].numpy()))
    return action_index


env = MountainCar()

model = QNetwork(STATE_DIM, ACTION_DIM)

dummy_state = np.zeros((1, STATE_DIM), dtype=np.float32)
model(dummy_state)

model.load_weights(MODEL_WEIGHTS_FILE)

print("Loaded tuned best weights from:", MODEL_WEIGHTS_FILE)
print("Testing greedy policy with epsilon = 0")
print("======================================")


test_rewards = []
test_steps = []
success_count = 0

for episode in range(1, TEST_EPISODES + 1):
    raw_state = env.reset()
    state = normalize_state(raw_state)

    total_reward = 0
    steps = 0
    done = False
    terminated_final = False

    while not done:
        action_index = select_greedy_action(state, model)
        real_action = ACTION_MAP[action_index]

        raw_next_state, reward, terminated, truncated = env.step(real_action)

        done = terminated or truncated
        terminated_final = terminated

        state = normalize_state(raw_next_state)

        total_reward += reward
        steps += 1

    if terminated_final:
        success_count += 1

    test_rewards.append(total_reward)
    test_steps.append(steps)

    print(
        f"Test Episode {episode:02d} | "
        f"Reward: {total_reward:6.1f} | "
        f"Steps: {steps:3d} | "
        f"Success: {terminated_final}"
    )


print("\n======================================")
print("Final tuned test finished.")
print("Success count:", success_count, "/", TEST_EPISODES)
print("Average reward:", np.mean(test_rewards))
print("Best reward:", np.max(test_rewards))
print("Worst reward:", np.min(test_rewards))
print("Average steps:", np.mean(test_steps))
print("======================================")