import random
from collections import deque

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from AIUT_Environments.AIUT_Environments import MountainCar


# =========================
# تنظیمات اصلی پروژه
# =========================

EPISODES = 500
STATE_DIM = 2
ACTION_DIM = 3
ACTION_MAP = [-1, 0, 1]

GAMMA = 0.99
LEARNING_RATE = 0.001

MEMORY_SIZE = 10000
BATCH_SIZE = 64
TRAIN_START = 1000

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

TARGET_UPDATE_FREQ = 1000


# =========================
# مدل شبکه عصبی Q
# =========================

class QNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(16, activation="relu")
        self.output_layer = tf.keras.layers.Dense(action_dim, activation="linear")

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        q_values = self.output_layer(x)
        return q_values


# =========================
# حافظه تجربه‌ها Replay Buffer
# =========================

class ReplayBuffer:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def store(self, state, action_index, reward, next_state, done):
        self.memory.append((state, action_index, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.memory)


# =========================
# انتخاب اکشن با روش epsilon-greedy
# =========================

def select_action(state, model, epsilon):
    if random.random() < epsilon:
        return random.randint(0, ACTION_DIM - 1)

    state_array = np.array([state], dtype=np.float32)
    q_values = model(state_array)
    action_index = int(np.argmax(q_values[0].numpy()))

    return action_index


# =========================
# یک مرحله آموزش با Double DQN
# =========================

def train_step_double_dqn(memory, online_model, target_model, optimizer):
    if memory.size() < TRAIN_START:
        return None

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    # Double DQN:
    # 1) انتخاب اکشن بعدی با online_model
    next_q_online = online_model(next_states)
    next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)

    # 2) ارزیابی همان اکشن با target_model
    next_q_target = target_model(next_states)
    batch_indices = tf.range(BATCH_SIZE, dtype=tf.int32)
    next_indices = tf.stack([batch_indices, next_actions], axis=1)
    selected_next_q = tf.gather_nd(next_q_target, next_indices)

    targets = rewards + (1.0 - dones) * GAMMA * selected_next_q

    with tf.GradientTape() as tape:
        q_values = online_model(states)

        action_indices = tf.stack([batch_indices, actions], axis=1)
        selected_q = tf.gather_nd(q_values, action_indices)

        loss = tf.reduce_mean(tf.square(targets - selected_q))

    gradients = tape.gradient(loss, online_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, online_model.trainable_variables))

    return float(loss.numpy())


# =========================
# تست اولیه تابع آموزش
# =========================

env = MountainCar()

online_model = QNetwork(STATE_DIM, ACTION_DIM)
target_model = QNetwork(STATE_DIM, ACTION_DIM)

dummy_state = np.zeros((1, STATE_DIM), dtype=np.float32)
online_model(dummy_state)
target_model(dummy_state)

target_model.set_weights(online_model.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
memory = ReplayBuffer(MEMORY_SIZE)

print("Environment: AIUT MountainCar")
print("Initial state:", env.reset())
print("Action mapping:", ACTION_MAP)

print("\nFilling replay buffer with random experiences...")

state = env.reset()

while memory.size() < TRAIN_START + BATCH_SIZE:
    action_index = random.randint(0, ACTION_DIM - 1)
    real_action = ACTION_MAP[action_index]

    next_state, reward, terminated, truncated = env.step(real_action)
    done = terminated or truncated

    memory.store(state, action_index, reward, next_state, done)

    state = next_state

    if done:
        state = env.reset()

print("Replay buffer size:", memory.size())

loss = train_step_double_dqn(memory, online_model, target_model, optimizer)

print("\nTesting one Double DQN training step...")
print("Loss:", loss)