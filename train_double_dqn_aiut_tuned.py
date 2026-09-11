import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from AIUT_Environments.AIUT_Environments import MountainCar


# =========================
# کاهش پیام‌های اضافی TensorFlow
# =========================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# =========================
# تنظیمات اصلی پروژه
# =========================

EPISODES = 800          # بیشتر از حداقل 300 برای پایداری بهتر
STATE_DIM = 2           # state = [position, velocity]
ACTION_DIM = 3          # خروجی‌های شبکه برای اکشن‌های -1, 0, 1
ACTION_MAP = [-1, 0, 1]

GAMMA = 0.99
LEARNING_RATE = 0.0005

MEMORY_SIZE = 50000
BATCH_SIZE = 64
TRAIN_START = 2000

EPSILON_START = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.995

TARGET_UPDATE_FREQ = 500

EVAL_EVERY = 25
EVAL_EPISODES = 10

MODEL_WEIGHTS_FILE = "double_dqn_aiut_mountaincar_tuned_best.weights.h5"
REWARD_PLOT_FILE = "double_dqn_aiut_rewards_tuned.png"
LOG_FILE = "double_dqn_aiut_training_log_tuned.csv"


# =========================
# Seed برای تکرارپذیری نسبی
# =========================

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# =========================
# نرمال‌سازی state
# =========================

def normalize_state(state):
    position = state[0]
    velocity = state[1]

    position_norm = (position + 1.2) / 1.7
    velocity_norm = (velocity + 0.07) / 0.14

    return np.array([position_norm, velocity_norm], dtype=np.float32)


# =========================
# reward shaping برای آموزش پایدارتر
# =========================

def shaped_reward(raw_state, raw_next_state, env_reward, terminated):
    position = raw_next_state[0]
    velocity = raw_next_state[1]

    if terminated:
        return 100.0

    # تشویق حرکت و گرفتن انرژی برای بالا رفتن از کوه
    shaped = env_reward
    shaped += 10.0 * abs(velocity)
    shaped += 2.0 * (position + 0.6)

    return shaped


# =========================
# مدل شبکه عصبی Q
# =========================

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


# =========================
# Replay Buffer
# =========================

class ReplayBuffer:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def store(self, state, action_index, reward, next_state, done):
        self.memory.append((state, action_index, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def size(self):
        return len(self.memory)


# =========================
# انتخاب اکشن epsilon-greedy
# =========================

def select_action(state, model, epsilon):
    if random.random() < epsilon:
        return random.randint(0, ACTION_DIM - 1)

    state_array = np.array([state], dtype=np.float32)
    q_values = model(state_array)
    action_index = int(np.argmax(q_values[0].numpy()))

    return action_index


# =========================
# یک مرحله آموزش Double DQN
# =========================

def train_step_double_dqn(memory, online_model, target_model, optimizer, huber_loss):
    if memory.size() < TRAIN_START:
        return None

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
    actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones_tf = tf.convert_to_tensor(dones, dtype=tf.float32)

    # Double DQN:
    # انتخاب اکشن بعدی با online_model
    next_q_online = online_model(next_states_tf)
    next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)

    # ارزیابی همان اکشن با target_model
    next_q_target = target_model(next_states_tf)
    batch_indices = tf.range(BATCH_SIZE, dtype=tf.int32)
    next_indices = tf.stack([batch_indices, next_actions], axis=1)
    selected_next_q = tf.gather_nd(next_q_target, next_indices)

    targets = rewards_tf + (1.0 - dones_tf) * GAMMA * selected_next_q
    targets = tf.stop_gradient(targets)

    with tf.GradientTape() as tape:
        q_values = online_model(states_tf)

        action_indices = tf.stack([batch_indices, actions_tf], axis=1)
        selected_q = tf.gather_nd(q_values, action_indices)

        loss = huber_loss(targets, selected_q)

    gradients = tape.gradient(loss, online_model.trainable_variables)
    gradients = [
        tf.clip_by_norm(g, 10.0) if g is not None else None
        for g in gradients
    ]

    optimizer.apply_gradients(
        (g, v)
        for g, v in zip(gradients, online_model.trainable_variables)
        if g is not None
    )

    return float(loss.numpy())


# =========================
# ارزیابی سیاست بدون exploration
# =========================

def evaluate_policy(model, episodes=10):
    eval_env = MountainCar()

    rewards = []
    successes = 0

    for _ in range(episodes):
        raw_state = eval_env.reset()
        state = normalize_state(raw_state)

        total_reward = 0
        done = False
        terminated_final = False

        while not done:
            action_index = select_action(state, model, epsilon=0.0)
            real_action = ACTION_MAP[action_index]

            raw_next_state, reward, terminated, truncated = eval_env.step(real_action)
            done = terminated or truncated

            state = normalize_state(raw_next_state)
            total_reward += reward
            terminated_final = terminated

        if terminated_final:
            successes += 1

        rewards.append(total_reward)

    return float(np.mean(rewards)), int(successes)


# =========================
# میانگین متحرک برای نمودار
# =========================

def moving_average(values, window_size=20):
    if len(values) < window_size:
        return np.array(values)

    return np.convolve(values, np.ones(window_size) / window_size, mode="valid")


# =========================
# شروع آموزش
# =========================

env = MountainCar()

online_model = QNetwork(STATE_DIM, ACTION_DIM)
target_model = QNetwork(STATE_DIM, ACTION_DIM)

dummy_state = np.zeros((1, STATE_DIM), dtype=np.float32)
online_model(dummy_state)
target_model(dummy_state)

target_model.set_weights(online_model.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
huber_loss = tf.keras.losses.Huber(delta=1.0)

memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPSILON_START
total_steps = 0

rewards_history = []
steps_history = []
epsilon_history = []
loss_history = []
eval_reward_history = []
eval_success_history = []

best_eval_reward = -999999.0
best_success_count = 0

print("======================================")
print("TUNED Question 1: AIUT MountainCar + Double DQN")
print("Episodes:", EPISODES)
print("Action mapping:", ACTION_MAP)
print("Model file:", MODEL_WEIGHTS_FILE)
print("======================================")


for episode in range(1, EPISODES + 1):
    raw_state = env.reset()
    state = normalize_state(raw_state)

    total_env_reward = 0
    episode_steps = 0
    episode_losses = []

    done = False
    terminated_final = False

    while not done:
        action_index = select_action(state, online_model, epsilon)
        real_action = ACTION_MAP[action_index]

        raw_next_state, env_reward, terminated, truncated = env.step(real_action)
        done = terminated or truncated
        terminated_final = terminated

        next_state = normalize_state(raw_next_state)

        train_reward = shaped_reward(raw_state, raw_next_state, env_reward, terminated)

        memory.store(state, action_index, train_reward, next_state, done)

        loss = train_step_double_dqn(memory, online_model, target_model, optimizer, huber_loss)
        if loss is not None:
            episode_losses.append(loss)

        raw_state = raw_next_state
        state = next_state

        total_env_reward += env_reward
        episode_steps += 1
        total_steps += 1

        if total_steps % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(online_model.get_weights())

    if epsilon > EPSILON_MIN:
        epsilon *= EPSILON_DECAY
        epsilon = max(epsilon, EPSILON_MIN)

    avg_loss = np.mean(episode_losses) if len(episode_losses) > 0 else np.nan

    rewards_history.append(total_env_reward)
    steps_history.append(episode_steps)
    epsilon_history.append(epsilon)
    loss_history.append(avg_loss)

    eval_text = ""

    if episode % EVAL_EVERY == 0:
        eval_avg_reward, eval_success_count = evaluate_policy(online_model, EVAL_EPISODES)

        eval_reward_history.append(eval_avg_reward)
        eval_success_history.append(eval_success_count)

        if (
            eval_success_count > best_success_count
            or (
                eval_success_count == best_success_count
                and eval_avg_reward > best_eval_reward
            )
        ):
            best_success_count = eval_success_count
            best_eval_reward = eval_avg_reward
            online_model.save_weights(MODEL_WEIGHTS_FILE)

        eval_text = (
            f" | Eval Avg: {eval_avg_reward:7.2f}"
            f" | Eval Success: {eval_success_count}/{EVAL_EPISODES}"
            f" | Best Eval: {best_eval_reward:7.2f}"
            f" | Best Success: {best_success_count}/{EVAL_EPISODES}"
        )

    print(
        f"Episode {episode:03d} | "
        f"Reward: {total_env_reward:6.1f} | "
        f"Steps: {episode_steps:3d} | "
        f"Success: {terminated_final} | "
        f"Epsilon: {epsilon:.4f} | "
        f"Memory: {memory.size():5d} | "
        f"Loss: {avg_loss}"
        f"{eval_text}"
    )


# اگر هیچ ارزیابی بهتر ذخیره نشده بود، آخرین مدل ذخیره شود
if not os.path.exists(MODEL_WEIGHTS_FILE):
    online_model.save_weights(MODEL_WEIGHTS_FILE)


# =========================
# ذخیره لاگ
# =========================

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("episode,reward,steps,epsilon,loss\n")
    for i in range(EPISODES):
        f.write(
            f"{i + 1},"
            f"{rewards_history[i]},"
            f"{steps_history[i]},"
            f"{epsilon_history[i]},"
            f"{loss_history[i]}\n"
        )


# =========================
# ذخیره نمودار
# =========================

plt.figure(figsize=(10, 5))
plt.plot(rewards_history, label="Episode reward")

ma_rewards = moving_average(rewards_history, window_size=20)
if len(ma_rewards) > 0:
    plt.plot(
        range(19, 19 + len(ma_rewards)),
        ma_rewards,
        label="Moving average (window=20)"
    )

plt.xlabel("Episode")
plt.ylabel("Total environment reward")
plt.title("AIUT MountainCar - Tuned Double DQN Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(REWARD_PLOT_FILE, dpi=300)
plt.close()


print("\n======================================")
print("Tuned training finished.")
print("Best eval reward:", best_eval_reward)
print("Best eval success:", best_success_count, "/", EVAL_EPISODES)
print("Saved best model weights:", MODEL_WEIGHTS_FILE)
print("Saved reward plot:", REWARD_PLOT_FILE)
print("Saved training log:", LOG_FILE)
print("======================================")