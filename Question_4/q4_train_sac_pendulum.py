import csv
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# =========================
# تنظیمات سؤال 4 - SAC
# =========================

ENV_NAME = "Pendulum-v1"

STATE_DIM = 3
ACTION_DIM = 1
ACTION_LOW = -2.0
ACTION_HIGH = 2.0

EPISODES = 500
MAX_STEPS_PER_EPISODE = 200

GAMMA = 0.99
TAU = 0.005

ACTOR_LR = 0.0003
CRITIC_LR = 0.0003
ALPHA_LR = 0.0003

MEMORY_SIZE = 200000
BATCH_SIZE = 256
WARMUP_STEPS = 5000
UPDATES_PER_STEP = 1

EVAL_EVERY = 10
EVAL_EPISODES = 10

TARGET_ENTROPY = -ACTION_DIM

LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0

BASE_DIR = Path(__file__).resolve().parent.parent

ACTOR_WEIGHTS_FILE = BASE_DIR / "sac_pendulum_actor_best.weights.h5"
REWARD_PLOT_FILE = BASE_DIR / "sac_pendulum_rewards.png"
LOG_FILE = BASE_DIR / "sac_pendulum_training_log.csv"


# =========================
# Seed
# =========================

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# =========================
# نرمال‌سازی state
# =========================

def normalize_state(state):
    cos_theta = state[0]
    sin_theta = state[1]
    angular_velocity = state[2] / 8.0

    return np.array([cos_theta, sin_theta, angular_velocity], dtype=np.float32)


# =========================
# Replay Buffer
# =========================

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def store(self, state, action_norm, reward, next_state, terminal):
        self.buffer.append((state, action_norm, reward, next_state, terminal))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, terminals = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(terminals, dtype=np.float32)
        )

    def size(self):
        return len(self.buffer)


# =========================
# Actor: Squashed Gaussian Policy
# =========================

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
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample_action(self, state):
        mean, log_std = self(state)

        std = tf.exp(log_std)
        noise = tf.random.normal(shape=tf.shape(mean))

        z = mean + std * noise
        action_norm = tf.tanh(z)

        log_prob = -0.5 * (
            tf.square((z - mean) / (std + 1e-8))
            + 2.0 * log_std
            + tf.math.log(2.0 * np.pi)
        )

        log_prob = tf.reduce_sum(log_prob, axis=1)

        # correction برای tanh squash
        correction = tf.reduce_sum(
            tf.math.log(1.0 - tf.square(action_norm) + 1e-6),
            axis=1
        )

        log_prob = log_prob - correction

        return action_norm, log_prob

    def deterministic_action(self, state):
        mean, _ = self(state)
        action_norm = tf.tanh(mean)
        return action_norm


# =========================
# Critic Q Network
# =========================

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(256, activation="relu")
        self.q_layer = tf.keras.layers.Dense(1, activation="linear")

    def call(self, state, action_norm):
        x = tf.concat([state, action_norm], axis=1)

        x = self.dense1(x)
        x = self.dense2(x)

        q_value = self.q_layer(x)

        return q_value


# =========================
# توابع کمکی اکشن
# =========================

def scale_action_to_env(action_norm):
    action_env = action_norm * ACTION_HIGH
    action_env = np.clip(action_env, ACTION_LOW, ACTION_HIGH)
    return action_env.astype(np.float32)


def select_random_action():
    action_env = np.random.uniform(ACTION_LOW, ACTION_HIGH, size=(ACTION_DIM,))
    action_norm = action_env / ACTION_HIGH

    return action_env.astype(np.float32), action_norm.astype(np.float32)


def select_sac_action(state, actor):
    state_array = np.array([state], dtype=np.float32)
    state_tf = tf.convert_to_tensor(state_array, dtype=tf.float32)

    action_norm_tf, _ = actor.sample_action(state_tf)
    action_norm = action_norm_tf[0].numpy()

    action_env = scale_action_to_env(action_norm)

    return action_env, action_norm.astype(np.float32)


def select_deterministic_action(state, actor):
    state_array = np.array([state], dtype=np.float32)
    state_tf = tf.convert_to_tensor(state_array, dtype=tf.float32)

    action_norm_tf = actor.deterministic_action(state_tf)
    action_norm = action_norm_tf[0].numpy()

    action_env = scale_action_to_env(action_norm)

    return action_env


# =========================
# Soft Update
# =========================

def soft_update(target_model, source_model, tau):
    target_weights = target_model.get_weights()
    source_weights = source_model.get_weights()

    new_weights = []

    for target_w, source_w in zip(target_weights, source_weights):
        new_w = tau * source_w + (1.0 - tau) * target_w
        new_weights.append(new_w)

    target_model.set_weights(new_weights)


# =========================
# یک مرحله آپدیت SAC
# =========================

def sac_update(
    replay_buffer,
    actor,
    critic1,
    critic2,
    target_critic1,
    target_critic2,
    actor_optimizer,
    critic1_optimizer,
    critic2_optimizer,
    alpha_optimizer,
    log_alpha
):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(BATCH_SIZE)

    states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
    actions_tf = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
    terminals_tf = tf.convert_to_tensor(terminals, dtype=tf.float32)

    alpha = tf.exp(log_alpha)

    # -------------------------
    # Target Q
    # -------------------------
    next_actions, next_log_probs = actor.sample_action(next_states_tf)

    target_q1 = target_critic1(next_states_tf, next_actions)
    target_q2 = target_critic2(next_states_tf, next_actions)

    target_min_q = tf.minimum(target_q1, target_q2)
    target_min_q = tf.squeeze(target_min_q, axis=1)

    target_value = target_min_q - alpha * next_log_probs

    q_target = rewards_tf + GAMMA * (1.0 - terminals_tf) * target_value
    q_target = tf.stop_gradient(q_target)

    # -------------------------
    # Critic 1 Update
    # -------------------------
    with tf.GradientTape() as tape1:
        current_q1 = critic1(states_tf, actions_tf)
        current_q1 = tf.squeeze(current_q1, axis=1)

        critic1_loss = tf.reduce_mean(tf.square(q_target - current_q1))

    critic1_grads = tape1.gradient(critic1_loss, critic1.trainable_variables)
    critic1_optimizer.apply_gradients(zip(critic1_grads, critic1.trainable_variables))

    # -------------------------
    # Critic 2 Update
    # -------------------------
    with tf.GradientTape() as tape2:
        current_q2 = critic2(states_tf, actions_tf)
        current_q2 = tf.squeeze(current_q2, axis=1)

        critic2_loss = tf.reduce_mean(tf.square(q_target - current_q2))

    critic2_grads = tape2.gradient(critic2_loss, critic2.trainable_variables)
    critic2_optimizer.apply_gradients(zip(critic2_grads, critic2.trainable_variables))

    # -------------------------
    # Actor Update
    # -------------------------
    with tf.GradientTape() as tape_actor:
        new_actions, log_probs = actor.sample_action(states_tf)

        q1_new = critic1(states_tf, new_actions)
        q2_new = critic2(states_tf, new_actions)

        min_q_new = tf.minimum(q1_new, q2_new)
        min_q_new = tf.squeeze(min_q_new, axis=1)

        actor_loss = tf.reduce_mean(alpha * log_probs - min_q_new)

    actor_grads = tape_actor.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    # -------------------------
    # Alpha / Entropy Update
    # -------------------------
    with tf.GradientTape() as tape_alpha:
        _, log_probs_for_alpha = actor.sample_action(states_tf)

        alpha_loss = -tf.reduce_mean(
            log_alpha * tf.stop_gradient(log_probs_for_alpha + TARGET_ENTROPY)
        )

    alpha_grad = tape_alpha.gradient(alpha_loss, [log_alpha])
    alpha_optimizer.apply_gradients(zip(alpha_grad, [log_alpha]))

    return (
        float(actor_loss.numpy()),
        float(critic1_loss.numpy()),
        float(critic2_loss.numpy()),
        float(alpha_loss.numpy()),
        float(tf.exp(log_alpha).numpy())
    )


# =========================
# ارزیابی deterministic
# =========================

def evaluate_policy(actor, episodes=10, render=False):
    if render:
        eval_env = gym.make(ENV_NAME, render_mode="human")
    else:
        eval_env = gym.make(ENV_NAME)

    rewards = []

    for _ in range(episodes):
        raw_state, info = eval_env.reset()
        state = normalize_state(raw_state)

        total_reward = 0.0
        done = False

        while not done:
            action_env = select_deterministic_action(state, actor)

            raw_next_state, reward, terminated, truncated, info = eval_env.step(action_env)

            done = terminated or truncated
            state = normalize_state(raw_next_state)

            total_reward += reward

        rewards.append(total_reward)

    eval_env.close()

    return float(np.mean(rewards)), float(np.max(rewards)), float(np.min(rewards))


def moving_average(values, window_size=20):
    if len(values) < window_size:
        return np.array(values)

    return np.convolve(values, np.ones(window_size) / window_size, mode="valid")


# =========================
# شروع آموزش SAC
# =========================

env = gym.make(ENV_NAME)

actor = Actor(STATE_DIM, ACTION_DIM)

critic1 = Critic(STATE_DIM, ACTION_DIM)
critic2 = Critic(STATE_DIM, ACTION_DIM)

target_critic1 = Critic(STATE_DIM, ACTION_DIM)
target_critic2 = Critic(STATE_DIM, ACTION_DIM)

dummy_state = np.zeros((1, STATE_DIM), dtype=np.float32)
dummy_action = np.zeros((1, ACTION_DIM), dtype=np.float32)

actor(dummy_state)
critic1(dummy_state, dummy_action)
critic2(dummy_state, dummy_action)
target_critic1(dummy_state, dummy_action)
target_critic2(dummy_state, dummy_action)

target_critic1.set_weights(critic1.get_weights())
target_critic2.set_weights(critic2.get_weights())

actor_optimizer = tf.keras.optimizers.Adam(learning_rate=ACTOR_LR)
critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=CRITIC_LR)
critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=CRITIC_LR)
alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA_LR)

log_alpha = tf.Variable(np.log(0.2), dtype=tf.float32, trainable=True)

replay_buffer = ReplayBuffer(MEMORY_SIZE)

episode_rewards = []
episode_steps = []
eval_rewards = []

actor_losses = []
critic1_losses = []
critic2_losses = []
alpha_values = []

total_steps = 0
best_eval_reward = -999999.0

print("======================================")
print("Question 4: Pendulum-v1 with SAC")
print("Environment:", ENV_NAME)
print("Episodes:", EPISODES)
print("Warmup steps:", WARMUP_STEPS)
print("Replay buffer size:", MEMORY_SIZE)
print("Batch size:", BATCH_SIZE)
print("======================================")


for episode in range(1, EPISODES + 1):
    raw_state, info = env.reset()
    state = normalize_state(raw_state)

    total_reward = 0.0
    steps = 0
    done = False

    episode_actor_losses = []
    episode_critic1_losses = []
    episode_critic2_losses = []
    episode_alpha_values = []

    while not done and steps < MAX_STEPS_PER_EPISODE:
        if total_steps < WARMUP_STEPS:
            action_env, action_norm = select_random_action()
        else:
            action_env, action_norm = select_sac_action(state, actor)

        raw_next_state, reward, terminated, truncated, info = env.step(action_env)

        done = terminated or truncated
        terminal = float(terminated)

        next_state = normalize_state(raw_next_state)

        # پاداش برای آموزش مقیاس‌دهی می‌شود؛ پاداش گزارش اصلی حفظ می‌شود.
        train_reward = reward / 10.0

        replay_buffer.store(
            state,
            action_norm,
            train_reward,
            next_state,
            terminal
        )

        state = next_state
        total_reward += reward
        steps += 1
        total_steps += 1

        if replay_buffer.size() >= BATCH_SIZE and total_steps >= WARMUP_STEPS:
            for _ in range(UPDATES_PER_STEP):
                a_loss, c1_loss, c2_loss, alpha_loss, alpha_value = sac_update(
                    replay_buffer,
                    actor,
                    critic1,
                    critic2,
                    target_critic1,
                    target_critic2,
                    actor_optimizer,
                    critic1_optimizer,
                    critic2_optimizer,
                    alpha_optimizer,
                    log_alpha
                )

                soft_update(target_critic1, critic1, TAU)
                soft_update(target_critic2, critic2, TAU)

                episode_actor_losses.append(a_loss)
                episode_critic1_losses.append(c1_loss)
                episode_critic2_losses.append(c2_loss)
                episode_alpha_values.append(alpha_value)

    episode_rewards.append(total_reward)
    episode_steps.append(steps)

    avg_actor_loss = np.mean(episode_actor_losses) if len(episode_actor_losses) > 0 else np.nan
    avg_critic1_loss = np.mean(episode_critic1_losses) if len(episode_critic1_losses) > 0 else np.nan
    avg_critic2_loss = np.mean(episode_critic2_losses) if len(episode_critic2_losses) > 0 else np.nan
    avg_alpha_value = np.mean(episode_alpha_values) if len(episode_alpha_values) > 0 else float(tf.exp(log_alpha).numpy())

    actor_losses.append(avg_actor_loss)
    critic1_losses.append(avg_critic1_loss)
    critic2_losses.append(avg_critic2_loss)
    alpha_values.append(avg_alpha_value)

    eval_text = ""

    if episode % EVAL_EVERY == 0:
        eval_avg, eval_best, eval_worst = evaluate_policy(
            actor,
            episodes=EVAL_EPISODES,
            render=False
        )

        eval_rewards.append(eval_avg)

        if eval_avg > best_eval_reward:
            best_eval_reward = eval_avg
            actor.save_weights(str(ACTOR_WEIGHTS_FILE))

        eval_text = (
            f" | Eval Avg: {eval_avg:8.2f}"
            f" | Eval Best: {eval_best:8.2f}"
            f" | Eval Worst: {eval_worst:8.2f}"
            f" | Best Eval: {best_eval_reward:8.2f}"
        )

    print(
        f"Episode {episode:03d} | "
        f"Reward: {total_reward:8.2f} | "
        f"Steps: {steps:3d} | "
        f"Memory: {replay_buffer.size():6d} | "
        f"Actor Loss: {avg_actor_loss:8.4f} | "
        f"Critic Loss: {avg_critic1_loss:8.4f} | "
        f"Alpha: {avg_alpha_value:7.4f}"
        f"{eval_text}"
    )


env.close()

if not ACTOR_WEIGHTS_FILE.exists():
    actor.save_weights(str(ACTOR_WEIGHTS_FILE))


# =========================
# ذخیره لاگ
# =========================

with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "episode",
        "reward",
        "steps",
        "actor_loss",
        "critic1_loss",
        "critic2_loss",
        "alpha"
    ])

    for i in range(len(episode_rewards)):
        writer.writerow([
            i + 1,
            episode_rewards[i],
            episode_steps[i],
            actor_losses[i],
            critic1_losses[i],
            critic2_losses[i],
            alpha_values[i]
        ])


# =========================
# رسم نمودار
# =========================

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Episode reward")

ma_rewards = moving_average(episode_rewards, window_size=20)

if len(ma_rewards) > 0:
    plt.plot(
        range(19, 19 + len(ma_rewards)),
        ma_rewards,
        label="Moving average (window=20)"
    )

plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Pendulum-v1 - SAC Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(REWARD_PLOT_FILE, dpi=300)
plt.close()


print("\n======================================")
print("SAC training finished.")
print("Best eval reward:", best_eval_reward)
print("Saved best actor weights:", ACTOR_WEIGHTS_FILE)
print("Saved reward plot:", REWARD_PLOT_FILE)
print("Saved training log:", LOG_FILE)
print("======================================")