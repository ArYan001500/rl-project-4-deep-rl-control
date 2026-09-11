import csv
import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# =========================
# تنظیمات اصلی سؤال 3
# =========================

ENV_NAME = "Pendulum-v1"

STATE_DIM = 3
ACTION_DIM = 1
ACTION_LOW = -2.0
ACTION_HIGH = 2.0

EPISODES = 500
GAMMA = 0.99
LAMBDA_GAE = 0.95

LEARNING_RATE = 0.0003
PPO_EPOCHS = 10
CLIP_RATIO = 0.2

VALUE_COEF = 0.5
ENTROPY_COEF = 0.001

EVAL_EVERY = 25
EVAL_EPISODES = 5

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_WEIGHTS_FILE = BASE_DIR / "ppo_pendulum_best.weights.h5"
REWARD_PLOT_FILE = BASE_DIR / "ppo_pendulum_rewards.png"
LOG_FILE = BASE_DIR / "ppo_pendulum_training_log.csv"


# =========================
# ثابت کردن seed
# =========================

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# =========================
# نرمال‌سازی state
# =========================

def normalize_state(state):
    # state = [cos(theta), sin(theta), angular_velocity]
    cos_theta = state[0]
    sin_theta = state[1]
    angular_velocity = state[2] / 8.0

    return np.array([cos_theta, sin_theta, angular_velocity], dtype=np.float32)


# =========================
# مدل Actor-Critic برای PPO
# =========================

class ActorCritic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.common1 = tf.keras.layers.Dense(64, activation="tanh")
        self.common2 = tf.keras.layers.Dense(64, activation="tanh")

        self.mean_layer = tf.keras.layers.Dense(action_dim, activation="tanh")
        self.value_layer = tf.keras.layers.Dense(1, activation="linear")

        self.log_std = self.add_weight(
            name="log_std",
            shape=(action_dim,),
            initializer=tf.keras.initializers.Constant(-0.5),
            trainable=True
        )

    def call(self, state):
        x = self.common1(state)
        x = self.common2(x)

        # خروجی tanh در بازه [-1, 1] است؛ ضربدر 2 می‌شود بازه اکشن Pendulum
        mean = self.mean_layer(x) * ACTION_HIGH
        value = self.value_layer(x)

        return mean, value


def gaussian_log_prob(action, mean, log_std):
    std = tf.exp(log_std)

    log_prob = -0.5 * (
        tf.square((action - mean) / std)
        + 2.0 * log_std
        + tf.math.log(2.0 * np.pi)
    )

    return tf.reduce_sum(log_prob, axis=1)


def gaussian_entropy(log_std):
    return tf.reduce_sum(
        0.5 * (1.0 + tf.math.log(2.0 * np.pi)) + log_std
    )


def select_action(state, model):
    state_array = np.array([state], dtype=np.float32)

    mean, value = model(state_array)
    mean = mean[0]

    std = tf.exp(model.log_std)
    noise = tf.random.normal(shape=(ACTION_DIM,))

    action = mean + std * noise
    action = tf.clip_by_value(action, ACTION_LOW, ACTION_HIGH)

    action_array = tf.reshape(action, (1, ACTION_DIM))
    mean_array = tf.reshape(mean, (1, ACTION_DIM))

    log_prob = gaussian_log_prob(action_array, mean_array, model.log_std)

    return (
        action.numpy().astype(np.float32),
        float(log_prob.numpy()[0]),
        float(value.numpy()[0, 0])
    )


def select_deterministic_action(state, model):
    state_array = np.array([state], dtype=np.float32)

    mean, _ = model(state_array)
    action = mean[0].numpy()
    action = np.clip(action, ACTION_LOW, ACTION_HIGH)

    return action.astype(np.float32)


# =========================
# محاسبه advantage با GAE
# =========================

def compute_gae(rewards, values, dones):
    advantages = []
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]

        delta = rewards[t] + GAMMA * next_value * mask - values[t]
        gae = delta + GAMMA * LAMBDA_GAE * mask * gae

        advantages.insert(0, gae)
        next_value = values[t]

    advantages = np.array(advantages, dtype=np.float32)
    returns = advantages + np.array(values, dtype=np.float32)

    return advantages, returns


# =========================
# یک مرحله آپدیت PPO
# =========================

def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages):
    states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
    actions_tf = tf.convert_to_tensor(actions, dtype=tf.float32)
    old_log_probs_tf = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
    returns_tf = tf.convert_to_tensor(returns, dtype=tf.float32)
    advantages_tf = tf.convert_to_tensor(advantages, dtype=tf.float32)

    advantages_tf = (advantages_tf - tf.reduce_mean(advantages_tf)) / (
        tf.math.reduce_std(advantages_tf) + 1e-8
    )

    total_losses = []

    for _ in range(PPO_EPOCHS):
        with tf.GradientTape() as tape:
            mean, values = model(states_tf)
            values = tf.squeeze(values, axis=1)

            new_log_probs = gaussian_log_prob(actions_tf, mean, model.log_std)

            ratio = tf.exp(new_log_probs - old_log_probs_tf)

            clipped_ratio = tf.clip_by_value(
                ratio,
                1.0 - CLIP_RATIO,
                1.0 + CLIP_RATIO
            )

            surrogate1 = ratio * advantages_tf
            surrogate2 = clipped_ratio * advantages_tf

            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            critic_loss = tf.reduce_mean(tf.square(returns_tf - values))
            entropy = gaussian_entropy(model.log_std)

            total_loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

        gradients = tape.gradient(total_loss, model.trainable_variables)
        gradients = [
            tf.clip_by_norm(g, 0.5) if g is not None else None
            for g in gradients
        ]

        optimizer.apply_gradients(
            (g, v)
            for g, v in zip(gradients, model.trainable_variables)
            if g is not None
        )

        total_losses.append(float(total_loss.numpy()))

    return float(np.mean(total_losses))


# =========================
# ارزیابی بدون exploration
# =========================

def evaluate_policy(model, episodes=5):
    eval_env = gym.make(ENV_NAME)

    rewards = []

    for _ in range(episodes):
        raw_state, info = eval_env.reset()
        state = normalize_state(raw_state)

        done = False
        total_reward = 0.0

        while not done:
            action = select_deterministic_action(state, model)

            raw_next_state, reward, terminated, truncated, info = eval_env.step(action)

            done = terminated or truncated
            state = normalize_state(raw_next_state)

            total_reward += reward

        rewards.append(total_reward)

    eval_env.close()

    return float(np.mean(rewards))


def moving_average(values, window_size=20):
    if len(values) < window_size:
        return np.array(values)

    return np.convolve(values, np.ones(window_size) / window_size, mode="valid")


# =========================
# شروع آموزش PPO
# =========================

env = gym.make(ENV_NAME)

model = ActorCritic(STATE_DIM, ACTION_DIM)

dummy_state = np.zeros((1, STATE_DIM), dtype=np.float32)
model(dummy_state)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

episode_rewards = []
episode_steps = []
loss_history = []
eval_history = []

best_eval_reward = -999999.0

print("======================================")
print("Question 3: Pendulum-v1 with PPO")
print("Environment:", ENV_NAME)
print("Episodes:", EPISODES)
print("Action range:", ACTION_LOW, "to", ACTION_HIGH)
print("======================================")


for episode in range(1, EPISODES + 1):
    raw_state, info = env.reset()
    state = normalize_state(raw_state)

    states = []
    actions = []
    rewards_for_training = []
    rewards_original = []
    dones = []
    old_log_probs = []
    values = []

    done = False
    steps = 0

    while not done:
        action, log_prob, value = select_action(state, model)

        raw_next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        next_state = normalize_state(raw_next_state)

        # برای پایداری آموزش، پاداش را کوچک‌تر می‌کنیم
        train_reward = reward / 10.0

        states.append(state)
        actions.append(action)
        rewards_for_training.append(train_reward)
        rewards_original.append(reward)
        dones.append(float(done))
        old_log_probs.append(log_prob)
        values.append(value)

        state = next_state
        steps += 1

    advantages, returns = compute_gae(
        rewards_for_training,
        values,
        dones
    )

    loss = ppo_update(
        model,
        optimizer,
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(old_log_probs, dtype=np.float32),
        returns,
        advantages
    )

    total_reward = float(np.sum(rewards_original))

    episode_rewards.append(total_reward)
    episode_steps.append(steps)
    loss_history.append(loss)

    eval_text = ""

    if episode % EVAL_EVERY == 0:
        eval_reward = evaluate_policy(model, EVAL_EPISODES)
        eval_history.append(eval_reward)

        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            model.save_weights(str(MODEL_WEIGHTS_FILE))

        eval_text = (
            f" | Eval Avg Reward: {eval_reward:8.2f}"
            f" | Best Eval: {best_eval_reward:8.2f}"
        )

    print(
        f"Episode {episode:03d} | "
        f"Reward: {total_reward:8.2f} | "
        f"Steps: {steps:3d} | "
        f"Loss: {loss:8.4f}"
        f"{eval_text}"
    )


env.close()

if not MODEL_WEIGHTS_FILE.exists():
    model.save_weights(str(MODEL_WEIGHTS_FILE))


# =========================
# ذخیره لاگ
# =========================

with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "steps", "loss"])

    for i in range(len(episode_rewards)):
        writer.writerow([
            i + 1,
            episode_rewards[i],
            episode_steps[i],
            loss_history[i]
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
plt.title("Pendulum-v1 - PPO Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(REWARD_PLOT_FILE, dpi=300)
plt.close()


print("\n======================================")
print("PPO training finished.")
print("Best eval reward:", best_eval_reward)
print("Saved best model weights:", MODEL_WEIGHTS_FILE)
print("Saved reward plot:", REWARD_PLOT_FILE)
print("Saved training log:", LOG_FILE)
print("======================================")