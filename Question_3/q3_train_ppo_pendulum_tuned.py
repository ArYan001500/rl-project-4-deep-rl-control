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

TOTAL_ITERATIONS = 80
ROLLOUT_STEPS = 2048

GAMMA = 0.99
LAMBDA_GAE = 0.95

LEARNING_RATE = 0.00025
PPO_EPOCHS = 10
MINIBATCH_SIZE = 128
CLIP_RATIO = 0.2

VALUE_COEF = 0.5
ENTROPY_COEF = 0.005
MAX_GRAD_NORM = 0.5

EVAL_EVERY = 5
EVAL_EPISODES = 5

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_WEIGHTS_FILE = BASE_DIR / "ppo_pendulum_tuned_best.weights.h5"
REWARD_PLOT_FILE = BASE_DIR / "ppo_pendulum_rewards_tuned.png"
LOG_FILE = BASE_DIR / "ppo_pendulum_training_log_tuned.csv"


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
# مدل Actor-Critic
# =========================

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
            initializer=tf.keras.initializers.Constant(-0.3),
            trainable=True
        )

    def call(self, state):
        x = self.shared1(state)
        x = self.shared2(x)

        # mean در بازه [-1, 1] تولید می‌شود
        mean_normalized = self.mean_layer(x)

        value = self.value_layer(x)

        return mean_normalized, value


# =========================
# توابع احتمال برای سیاست Gaussian
# =========================

def gaussian_log_prob(action_normalized, mean_normalized, log_std):
    std = tf.exp(log_std)

    log_prob = -0.5 * (
        tf.square((action_normalized - mean_normalized) / std)
        + 2.0 * log_std
        + tf.math.log(2.0 * np.pi)
    )

    return tf.reduce_sum(log_prob, axis=1)


def gaussian_entropy(log_std):
    entropy = 0.5 * (1.0 + tf.math.log(2.0 * np.pi)) + log_std
    return tf.reduce_sum(entropy)


# =========================
# انتخاب اکشن stochastic برای آموزش
# =========================

def select_action(state, model):
    state_array = np.array([state], dtype=np.float32)

    mean_normalized, value = model(state_array)
    mean_normalized = mean_normalized[0]

    std = tf.exp(model.log_std)
    noise = tf.random.normal(shape=(ACTION_DIM,))

    action_normalized = mean_normalized + std * noise
    action_normalized = tf.clip_by_value(action_normalized, -1.0, 1.0)

    action_env = action_normalized * ACTION_HIGH

    log_prob = gaussian_log_prob(
        tf.reshape(action_normalized, (1, ACTION_DIM)),
        tf.reshape(mean_normalized, (1, ACTION_DIM)),
        model.log_std
    )

    return (
        action_env.numpy().astype(np.float32),
        action_normalized.numpy().astype(np.float32),
        float(log_prob.numpy()[0]),
        float(value.numpy()[0, 0])
    )


# =========================
# انتخاب اکشن deterministic برای تست
# =========================

def select_deterministic_action(state, model):
    state_array = np.array([state], dtype=np.float32)

    mean_normalized, _ = model(state_array)
    action_normalized = mean_normalized[0].numpy()

    action_env = action_normalized * ACTION_HIGH
    action_env = np.clip(action_env, ACTION_LOW, ACTION_HIGH)

    return action_env.astype(np.float32)


# =========================
# محاسبه GAE با bootstrap
# =========================

def compute_gae(rewards, values, dones, last_value):
    advantages = np.zeros_like(rewards, dtype=np.float32)

    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + GAMMA * next_value * next_non_terminal - values[t]
        gae = delta + GAMMA * LAMBDA_GAE * next_non_terminal * gae

        advantages[t] = gae

    returns = advantages + values

    return advantages.astype(np.float32), returns.astype(np.float32)


# =========================
# جمع‌آوری rollout
# =========================

def collect_rollout(env, model, start_state, start_raw_state):
    states = []
    actions_normalized = []
    old_log_probs = []
    rewards_scaled = []
    rewards_original = []
    dones = []
    values = []

    episode_rewards = []
    episode_steps = []

    current_state = start_state
    current_raw_state = start_raw_state

    current_episode_reward = 0.0
    current_episode_steps = 0

    for _ in range(ROLLOUT_STEPS):
        action_env, action_norm, log_prob, value = select_action(current_state, model)

        next_raw_state, reward, terminated, truncated, info = env.step(action_env)

        done = terminated or truncated
        next_state = normalize_state(next_raw_state)

        # پاداش برای آموزش مقیاس‌بندی می‌شود؛ پاداش گزارش اصلی دست‌نخورده می‌ماند
        reward_scaled = reward / 100.0

        states.append(current_state)
        actions_normalized.append(action_norm)
        old_log_probs.append(log_prob)
        rewards_scaled.append(reward_scaled)
        rewards_original.append(reward)
        dones.append(float(done))
        values.append(value)

        current_episode_reward += reward
        current_episode_steps += 1

        current_state = next_state
        current_raw_state = next_raw_state

        if done:
            episode_rewards.append(current_episode_reward)
            episode_steps.append(current_episode_steps)

            current_raw_state, info = env.reset()
            current_state = normalize_state(current_raw_state)

            current_episode_reward = 0.0
            current_episode_steps = 0

    # bootstrap value برای state آخر rollout
    state_array = np.array([current_state], dtype=np.float32)
    _, last_value = model(state_array)
    last_value = float(last_value.numpy()[0, 0])

    rollout = {
        "states": np.array(states, dtype=np.float32),
        "actions_normalized": np.array(actions_normalized, dtype=np.float32),
        "old_log_probs": np.array(old_log_probs, dtype=np.float32),
        "rewards_scaled": np.array(rewards_scaled, dtype=np.float32),
        "rewards_original": np.array(rewards_original, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "last_value": last_value,
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "last_state": current_state,
        "last_raw_state": current_raw_state
    }

    return rollout


# =========================
# آپدیت PPO با mini-batch
# =========================

def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages):
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    dataset_size = states.shape[0]
    indices = np.arange(dataset_size)

    losses = []

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(indices)

        for start in range(0, dataset_size, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            batch_idx = indices[start:end]

            states_tf = tf.convert_to_tensor(states[batch_idx], dtype=tf.float32)
            actions_tf = tf.convert_to_tensor(actions[batch_idx], dtype=tf.float32)
            old_log_probs_tf = tf.convert_to_tensor(old_log_probs[batch_idx], dtype=tf.float32)
            returns_tf = tf.convert_to_tensor(returns[batch_idx], dtype=tf.float32)
            advantages_tf = tf.convert_to_tensor(advantages[batch_idx], dtype=tf.float32)

            with tf.GradientTape() as tape:
                mean_normalized, values = model(states_tf)
                values = tf.squeeze(values, axis=1)

                new_log_probs = gaussian_log_prob(
                    actions_tf,
                    mean_normalized,
                    model.log_std
                )

                ratio = tf.exp(new_log_probs - old_log_probs_tf)

                clipped_ratio = tf.clip_by_value(
                    ratio,
                    1.0 - CLIP_RATIO,
                    1.0 + CLIP_RATIO
                )

                actor_loss = -tf.reduce_mean(
                    tf.minimum(
                        ratio * advantages_tf,
                        clipped_ratio * advantages_tf
                    )
                )

                critic_loss = tf.reduce_mean(tf.square(returns_tf - values))

                entropy = gaussian_entropy(model.log_std)

                total_loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

            gradients = tape.gradient(total_loss, model.trainable_variables)
            gradients = [
                tf.clip_by_norm(g, MAX_GRAD_NORM) if g is not None else None
                for g in gradients
            ]

            optimizer.apply_gradients(
                (g, v)
                for g, v in zip(gradients, model.trainable_variables)
                if g is not None
            )

            losses.append(float(total_loss.numpy()))

    return float(np.mean(losses))


# =========================
# ارزیابی نهایی بدون exploration
# =========================

def evaluate_policy(model, episodes=5, render=False):
    if render:
        eval_env = gym.make(ENV_NAME, render_mode="human")
    else:
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
# شروع آموزش
# =========================

env = gym.make(ENV_NAME)

model = ActorCritic(STATE_DIM, ACTION_DIM)
dummy_state = np.zeros((1, STATE_DIM), dtype=np.float32)
model(dummy_state)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

raw_state, info = env.reset()
state = normalize_state(raw_state)

all_episode_rewards = []
all_episode_steps = []
iteration_losses = []
eval_rewards = []

best_eval_reward = -999999.0

print("======================================")
print("TUNED Question 3: Pendulum-v1 with PPO")
print("Environment:", ENV_NAME)
print("Iterations:", TOTAL_ITERATIONS)
print("Rollout steps:", ROLLOUT_STEPS)
print("Total training steps:", TOTAL_ITERATIONS * ROLLOUT_STEPS)
print("======================================")


for iteration in range(1, TOTAL_ITERATIONS + 1):
    rollout = collect_rollout(env, model, state, raw_state)

    state = rollout["last_state"]
    raw_state = rollout["last_raw_state"]

    advantages, returns = compute_gae(
        rollout["rewards_scaled"],
        rollout["values"],
        rollout["dones"],
        rollout["last_value"]
    )

    loss = ppo_update(
        model,
        optimizer,
        rollout["states"],
        rollout["actions_normalized"],
        rollout["old_log_probs"],
        returns,
        advantages
    )

    all_episode_rewards.extend(rollout["episode_rewards"])
    all_episode_steps.extend(rollout["episode_steps"])
    iteration_losses.append(loss)

    recent_reward = np.mean(all_episode_rewards[-10:]) if len(all_episode_rewards) >= 10 else np.mean(all_episode_rewards)

    eval_text = ""

    if iteration % EVAL_EVERY == 0:
        eval_reward = evaluate_policy(model, episodes=EVAL_EPISODES, render=False)
        eval_rewards.append(eval_reward)

        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            model.save_weights(str(MODEL_WEIGHTS_FILE))

        eval_text = (
            f" | Eval Avg: {eval_reward:8.2f}"
            f" | Best Eval: {best_eval_reward:8.2f}"
        )

    print(
        f"Iteration {iteration:03d} | "
        f"Episodes so far: {len(all_episode_rewards):4d} | "
        f"Recent Reward: {recent_reward:8.2f} | "
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
    writer.writerow(["episode", "reward", "steps"])

    for i in range(len(all_episode_rewards)):
        writer.writerow([
            i + 1,
            all_episode_rewards[i],
            all_episode_steps[i]
        ])


# =========================
# نمودار پاداش
# =========================

plt.figure(figsize=(10, 5))
plt.plot(all_episode_rewards, label="Episode reward")

ma_rewards = moving_average(all_episode_rewards, window_size=20)

if len(ma_rewards) > 0:
    plt.plot(
        range(19, 19 + len(ma_rewards)),
        ma_rewards,
        label="Moving average (window=20)"
    )

plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Pendulum-v1 - Tuned PPO Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(REWARD_PLOT_FILE, dpi=300)
plt.close()


print("\n======================================")
print("Tuned PPO training finished.")
print("Best eval reward:", best_eval_reward)
print("Saved best model weights:", MODEL_WEIGHTS_FILE)
print("Saved reward plot:", REWARD_PLOT_FILE)
print("Saved training log:", LOG_FILE)
print("======================================")