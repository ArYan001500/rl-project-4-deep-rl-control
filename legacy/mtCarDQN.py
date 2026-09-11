# In the Name of Allah
# Mountain Car problem with DQN and Tensorflow
# Programmer : Maziar Palhang
# Last  Edit : Thu. 1405/3/21

import gymnasium as gym
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import random

class DQNmodel(tf.keras.Model):
    def __init__(self,state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = int(action_dim)

        self.d1 = tf.keras.layers.Dense(64,activation='relu')
        self.d2 = tf.keras.layers.Dense(16,activation='relu')
        self.out = tf.keras.layers.Dense(self.action_dim, activation='linear')

    def call(self, input_data):
        x = tf.convert_to_tensor(input_data)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)      #outputs Q Value of each action
        return x

    def get_config(self):
        # دریافت کانفیگ پایه
        config = super(DQNmodel, self).get_config()
        # اضافه کردن پارامترهای مدل به دیکشنری کانفیگ
        config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        })
        return config

class DQNTargetReplayAgent():
    def __init__(self,model,targetModel):
        self.model = model
        self.targetModel = targetModel
        self.targetUpdate()              
        self.epsilon = 1
        self.EPSILON_END = 0.01          # Minimum exploration rate
        self.EPSILON_DECAY = 0.995       # Decay rate for exploration

        self.gamma = 0.99
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.train_start = 1000
        self.target_update_freq = 1000
        self.state_dim = model.state_dim
        self.action_dim = model.action_dim
        return

    def act(self, state, action_dim):
        if np.random.rand() < self.epsilon:
            return np.random.randint(action_dim)

        q_values = self.model(np.array([state]))
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def train(self):
        if len(self.memory) < self.train_start:
            return

        mini_batch = random.sample(self.memory, self.batch_size)

        update_input = np.zeros((self.batch_size, self.state_dim))
        update_target = np.zeros((self.batch_size, self.action_dim))

        for i in range(self.batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            #target = self.model.predict(state)[0]

            q_current = self.model(np.array([state]), training=False).numpy()[0]  # shape (action_dim,)
            q_next = self.targetModel(np.array([next_state]), training=False).numpy()[0] # shape (action_dim,)

            # Create target vector (copy so we can edit safely)
            target = q_current.copy()

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.max(q_next)

            update_input[i] = state
            update_target[i] = target

        self.model.fit(update_input, update_target,
                       batch_size=self.batch_size, epochs=1, verbose=0)

            #self.model.fit(x, y, epochs=1, batch_size=self.batch_size, verbose=0)
        if self.epsilon > self.EPSILON_END:
            self.epsilon *= self.EPSILON_DECAY

        return
        
    def targetUpdate(self):
        # Simply copy the weights of the model to target_model
        self.targetModel.set_weights(self.model.get_weights())
        return

    def load(self, file_name="dqn.keras"):
        self.model.load_weights(file_name)
        self.targetModel.load_weights(file_name)
        print(f"Loaded weights from {file_name}")
    
    def save(self,fileName="dqn.keras"):
        self.model.save(fileName)
        return

env = gym.make('MountainCar-v0', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = DQNmodel(state_dim,action_dim)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error')
targetModel = DQNmodel(state_dim,action_dim)
targetModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error')
dummy_input = np.zeros((1, state_dim)) 
model(dummy_input)  # This triggers the weight creation
targetModel(dummy_input) # This triggers the weight creation 

agent = DQNTargetReplayAgent(model,targetModel)
agent.targetUpdate()
trials = 200
for trial in range(trials):
    done = False
    state, info = env.reset()
    total_reward = 0
    while not done:
        env.render()
        action = agent.act(state,action_dim)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.remember(state, action, reward, next_state, terminated)
        agent.train()
        state = next_state
        total_reward += reward
        if terminated or truncated:
            done = True
            agent.targetUpdate()
        
    print("total reward after {} episodes is {}".format(trial, total_reward))

agent.save("dqn_mountaincar.keras")
env.close()
