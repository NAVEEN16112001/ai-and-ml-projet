import numpy as np
import matplotlib.pyplot as plt
import gym
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

# Q-learning implementation
class QLearningAgent:
    def __init__(self, action_size):
        self.q_table = np.zeros((1, action_size))  # Simplified for example
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_delta

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# SARSA implementation
class SARSAAgent:
    def __init__(self, action_size):
        self.q_table = np.zeros((1, action_size))  # Simplified for example
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state, next_action):
        td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_delta

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# DQN implementation
class DQNAgent:
    def __init__(self, action_size):
        self.memory = deque(maxlen=2000)
        self.action_size = action_size
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=4, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])  # Explore
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.discount_factor * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(agent, env, episodes):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        state = np.reshape(state, [1, 4])

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            total_reward += reward

            if isinstance(agent, DQNAgent):
                agent.remember(state, action, reward, next_state, done)
                agent.replay(32)
            else:
                agent.update(state[0], action, reward, next_state[0])
                agent.decay_epsilon()

            state = next_state

        rewards.append(total_reward)
    return rewards

# Initialize environment
env = gym.make('CartPole-v1')

# Train each agent
episodes = 100
q_rewards = train_agent(QLearningAgent(env.action_space.n), env, episodes)
sarsa_rewards = train_agent(SARSAAgent(env.action_space.n), env, episodes)
dqn_agent = DQNAgent(env.action_space.n)
dqn_rewards = train_agent(dqn_agent, env, episodes)

# Plotting the rewards
plt.figure(figsize=(10, 5))
plt.plot(q_rewards, label='Q-Learning')
plt.plot(sarsa_rewards, label='SARSA')
plt.plot(dqn_rewards, label='DQN')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Performance Comparison of Q-Learning, SARSA, and DQN on CartPole')
plt.legend()
plt.show()
