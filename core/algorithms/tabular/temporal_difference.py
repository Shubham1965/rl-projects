import random

import matplotlib.pyplot as plt
import numpy as np

from core.env.grid_world import GridWorld


class SARSA:
    def __init__(
        self, grid_world: GridWorld, alpha: float, gamma: float, epsilon: float, num_episodes: int
    ):
        self.env = grid_world
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes

        # Initialize Q table
        self.Q = {}
        for state in self.env.all_states:
            for action in self.env.actions:
                self.Q[(state, action)] = 0.0

    def decay_epsilon(self):
        self.epsilon = self.epsilon * 0.9

    def choose_action(self, state):
        # Epsilon-greedy policy and decay epsilon

        if state != self.env.goal_state:
            if np.random.uniform(0, 1) < self.epsilon:
                return random.choice(self.env.actions)  # Exploration
            else:
                return max(
                    self.env.actions, key=lambda action: self.Q[(state, action)]
                )  # Exploitation

    def update_Q(self, state, action, reward, next_state, next_action):
        if next_state == self.env.goal_state:
            predict = self.Q[(state, action)]
            target = reward
            self.Q[(state, action)] += self.alpha * (target - predict)
        else:
            predict = self.Q[(state, action)]
            target = reward + self.gamma * self.Q[(next_state, next_action)]
            self.Q[(state, action)] += self.alpha * (target - predict)

    def train(
        self,
    ):
        rewards = []
        for _ in range(self.num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            reward_episode = 0
            while not done:
                next_state, reward, done = self.env.step(state, action)
                next_action = self.choose_action(next_state)
                self.update_Q(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
                reward_episode += reward

            rewards.append(reward_episode)

            # Decay epsilon
            # if self.env.is_terminal(state):
            #     self.decay_epsilon()

        # Get the optimal policy
        self.possible_states = self.env.all_states.copy()
        for obstacle in self.env.obstacles:
            self.possible_states.remove(obstacle)

        policy = {
            state: {max(self.env.actions, key=lambda action: self.Q[(state, action)]): 1}
            for state in self.possible_states
            if state != self.env.goal_state
        }
        policy[self.env.goal_state] = {}

        return self.Q, policy, rewards

    def plot_rewards(self, rewards: list, per_episode: int = 1):
        average_rewards = [
            sum(rewards[i : i + per_episode]) / per_episode
            for i in range(0, len(rewards), per_episode)
        ]  # Average every 10 episodes

        plt.figure(figsize=(10, 5))
        plt.plot(average_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Average Reward per {per_episode} Episode")
        plt.savefig("results/sarsa_rewards.png")
        plt.show()

    def plot_q_table(self, q_table: dict):
        # Extract unique states and actions
        states = self.env.all_states
        actions = self.env.actions

        # Create a grid to hold Q-values
        q_values = np.zeros((len(states), len(actions)))

        # Fill the grid with Q-valuess
        for state, action in q_table.keys():
            state_index = states.index(state)
            action_index = actions.index(action)
            q_values[state_index, action_index] = q_table[(state, action)]

        plt.figure(figsize=(10, 7))
        plt.pcolor(q_values, cmap="coolwarm", vmin=q_values.min(), vmax=q_values.max())

        # Highlight the maximum value in each row
        for i in range(len(states)):
            max_index = np.argmax(q_values[i])
            plt.scatter(max_index + 0.5, i + 0.5, marker="o", color="lime", s=100, zorder=10)

        plt.title("Q-table with Max Value Highlighted")
        plt.xlabel("Actions")
        plt.ylabel("States")

        # Set custom tick positions and labels
        plt.xticks(
            np.arange(len(actions)) + 0.5, actions
        )  # Shift ticks to the center of each action bin
        plt.yticks(np.arange(len(states)) + 0.5, [str(state) for state in states])

        plt.colorbar(label="Q-value")
        plt.gca().invert_yaxis()  # Invert y-axis to align with matrix
        plt.grid(True)  # Add gridlines
        plt.savefig("results/sarsa_q_table.png")
        plt.show()


class QLearning:
    def __init__(
        self, grid_world: GridWorld, alpha: float, gamma: float, epsilon: float, num_episodes: int
    ):
        self.env = grid_world
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes

        self.Q = {}
        for state in self.env.all_states:
            for action in self.env.actions:
                self.Q[(state, action)] = 0.0

    def decay_epsilon(self):
        self.epsilon = self.epsilon * 0.9

    def choose_action(self, state):
        # Epsilon-greedy policy and decay epsilon

        if state != self.env.goal_state:
            if np.random.uniform(0, 1) < self.epsilon:
                return random.choice(self.env.actions)  # Exploration
            else:
                return max(
                    self.env.actions, key=lambda action: self.Q[(state, action)]
                )  # Exploitation

    def update_Q(self, state, action, reward, next_state):
        if next_state == self.env.goal_state:
            predict = self.Q[(state, action)]
            target = reward
            self.Q[(state, action)] += self.alpha * (target - predict)
        else:
            predict = self.Q[(state, action)]
            target = reward + self.gamma * max(
                self.Q[(next_state, action)] for action in self.env.actions
            )
            self.Q[(state, action)] += self.alpha * (target - predict)

    def train(self):
        rewards = []

        for episode in range(self.num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            reward_episode = 0
            done = False

            while not done:
                next_state, reward, done = self.env.step(state, action)
                self.update_Q(state, action, reward, next_state)
                state = next_state
                action = self.choose_action(state)
                reward_episode += reward

            # Decay epsilon
            # if self.env.is_terminal(state):
            #     self.decay_epsilon()

            rewards.append(reward_episode)

        # Get the optimal policy
        self.possible_states = self.env.all_states.copy()
        for obstacle in self.env.obstacles:
            self.possible_states.remove(obstacle)

        policy = {
            state: {max(self.env.actions, key=lambda action: self.Q[(state, action)]): 1}
            for state in self.possible_states
            if state != self.env.goal_state
        }
        policy[self.env.goal_state] = {}

        return self.Q, policy, rewards

    def plot_rewards(self, rewards: list, per_episode: int = 1):
        average_rewards = [
            sum(rewards[i : i + per_episode]) / per_episode
            for i in range(0, len(rewards), per_episode)
        ]  # Average every 10 episodes

        plt.figure(figsize=(10, 5))
        plt.plot(average_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Average Reward per {per_episode} Episode")
        plt.savefig("results/q_learning_rewards.png")
        plt.show()

    def plot_q_table(self, q_table: dict):
        # Extract unique states and actions
        states = self.env.all_states
        actions = self.env.actions

        # Create a grid to hold Q-values
        q_values = np.zeros((len(states), len(actions)))

        # Fill the grid with Q-valuess
        for state, action in q_table.keys():
            state_index = states.index(state)
            action_index = actions.index(action)
            q_values[state_index, action_index] = q_table[(state, action)]

        plt.figure(figsize=(10, 7))
        plt.pcolor(q_values, cmap="coolwarm", vmin=q_values.min(), vmax=q_values.max())

        # Highlight the maximum value in each row
        for i in range(len(states)):
            max_index = np.argmax(q_values[i])
            plt.scatter(max_index + 0.5, i + 0.5, marker="o", color="lime", s=100, zorder=10)

        plt.title("Q-table with Max Value Highlighted")
        plt.xlabel("Actions")
        plt.ylabel("States")

        # Set custom tick positions and labels
        plt.xticks(
            np.arange(len(actions)) + 0.5, actions
        )  # Shift ticks to the center of each action bin
        plt.yticks(np.arange(len(states)) + 0.5, [str(state) for state in states])

        plt.colorbar(label="Q-value")
        plt.gca().invert_yaxis()  # Invert y-axis to align with matrix
        plt.grid(True)  # Add gridlines
        plt.savefig("results/q_learning_q_table.png")
        plt.show()
