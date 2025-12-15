Let's explore the world of Reinforcement Learning through implementation using python as simplified as possible. 

I'm assuming you have basic foundational knowledge of Markov Decision Processes (MDPs) and Dynamic Programming (DP). Most RL algorithms can be viewed as attempts to achieve much the same effect as DP, only with less computation. 

You'll see the implementation of the classical reinforcement learning algorithms from [Reinforcement Learning: An Introduction](https://inst.eecs.berkeley.edu/~cs188/sp20/assets/files/SuttonBartoIPRLBook2ndEd.pdf) on various environments
 - Dynamic Programming (Policy and Value Iteration)
 - Monte Carlo Methods (Prediction and Control)
 - Temporal Difference (SARSA and Q-Learning) 
 - Value Function Approximation (DQN, DDQN)
 - Policy gradient methods (REINFORCE)
 - Actor Critic methods (DDPG, PPO, TRPO, A2C, TD3, SAC, RPO, AMP)
 - Model Based methods (Dyna-Q, PETS)

## Project layout
- `core/algorithms/monte_carlo`: Blackjack Monte Carlo prediction and control.
- `core/algorithms/tabular`: Dynamic programming and temporal-difference methods for grid worlds.
- `core/env`: The grid world environment and configs.
- `core/utils`: Small numeric helpers.
- `examples`: Runnable scripts demonstrating the algorithms.
- `tests`: Placeholder suite ready for real unit tests.
- `results`: Saved figures produced by the examples.

## Setup
- Install dependencies: `pip install -r requirements.txt`
- (Optional) Editable install with dev tools: `pip install -e .[dev]`
- Lint/format/test: `ruff format --check . && ruff check . && pytest`



