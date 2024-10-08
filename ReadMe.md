Let's explore the world of Reinforcement Learning through implementation using python as simplified as possible. 

I'm assuming you have basic foundational knowledge of Markov Decision Processes (MDPs) and Dynamic Programming (DP). Most RL algorithms can be viewed as attempts to achieve much the same effect as DP, only with less computation. 

You'll see the implementation of the classical reinforcement learning algorithms from [Reinforcement Learning: An Introduction](https://inst.eecs.berkeley.edu/~cs188/sp20/assets/files/SuttonBartoIPRLBook2ndEd.pdf) on various environments
 - Dynamic Programming (Policy and Value Iteration)
 - Monte Carlo Methods (Prediction and Control)
 - Temporal Difference (SARSA and Q-Learning) 
 - Value Function Approximation (DQN)
 - Policy gradient methods (REINFORCE)
 - Actor Critic methods (DDPG, PPO, TRPO, A3C, TD3, SAC)

## Dynamic Programming
Dynamic programming (DP) is a powerful approach to solving reinforcement learning problems by breaking them down into smaller subproblems and solving them iteratively. One of the key concepts in DP is the notion of policy iteration.

Policy iteration is a general framework for finding an optimal policy in reinforcement learning. It consists of two main steps: policy evaluation and policy improvement.

Policy Evaluation: In this step, the value function of a given policy is iteratively updated until it converges to the true value function. This is typically done using the Bellman equation, which expresses the value of a state as the expected sum of rewards that can be obtained from that state following the current policy.

Policy Improvement: Once the value function has been evaluated, the policy is improved based on the current value function. This is done by selecting actions that lead to states with higher values according to the current value function. The new policy may be deterministic (greedy) or stochastic, depending on the problem and the specific algorithm being used.

These two steps are repeated iteratively until the policy converges to the optimal policy, which maximizes the expected cumulative reward over time.

- ### Policy Iteration: 

    The following code snippets perform policy iteration on a grid world from `policy_iteration_gridworld.py`.

    First, it defines the dimensions of the grid world, the start state, the goal state, and the locations of obstacles. Then, it creates an instance of the GridWorld class with these parameters.
 
        ncols = 4
        start_state = (0, 0)
        goal_state = (3, 3)
        obstacles = [(1, 1),(0, 2)]

        grid_world = GridWorld(nrows, ncols, start_state, goal_state)
   


    Next, it adds obstacles, rewards, and dynamics to the grid world.

        grid_world.add_obstacles(obstacles)
        grid_world.add_rewards(-1.0, 100.0)
        grid_world.dynamics() 

    After that, it generates a random policy for the grid world.
        
        policy = grid_world.random_policy()

    Then, it uses the PolicyIteration function to iteratively update the value function and policy until convergence. The value function V and the optimal policy policy are returned.

        V, policy = PolicyIteration(grid_world, theta=0.0001)

    Finally, it creates an instance of the GridWorldVisualization class and plots the grid world with arrows indicating the optimal policy.

        visualization = GridWorldVisualization(grid_world)
        visualization.plot_grid_with_arrows(grid_world, policy)

    ![Results of Policy Iteration ](results/policy_iteration_gridworld.png)


One drawback to policy iteration is that each of its iterations involves policy evaluation, which may itself be a protracted iterative computation requiring multiple sweeps through the state set.  Must we wait for exact convergence, or can we stop short of that? 

In fact, the policy evaluation step of policy iteration can be truncated in several ways without losing the convergence guarantees of policy iteration. One important special case is when policy evaluation is stopped after just one sweep (one update of each state). This algorithm is called value iteration.


- ### Value Iteration
    The file `value_iteration_gridworld.py` has the exact similar structure as above which results in the same policy but since it performs only on step of policy evaluation instead of converging, it is much faster than `policy_iteration_gridworld.py`. In order to check the performance of both try running the hard grid problem for both and time it. Following is the result for the hard grid problem using value iteration.

    ![Results of Policy Iteration ](results/value_iteration_gridworld.png)

## Monte Carlo methods (tbd)

## Temporal Difference

Temporal Difference (TD) learning is a type of reinforcement learning algorithm that combines aspects of both dynamic programming and Monte Carlo methods.

DP methods involve solving problems by breaking them down into smaller subproblems, typically using a bottom-up approach. In the context of reinforcement learning, DP algorithms (like the Bellman equation) are used to calculate the value function by iteratively updating estimates based on the expected future rewards.

Monte Carlo (MC) methods, on the other hand, involve learning from experience by averaging sampled returns obtained from complete episodes. MC methods do not require a model of the environment and are often used in situations where the environment is stochastic or episodic.

Temporal Difference learning bridges the gap between DP and MC methods by updating value estimates based on a combination of bootstrapping (using current estimates to update future estimates) and sampling (using actual experiences). Instead of waiting until the end of an episode to update value estimates, TD methods update them after each time step based on the observed reward and the estimate of the value of the next state.

Here, I'm showing you the results on the classic cliff world example using SARSA (on-policy control TD(0)) and Q-learning (off-policy control TD(0)). Fell free to run the scripts named ``sarsa_cliffworld.py`` and ``qLearning_cliffworld.py ``.


- SARSA Rewards: 
![](results/sarsa_rewards.png) 

- SARSA policy and optimal trajectory: 
![](results/sarsa_cliffWorld.png) 

- Q-learning Rewards: 
![](results/q_learning_rewards.png) 

- Q-learning policy and optimal trajectory:
![](results/qlearning_cliffWorld.png) 


Q-learning learns values for the optimal policy, that which travels right along the edge of the cliff. Unfortunately, this re sults in its occasionally falling off
the cliff because of the $\epsilon$-greedy action selection. Sarsa, on the other hand, takes the action selection into account and learns the longer but safer path through the upper part per epsiode of the grid.
## Value Function Approximation (tbd)

## Policy Gradient methods (tbd)

## Actor-Critic methods (tbd)
