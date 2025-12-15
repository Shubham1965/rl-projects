from core.algorithms.tabular.temporal_difference import QLearning
from core.envs.grid_world.grid_world import GridWorld, GridWorldVisualization

nrows = 4
ncols = 8
start_state = (3, 0)
goal_state = (3, 7)
obstacles = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)]

# hard grid
# nrows = 20
# ncols = 20
# start_state = (0, 0)
# goal_state = (19, 19)
# obstacles = [(5, 5), (5, 6), (5, 7), (5, 8), (5, 9),
#              (6, 9), (7, 9), (8, 9), (9, 9), (10, 9),
#              (10, 8), (10, 7), (10, 6), (10, 5), (10, 4),
#              (11, 4), (12, 4), (13, 4), (14, 4), (15, 4),
#              (15, 5), (15, 6), (15, 7), (15, 8), (15, 9),
#              (16, 9), (17, 9), (18, 9), (19, 9), (19, 10)]

grid_world = GridWorld(nrows, ncols, start_state, goal_state)
grid_world.add_obstacles(obstacles)
grid_world.add_rewards(-1.0, -1.0, -100.0)

qlearning_agent = QLearning(grid_world, alpha=0.1, gamma=0.95, epsilon=0.1, num_episodes=1000)
value, policy, rewards = qlearning_agent.train()
qlearning_agent.plot_rewards(rewards, per_episode=10)

visualization = GridWorldVisualization(grid_world)
visualization.plot_grid_with_arrows(grid_world, policy, fig_name="qlearning_cliffWorld")
