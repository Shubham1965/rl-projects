from core.algorithms.tabular.dynamic_programming import ValueIteration
from core.env.grid_world import GridWorld, GridWorldVisualization

# nrows = 4
# ncols = 4
# start_state = (0, 0)
# goal_state = (3, 3)
# obstacles = [(1, 1),(0, 2)]

# hard grid
nrows = 20
ncols = 20
start_state = (0, 0)
goal_state = (19, 19)
obstacles = [
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 9),
    (6, 9),
    (7, 9),
    (8, 9),
    (9, 9),
    (10, 9),
    (10, 8),
    (10, 7),
    (10, 6),
    (10, 5),
    (10, 4),
    (11, 4),
    (12, 4),
    (13, 4),
    (14, 4),
    (15, 4),
    (15, 5),
    (15, 6),
    (15, 7),
    (15, 8),
    (15, 9),
    (16, 9),
    (17, 9),
    (18, 9),
    (19, 9),
    (19, 10),
]

grid_world = GridWorld(nrows, ncols, start_state, goal_state)
grid_world.add_obstacles(obstacles)
grid_world.add_rewards(-1.0, 100.0)
grid_world.dynamics()
policy = grid_world.random_policy()

# Policy Iteration:
V, policy = ValueIteration(grid_world, theta=0.0001)

# Visualization results:
visualization = GridWorldVisualization(grid_world)
visualization.plot_grid_with_arrows(grid_world, policy, fig_name="value_iteration_gridworld")
