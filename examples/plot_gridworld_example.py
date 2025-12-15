from core.envs.grid_world.grid_world import GridWorld, GridWorldVisualization

if __name__ == "__main__":
    nrows = 4
    ncols = 4
    start_state = (0, 0)
    goal_state = (3, 3)
    obstacles = [(1, 1), (0, 2)]
    grid_world = GridWorld(nrows, ncols, start_state, goal_state)
    grid_world.add_obstacles(obstacles)
    grid_world.add_rewards(-1.0, 100.0)
    grid_world.dynamics()

    visualization = GridWorldVisualization(grid_world)
    visualization.plot_grid_with_arrows(grid_world, grid_world.random_policy())
