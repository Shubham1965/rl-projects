import numpy as np
import matplotlib.pyplot as plt	
import matplotlib.patches as mpatches

class GridWorld:
    def __init__(self, nrows : int, ncols : int, start_state : tuple, goal_state : tuple):
        """
        Initializes a new instance of the GridWorld class.

        Args:
            nrows (int): The number of rows in the grid.
            ncols (int): The number of columns in the grid.
            start_state (tuple): The starting state of the grid world.
            goal_state (tuple): The goal state of the grid world.

        Returns:
            None
        """
        self.num_rows = nrows
        self.num_cols = ncols
        self.start_state = start_state
        self.goal_state = goal_state
        self.obstacles = None
        self.reward_step = None
        self.reward_goal = None
        self.gamma = 1 #default discount factor is 1
        self.actions = ["up", "down", "left", "right"]
        self.num_actions = 4
        self.all_states = [(i, j) for i in range(self.num_rows) for j in range(self.num_cols)]
        self.num_states = self.num_cols * self.num_rows 

    def add_obstacles(self, obstacles : list):
        """
        Adds obstacles to the grid world.

        Parameters:
            obstacles (list): A list of obstacle coordinates in the grid.

        Returns:
            None
        """
        self.obstacles = obstacles

        for obstacle in obstacles:
            self.all_states.remove(obstacle)
        
        self.num_states = len(self.all_states)

    def is_valid_state(self, state: tuple) -> bool:
        """
        Check if the given state is valid in the grid world.

        Args:
            state (tuple): The state to check in the form of (row, col).

        Returns:
            bool: True if the state is valid, False otherwise.

        Raises:
            None

        Notes:
            - The state is considered valid if it is within the grid boundaries and not an obstacle.
            - The grid boundaries are defined by the attributes `num_rows` and `num_cols`.
            - The obstacles are defined by the attribute `obstacles`.
        """
        row, col = state
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols and state not in self.obstacles
    
    def is_terminal(self, state: tuple) -> bool:
        """
        Check if the provided state is the goal state.

        Args:
            state (tuple): The state to check for being the goal state.

        Returns:
            bool: True if the state is the goal state, False otherwise.
        """
        return state == self.goal_state

    def add_rewards(self, reward_step: float, reward_goal: float):
        """
        Set the reward values for the step and goal states.

        Args:
            reward_step (float): The reward value for each step.
            reward_goal (float): The reward value for reaching the goal state.

        Returns:
            None
        """
        self.reward_step = reward_step
        self.reward_goal = reward_goal

    def add_discout_factor(self, gamma: float):
        """
        Set the discount factor for the GridWorld.

        Args:
            gamma (float): The discount factor to be set.

        Returns:
            None
        """
        self.gamma = gamma

    def get_next_state(self, state: tuple, action: str) -> tuple:
        """
        Calculate the next state based on the current state and action.
        
        Args:
            state (tuple): The current state of the grid world in the form of (row, col).
            action (str): The action to take in the grid world. Can be "up", "down", "left", or "right".
        
        Returns:
            tuple: The next state of the grid world
        """
        state = list(state)
        next_state = state.copy()

        if action == "up":
            next_state[0] -= 1
        elif action == "down":
            next_state[0] += 1
        elif action == "left":
            next_state[1] -= 1
        elif action == "right":
            next_state[1] += 1
        
        next_state = tuple(next_state)
        state = tuple(state)
        if self.is_valid_state(next_state):
            return next_state
        else:
            return state # stay in the same state

    def get_reward(self, state: tuple) -> float:
        """
        Returns the reward for the given state.

        Parameters:
            state (tuple): The current state of the grid world in the form of (row, col).

        Returns:
            float: The reward value for the given state. If the state is a goal state, the reward is the value of self.reward_goal. If the state is not a goal state, the reward is the value of self.reward_step.

        Raises:
            None
        """

        if self.is_valid_state(state):
            return self.reward_goal if self.is_terminal(state) else self.reward_step

    
    def dynamics(self):
        """
        Initialize the environment dynamics for the GridWorld.

        The environment dynamics are defined by the `P` dictionary. `P` is a nested dictionary that maps each state in `all_states` to a dictionary 
        that maps each action in `actions` to a list containing two elements: the next state after taking that action in that state, 
        and the reward obtained after taking that action in that state.

        Parameters:
            None

        Returns:
            None
        """

        # Initialize the environment's dyanmics 
        # self.V = {state: 0 for state in self.all_states}
        # self.Q = {state: {action: 0 for action in self.actions} for state in self.all_states}
        self.P = {state: {action: [self.get_next_state(state, action), self.get_reward(self.get_next_state(state, action)), 1] for action in self.actions} for state in self.all_states}

    def random_policy(self):
        """
        Generate a random policy for the grid world.

        Returns:
            dict: A dictionary representing the random policy. 
            The keys are the states in the grid world, and the values are dictionaries that map each action to a probability of taking that action. 
            The probabilities are all equal to 0.25.
        """
        
        return {state: {action: 0.25 for action in self.actions} for state in self.all_states}

    def visualize_gridWorld(self):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if [i, j] in self.start_state.tolist():
                    print("S", end=" ")
                elif [i, j] in self.goal_state.tolist():
                    print("G", end=" ")
                elif [i, j] in self.obstacles.tolist():
                    print("#", end=" ")
                else:
                    print(".", end=" ")
            print()


class GridWorldVisualization:
    def __init__(self, grid_world: GridWorld):
        """
        Initializes a new instance of the GridWorldVisualization class.

        Args:
            grid_world (GridWorld): An instance of the GridWorld class.

        Returns:
            None
        """
        self.grid_world = grid_world
        self.grid = np.zeros((grid_world.num_rows, grid_world.num_cols))

    def plot_grid_with_arrows(self, grid_world, grid_dict):
        """
        Plots a grid world with arrows indicating possible actions.

        Parameters:
            grid_world (GridWorld): The grid world to plot.
            grid_dict (dict): A dictionary mapping coordinates to a dictionary of actions and their values.

        Returns:
            None
        """
        def plot_arrow(coord, direction):
            dx, dy = {'up': (0, -0.3), 'down': (0, 0.3), 'left': (-0.3, 0), 'right': (0.3, 0)}[direction]
            plt.arrow(coord[1], coord[0], dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')

        grid = np.zeros((grid_world.num_rows, grid_world.num_cols))

        for coord, actions in grid_dict.items():
            if actions:
                for action, val in actions.items():
                    if val == 1 or val == 0.25:
                        grid[coord[0], coord[1]] = 1
                        plot_arrow(coord, action)
            else:
                grid[coord[0], coord[1]] = 1  # Terminal state

        plt.imshow(grid, cmap='gray', origin='upper')

        # Highlighting start state and goal state
        plt.scatter(grid_world.start_state[1], grid_world.start_state[0], color='green', marker='o', s=100, label='Start')
        plt.scatter(grid_world.goal_state[1], grid_world.goal_state[0], color='red', marker='x', s=100, label='Goal')

        plt.yticks(np.arange(-0.5, grid_world.num_rows+0.5, step=1))
        plt.xticks(np.arange(-0.5, grid_world.num_cols+0.5, step=1))
        plt.grid()

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
        plt.show()






    