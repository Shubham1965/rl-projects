import numpy as np

from core.envs.grid_world.grid_world import GridWorld


def PolicyEvaluation(env: GridWorld, policy: dict, theta: float):
    """
    Perform policy evaluation using iterative policy evaluation.

    Parameters:
        env (GridWorld): The environment for which the policy evaluation is performed.
        policy (dict): The policy to be evaluated.
        theta (float): A small positive value determining the accuracy of estimation.

    Returns:
        V (dict): A dictionary representing the value function for each state.
        policy (dict): The evaluated policy.
    """
    V = {state: 0 for state in env.all_states}  # Initialize the value function

    while True:
        delta = 0
        for state in env.all_states:
            if not env.is_terminal(state):
                v = V[state]
                new_v = 0
                for action, action_prob in policy[state].items():
                    next_state, reward, prob = env.P[state][action]
                    new_v += action_prob * prob * (reward + env.gamma * V[next_state])
                V[state] = new_v
                delta = max(delta, abs(new_v - v))

        if delta < theta:
            break

    return V, policy


def PolicyImprovement(env: GridWorld, policy: dict, V: dict):
    """
    Improves the given policy in the GridWorld environment based on the current value function.

    Args:
        env (GridWorld): The GridWorld environment.
        policy (dict): The policy to be improved.
        V (dict): The current value function.

    Returns:
        new_policy (dict): The improved policy.
        policy_stable (bool): True if the policy is stable, False otherwise.
    """
    new_policy = {state: {} for state in env.all_states}
    policy_stable = True

    for state in env.all_states:
        if not env.is_terminal(state):
            old_action = max(policy[state], key=policy[state].get)
            best_action = None
            best_value = float("-inf")
            for action in env.actions:
                next_state, reward, _ = env.P[state][action]
                value = reward + env.gamma * V[next_state]
                if value > best_value:
                    best_value = value
                    best_action = action
            new_policy[state] = {
                action: 1 if action == best_action else 0 for action in env.actions
            }
            if old_action != best_action:
                policy_stable = False

    return new_policy, policy_stable


def PolicyIteration(env: GridWorld, theta: float):
    """
    Perform policy iteration to find the optimal policy for a given GridWorld environment.

    Parameters:
        env (GridWorld): The environment for which policy iteration is performed.
        theta (float): A small positive value determining the accuracy of estimation.

    Returns:
        V (dict): A dictionary representing the value function for each state.
        policy (dict): The optimal policy obtained from policy iteration.
    """
    policy = {
        state: {action: 1 / len(env.actions) for action in env.actions} for state in env.all_states
    }
    while True:
        V, policy = PolicyEvaluation(env, policy, theta)
        new_policy, policy_stable = PolicyImprovement(env, policy, V)
        if policy_stable:
            break
        policy = new_policy
    return V, policy


def ValueIteration(env: GridWorld, theta: float) -> np.ndarray:
    """
    Perform value iteration to find the optimal value function and policy for a given GridWorld environment.

    Parameters:
        env (GridWorld): The environment for which value iteration is performed.
        theta (float): A small positive value determining the accuracy of estimation.

    Returns:
        V (np.ndarray): A numpy array representing the value function for each state.
        policy (dict): The optimal policy obtained from value iteration.
    """
    V = {state: 0 for state in env.all_states}
    while True:
        delta = 0
        for state in env.all_states:
            if not env.is_terminal(state):
                v = V[state]
                vv = []
                for action in env.actions:
                    next_state, reward, _ = env.P[state][action]
                    vv.append(reward + env.gamma * V[next_state])
                new_v = max(vv)
                delta = max(delta, abs(new_v - v))
                V[state] = new_v
        if delta < theta:
            break

    # Get the optimal policy
    policy = {}
    for state in env.all_states:
        if not env.is_terminal(state):
            best_action = max(
                env.actions,
                key=lambda action: env.P[state][action][1] + env.gamma * V[env.P[state][action][0]],
            )
            policy[state] = {action: 0 if action != best_action else 1 for action in env.actions}
        else:
            policy[state] = {}  # Terminal states have no policy

    return V, policy
