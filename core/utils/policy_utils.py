from typing import Callable, Dict, Hashable

import numpy as np

State = Hashable


def make_epsilon_greedy(
    Q: Dict[State, np.ndarray], n_actions: int, epsilon: float
) -> Callable[[State], int]:
    """
    Returns a policy function π(a|s) that is ε-greedy w.r.t. Q for discrete actions.
    Lazily initializes unseen states with zeros.
    """

    def policy_fn(state: State) -> int:
        if state not in Q:
            Q[state] = np.zeros(n_actions, dtype=np.float64)
        if np.random.rand() < epsilon:
            return np.random.randint(n_actions)
        return int(np.argmax(Q[state]))

    return policy_fn


def greedy_action(Q: Dict[State, np.ndarray], state: State) -> int:
    """Greedy action selector; initializes missing states."""
    if state not in Q:
        Q[state] = np.zeros_like(next(iter(Q.values()))) if Q else np.zeros(2, dtype=np.float64)
    return int(np.argmax(Q[state]))


def greedy_policy(Q: Dict[State, np.ndarray], n_actions: int) -> Callable[[State], int]:
    """Deterministic policy wrapper around greedy_action for evaluation."""

    def policy_fn(state: State) -> int:
        if state not in Q:
            Q[state] = np.zeros(n_actions, dtype=np.float64)
        return int(np.argmax(Q[state]))

    return policy_fn


def tabular_softmax_policy(
    Q: Dict[State, np.ndarray], tau: float, n_actions: int
) -> Callable[[State], int]:
    """
    Boltzmann exploration for tabular Q-values.
    """

    def policy_fn(state: State) -> int:
        if state not in Q:
            Q[state] = np.zeros(n_actions, dtype=np.float64)
        logits = Q[state] / max(tau, 1e-8)
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return int(np.random.choice(n_actions, p=probs))

    return policy_fn
