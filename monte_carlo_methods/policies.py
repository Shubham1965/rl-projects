import numpy as np
from collections import defaultdict
from typing import Callable, Dict, Tuple

State = Tuple[int, int, bool]  # (player_sum, dealer_card, usable_ace)

def make_epsilon_greedy(Q: Dict[State, np.ndarray], nA: int, epsilon: float
                        ) -> Callable[[State], int]:
    """
    Returns a policy function π(a|s) that is ε-greedy w.r.t. Q.
    """
    def policy_fn(state: State) -> int:
        if state not in Q:
            Q[state] = np.zeros(nA, dtype=np.float64)
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        return int(np.argmax(Q[state]))
    return policy_fn

def greedy_action(Q: Dict[State, np.ndarray], state: State) -> int:
    if state not in Q:
        Q[state] = np.zeros(2, dtype=np.float64)
    return int(np.argmax(Q[state]))

def tabular_softmax_policy(Q: Dict[State, np.ndarray], tau: float, nA: int
                           ) -> Callable[[State], int]:
    """
    Optional: Boltzmann exploration.
    """
    def policy_fn(state: State) -> int:
        if state not in Q:
            Q[state] = np.zeros(nA, dtype=np.float64)
        logits = Q[state] / max(tau, 1e-8)
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return int(np.random.choice(nA, p=probs))
    return policy_fn
