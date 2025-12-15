from collections import defaultdict
from typing import Callable, Dict

import gymnasium as gym
import numpy as np

from .common import Episode, State, compute_returns, generate_episode

ValueFunction = Dict[State, float]


class _BaseMCPrediction:
    """Shared logic for Monte-Carlo state-value prediction."""

    def __init__(self, gamma: float = 1.0, first_visit: bool = True):
        self.gamma = gamma
        self.first_visit = first_visit
        self.returns = defaultdict(list)
        self.V: ValueFunction = defaultdict(float)

    def update_from_episode(self, episode: Episode):
        seen = set()
        for state, _action, Gt in compute_returns(episode, self.gamma):
            if self.first_visit and state in seen:
                continue
            seen.add(state)
            self.returns[state].append(Gt)
            self.V[state] = float(np.mean(self.returns[state]))


class FirstVisitMCPrediction(_BaseMCPrediction):
    """
    First-visit Monte-Carlo state-value prediction for episodic tasks.
    Tracks V(s) and returns its running mean.
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__(gamma=gamma, first_visit=True)


class EveryVisitMCPrediction(_BaseMCPrediction):
    """Every-visit Monte-Carlo state-value prediction for episodic tasks."""

    def __init__(self, gamma: float = 1.0):
        super().__init__(gamma=gamma, first_visit=False)


def mc_state_value_prediction(
    env_id: str,
    policy_fn: Callable[[State], int],
    gamma: float,
    episodes: int,
    first_visit: bool = True,
) -> ValueFunction:
    """
    Generic MC prediction driver that dispatches to first- or every-visit updates.
    """
    env = gym.make(env_id, sab=True)  # sab=True = Sutton & Barto’s rules
    predictor: _BaseMCPrediction = (
        FirstVisitMCPrediction(gamma) if first_visit else EveryVisitMCPrediction(gamma)
    )

    for _ in range(episodes):
        episode = generate_episode(env, policy_fn)
        predictor.update_from_episode(episode)

    env.close()
    return predictor.V


__all__ = [
    "EveryVisitMCPrediction",
    "FirstVisitMCPrediction",
    "mc_state_value_prediction",
]
