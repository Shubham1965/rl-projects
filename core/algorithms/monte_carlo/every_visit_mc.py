from collections import defaultdict
from typing import Tuple

import numpy as np

from .episode import compute_returns

State = Tuple[int, int, bool]


class EveryVisitMCPrediction:
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        self.returns = defaultdict(list)
        self.V = defaultdict(float)

    def update_from_episode(self, episode):
        # For every timestep t, compute G_t and add to that state's list
        for s, a, Gt in compute_returns(episode, self.gamma):
            self.returns[s].append(Gt)
            self.V[s] = float(np.mean(self.returns[s]))
