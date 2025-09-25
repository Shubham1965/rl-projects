import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List
from .episode import compute_returns

State = Tuple[int, int, bool]

class FirstVisitMCPrediction:
    """
    First-visit Monte-Carlo state-value prediction for episodic tasks.
    Tracks V(s) and returns its running mean.
    """
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        self.returns: Dict[State, List[float]] = defaultdict(list)
        self.V: Dict[State, float] = defaultdict(float)

    def update_from_episode(self, episode: List[Tuple[State, int, float]]):
        seen = set()
        for t, (s, a, Gt) in enumerate(compute_returns(episode, self.gamma)):
            if s in seen:
                continue
            seen.add(s)
            self.returns[s].append(Gt)
            self.V[s] = float(np.mean(self.returns[s]))
