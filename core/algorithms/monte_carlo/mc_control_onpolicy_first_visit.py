from collections import defaultdict
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

from .policies import make_epsilon_greedy

State = Tuple[int, int, bool]


def onpolicy_first_visit_mc_control(
    env_id: str, gamma: float = 1.0, episodes: int = 500000, epsilon: float = 0.1
):
    """
    On-policy First-Visit MC Control with ε-greedy policy improvement (GLIE-style).
    """
    env = gym.make(env_id, sab=True)
    nA = env.action_space.n
    Q: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(nA, dtype=np.float64))
    returns = defaultdict(list)
    policy = make_epsilon_greedy(Q, nA, epsilon)

    for _ in range(episodes):
        # Generate episode following current ε-greedy policy
        s, _ = env.reset()
        episode = []
        done = False
        while not done:
            a = policy(s)
            ns, r, term, trunc, _ = env.step(a)
            episode.append((s, a, r))
            s = ns
            done = term or trunc

        # First-visit MC update on Q
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            st, at, rt = episode[t]
            G = gamma * G + rt
            if (st, at) in visited:
                continue
            visited.add((st, at))
            returns[(st, at)].append(G)
            Q[st][at] = np.mean(returns[(st, at)])
        # Policy implicitly improves via ε-greedy on updated Q

    env.close()
    return Q
