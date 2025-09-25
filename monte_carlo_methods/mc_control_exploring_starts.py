import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict
from .policies import greedy_action

State = Tuple[int, int, bool]

def mc_control_exploring_starts(env_id: str, gamma: float = 1.0, episodes: int = 500000):
    """
    Monte-Carlo control with Exploring Starts (ES).
    Assumes we can start from any (s,a) with non-zero probability.
    For Blackjack, we simulate ES by forcing a random first action.
    """
    env = gym.make(env_id, sab=True)
    nA = env.action_space.n  # 0=stick, 1=hit
    Q: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(nA, dtype=np.float64))
    returns: Dict[Tuple[State, int], List[float]] = defaultdict(list)
    policy: Dict[State, int] = {}

    for _ in range(episodes):
        s, _ = env.reset()
        # Exploring start: choose a random first action
        a0 = np.random.randint(nA)
        episode = []
        done = False
        a = a0
        while not done:
            ns, r, term, trunc, _ = env.step(a)
            episode.append((s, a, r))
            s = ns
            done = term or trunc
            if not done:
                # Greedy thereafter (GLIE-ish)
                a = greedy_action(Q, s)

        # MC updates
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
            policy[st] = int(np.argmax(Q[st]))

    env.close()
    return Q, policy
