from collections import defaultdict
from typing import Callable, List, Tuple

import gymnasium as gym
import numpy as np

from .episode import compute_returns

State = Tuple[int, int, bool]


def generate_episode(env, policy_fn: Callable[[State], int]) -> List[Tuple[State, int, float]]:
    episode = []
    s, _ = env.reset()
    done = False
    while not done:
        a = policy_fn(s)
        ns, r, term, trunc, _ = env.step(a)
        episode.append((s, a, r))
        s = ns
        done = term or trunc
    return episode


def mc_state_value_prediction(
    env_id: str, policy_fn, gamma: float, episodes: int, first_visit: bool = True
):
    env = gym.make(env_id, sab=True)  # sab=True = Sutton & Barto’s rules
    returns = defaultdict(list)
    V = defaultdict(float)

    for _ in range(episodes):
        ep = generate_episode(env, policy_fn)
        seen = set()
        for t, (s, a, Gt) in enumerate(compute_returns(ep, gamma)):
            if first_visit and s in seen:
                continue
            seen.add(s)
            returns[s].append(Gt)
            V[s] = float(np.mean(returns[s]))
    env.close()
    return V
