# Off-policy EV MC control with importance sampling (Blackjack-v1)
from collections import defaultdict
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from .policies import make_epsilon_greedy

State = Tuple[int, int, bool]


def offpolicy_every_visit_mc_control_is(
    env_id: str,
    gamma: float = 1.0,
    episodes: int = 500_000,
    behavior_epsilon: float = 0.2,
    weighted: bool = True,
):
    """
    Off-policy Every-Visit MC Control on Blackjack using importance sampling.
      - Target policy: greedy w.r.t. Q (deterministic).
      - Behavior policy: ε-greedy w.r.t. current Q (with given epsilon).
    Supports both ordinary IS and weighted IS.
    """
    env = gym.make(env_id, sab=True)
    nA = env.action_space.n  # 2
    Q: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(nA, dtype=np.float64))

    # For weighted IS: C stores sum of weights per (s,a)
    C_w: Dict[Tuple[State, int], float] = defaultdict(float)
    # For ordinary IS: N counts episodes contributing to (s,a)
    N_o: Dict[Tuple[State, int], int] = defaultdict(int)

    behavior_policy = make_epsilon_greedy(Q, nA, behavior_epsilon)

    greedy_prob = (
        1.0 - behavior_epsilon + behavior_epsilon / nA
    )  # P_b of greedy action under ε-greedy

    for _ in range(episodes):
        # Generate episode from behavior
        episode: List[Tuple[State, int, float]] = []
        s, _ = env.reset()
        done = False
        while not done:
            a = behavior_policy(s)
            ns, r, term, trunc, _ = env.step(a)
            episode.append((s, a, r))
            s = ns
            done = term or trunc

        # Backward pass with IS ratios
        G = 0.0
        W = 1.0
        for t in reversed(range(len(episode))):
            st, at, rt = episode[t]
            G = gamma * G + rt

            # Deterministic target (greedy w.r.t. Q):
            greedy_a = int(np.argmax(Q[st]))
            if at != greedy_a:
                # pi(a_t|s_t)=0 ⇒ rho becomes 0 ⇒ earlier contributions are zero as well
                break

            # Behavior prob for the greedy action under ε-greedy
            b = greedy_prob  # strictly > 0 as long as ε < 1
            W *= 1.0 / b  # since pi(a|s)=1 for greedy action

            if weighted:
                # Weighted IS: Q <- weighted running average with weight sum C_w
                C_w[(st, at)] += W
                Q[st][at] += (W / C_w[(st, at)]) * (G - Q[st][at])
            else:
                # Ordinary IS: average of (W * G) over counts N_o
                N_o[(st, at)] += 1
                Q[st][at] += (1.0 / N_o[(st, at)]) * (W * G - Q[st][at])

    env.close()
    return Q
