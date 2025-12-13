from typing import Any, List, Tuple

Transition = Tuple[Any, int, float]  # (state, action, reward)


def compute_returns(episode: List[Transition], gamma: float):
    """
    Given a single episode [(s0,a0,r1), (s1,a1,r2), ...], compute
    return G_t for each time step and return list of (state, action, G_t).
    """
    G = 0.0
    out = []
    for s, a, r in reversed(episode):
        G = gamma * G + r
        out.append((s, a, G))
    out.reverse()
    return out
