from typing import Callable, List, Tuple

import gymnasium as gym

State = Tuple[int, int, bool]
Transition = Tuple[State, int, float]
Episode = List[Transition]


def compute_returns(episode: Episode, gamma: float) -> Episode:
    """
    Given a single episode [(s0,a0,r1), ...], compute G_t for each time step and
    return a list of (state, action, G_t).
    """
    G = 0.0
    out: Episode = []
    for state, action, reward in reversed(episode):
        G = gamma * G + reward
        out.append((state, action, G))
    out.reverse()
    return out


def generate_episode(env, policy_fn: Callable[[State], int]) -> Episode:
    """Run one episode using the provided policy."""
    episode: Episode = []
    state, _ = env.reset()
    done = False
    while not done:
        action = policy_fn(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated
    return episode


def evaluate_policy(env_id: str, policy_fn: Callable[[State], int], episodes: int = 50) -> float:
    """Average return over a number of evaluation episodes."""
    env = gym.make(env_id, sab=True)
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        done, total = False, 0.0
        while not done:
            action = policy_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(sum(returns) / max(len(returns), 1))
