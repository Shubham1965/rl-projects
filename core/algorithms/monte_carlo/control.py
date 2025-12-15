from collections import defaultdict
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from .common import Episode, State, generate_episode
from .policies import greedy_action, make_epsilon_greedy

ActionReturns = Dict[Tuple[State, int], List[float]]


def _first_visit_q_update(
    episode: Episode,
    gamma: float,
    Q: Dict[State, np.ndarray],
    returns: ActionReturns,
    policy: Dict[State, int] | None = None,
):
    """Applies first-visit MC updates to Q (and optional greedy policy)."""
    G = 0.0
    visited = set()
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G = gamma * G + reward
        if (state, action) in visited:
            continue
        visited.add((state, action))
        returns[(state, action)].append(G)
        Q[state][action] = np.mean(returns[(state, action)])
        if policy is not None:
            policy[state] = int(np.argmax(Q[state]))


def _exploring_start_episode(env, Q: Dict[State, np.ndarray]) -> Episode:
    """Generate an episode with a random first action then greedy thereafter."""
    nA = env.action_space.n
    state, _ = env.reset()
    action = np.random.randint(nA)
    episode: Episode = []
    done = False
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated
        if not done:
            action = greedy_action(Q, state)
    return episode


def mc_control_exploring_starts(
    env_id: str, gamma: float = 1.0, episodes: int = 500_000
):
    """
    Monte-Carlo control with Exploring Starts (ES).
    Assumes we can start from any (s,a) with non-zero probability.
    For Blackjack, we simulate ES by forcing a random first action.
    """
    env = gym.make(env_id, sab=True)
    nA = env.action_space.n
    Q: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(nA, dtype=np.float64))
    returns: ActionReturns = defaultdict(list)
    policy: Dict[State, int] = {}

    for _ in range(episodes):
        episode = _exploring_start_episode(env, Q)
        _first_visit_q_update(episode, gamma, Q, returns, policy)

    env.close()
    return Q, policy


def onpolicy_first_visit_mc_control(
    env_id: str, gamma: float = 1.0, episodes: int = 500_000, epsilon: float = 0.1
):
    """On-policy First-Visit MC Control with ε-greedy policy improvement."""
    env = gym.make(env_id, sab=True)
    nA = env.action_space.n
    Q: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(nA, dtype=np.float64))
    returns: ActionReturns = defaultdict(list)
    policy_fn = make_epsilon_greedy(Q, nA, epsilon)

    for _ in range(episodes):
        episode = generate_episode(env, policy_fn)
        _first_visit_q_update(episode, gamma, Q, returns)
        # Policy improves implicitly because policy_fn closes over Q

    env.close()
    return Q


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
    nA = env.action_space.n
    Q: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(nA, dtype=np.float64))

    # For weighted IS: C stores sum of weights per (s,a)
    C_w: Dict[Tuple[State, int], float] = defaultdict(float)
    # For ordinary IS: N counts episodes contributing to (s,a)
    N_o: Dict[Tuple[State, int], int] = defaultdict(int)

    behavior_policy = make_epsilon_greedy(Q, nA, behavior_epsilon)
    greedy_prob = 1.0 - behavior_epsilon + behavior_epsilon / nA  # P_b of greedy action

    for _ in range(episodes):
        episode = generate_episode(env, behavior_policy)

        G = 0.0
        W = 1.0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward

            greedy_action_idx = int(np.argmax(Q[state]))
            if action != greedy_action_idx:
                # pi(a_t|s_t)=0 ⇒ rho becomes 0 ⇒ earlier contributions are zero as well
                break

            # Behavior prob for the greedy action under ε-greedy
            W *= 1.0 / greedy_prob  # since pi(a|s)=1 for greedy action

            if weighted:
                C_w[(state, action)] += W
                Q[state][action] += (W / C_w[(state, action)]) * (G - Q[state][action])
            else:
                N_o[(state, action)] += 1
                Q[state][action] += (1.0 / N_o[(state, action)]) * (W * G - Q[state][action])

    env.close()
    return Q


__all__ = [
    "mc_control_exploring_starts",
    "offpolicy_every_visit_mc_control_is",
    "onpolicy_first_visit_mc_control",
]
