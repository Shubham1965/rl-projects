# Learning curves (average return vs. episodes) for Blackjack-v1
# Requires: gymnasium, numpy, matplotlib

from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

State = Tuple[int, int, bool]


# ---------- utilities ----------
def eval_policy(env_id: str, policy_fn: Callable[[State], int], episodes: int = 50) -> float:
    env = gym.make(env_id, sab=True)
    rets = []
    for _ in range(episodes):
        s, _ = env.reset()
        done, G = False, 0.0
        while not done:
            a = policy_fn(s)
            s, r, term, trunc, _ = env.step(a)
            G += r
            done = term or trunc
        rets.append(G)
    env.close()
    return float(np.mean(rets))


def greedy_from_Q(Q: Dict[State, np.ndarray]) -> Callable[[State], int]:
    def pi(s: State) -> int:
        if s not in Q:
            Q[s] = np.zeros(2, dtype=np.float64)
        return int(np.argmax(Q[s]))

    return pi


def eps_greedy_from_Q(Q: Dict[State, np.ndarray], eps: float) -> Callable[[State], int]:
    nA = 2

    def pi(s: State) -> int:
        if s not in Q:
            Q[s] = np.zeros(nA, dtype=np.float64)
        if np.random.rand() < eps:
            return np.random.randint(nA)
        return int(np.argmax(Q[s]))

    return pi


def generate_episode(env, policy_fn) -> List[Tuple[State, int, float]]:
    s, _ = env.reset()
    episode = []
    done = False
    while not done:
        a = policy_fn(s)
        ns, r, term, trunc, _ = env.step(a)
        episode.append((s, a, r))
        s = ns
        done = term or trunc
    return episode


# ---------- 1) Exploring Starts MC Control ----------
def run_exploring_starts(
    env_id="Blackjack-v1",
    gamma=1.0,
    episodes=300_000,
    eval_every=5_000,
    eval_episodes=50,
):
    env = gym.make(env_id, sab=True)
    nA = env.action_space.n
    Q: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(nA, dtype=np.float64))
    returns: Dict[Tuple[State, int], List[float]] = defaultdict(list)

    xs, ys = [], []
    for ep in range(1, episodes + 1):
        s, _ = env.reset()
        a = np.random.randint(nA)  # exploring start (random first action)
        episode = []
        done = False
        while not done:
            ns, r, term, trunc, _ = env.step(a)
            episode.append((s, a, r))
            s = ns
            done = term or trunc
            if not done:
                # greedy thereafter
                a = int(np.argmax(Q[s]))

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

        if ep % eval_every == 0:
            pi_greedy = greedy_from_Q(Q)
            avg_ret = eval_policy(env_id, pi_greedy, eval_episodes)
            xs.append(ep)
            ys.append(avg_ret)

    env.close()
    return xs, ys, Q


# ---------- 2) On-Policy First-Visit MC Control ----------
def run_onpolicy_fv(
    env_id="Blackjack-v1",
    gamma=1.0,
    episodes=500_000,
    eval_every=5_000,
    eval_episodes=50,
    eps_start=0.1,
    eps_end=0.01,
):
    env = gym.make(env_id, sab=True)
    nA = env.action_space.n
    Q: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(nA, dtype=np.float64))
    returns = defaultdict(list)

    xs, ys = [], []
    for ep in range(1, episodes + 1):
        # GLIE-ish linear decay
        frac = ep / float(episodes)
        eps = eps_start + (eps_end - eps_start) * frac
        policy = eps_greedy_from_Q(Q, eps)

        # generate episode
        episode = generate_episode(env, policy)

        # first-visit update on Q
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

        if ep % eval_every == 0:
            pi_greedy = greedy_from_Q(Q)
            avg_ret = eval_policy(env_id, pi_greedy, eval_episodes)
            xs.append(ep)
            ys.append(avg_ret)

    env.close()
    return xs, ys, Q


# ---------- 3) Off-Policy EV MC Control (Weighted IS) ----------
def run_offpolicy_wis(
    env_id="Blackjack-v1",
    gamma=1.0,
    episodes=500_000,
    eval_every=5_000,
    eval_episodes=50,
    behavior_eps=0.2,
):
    env = gym.make(env_id, sab=True)
    nA = env.action_space.n
    Q: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(nA, dtype=np.float64))
    C = defaultdict(float)  # cumulative weights for WIS
    greedy_prob = 1.0 - behavior_eps + behavior_eps / nA

    xs, ys = [], []
    for ep in range(1, episodes + 1):
        # behavior policy episode
        episode = []
        s, _ = env.reset()
        done = False
        while not done:
            a = eps_greedy_from_Q(Q, behavior_eps)(s)
            ns, r, term, trunc, _ = env.step(a)
            episode.append((s, a, r))
            s = ns
            done = term or trunc

        # weighted IS update
        G = 0.0
        W = 1.0
        for t in reversed(range(len(episode))):
            st, at, rt = episode[t]
            G = gamma * G + rt
            greedy_a = int(np.argmax(Q[st]))
            if at != greedy_a:
                break  # pi=0 ⇒ ratio 0 ⇒ stop
            b = greedy_prob
            W *= 1.0 / b  # pi=1 for greedy action
            C[(st, at)] += W
            Q[st][at] += (W / C[(st, at)]) * (G - Q[st][at])

        if ep % eval_every == 0:
            pi_greedy = greedy_from_Q(Q)
            avg_ret = eval_policy(env_id, pi_greedy, eval_episodes)
            xs.append(ep)
            ys.append(avg_ret)

    env.close()
    return xs, ys, Q


# ---------- plotting helpers ----------
def save_line(xs, ys, title, path):
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Episodes")
    plt.ylabel("Average return (greedy evaluation)")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main():
    # You can shorten episodes/eval_every while testing.
    es_x, es_y, _ = run_exploring_starts(episodes=150_000, eval_every=5_000)
    on_x, on_y, _ = run_onpolicy_fv(episodes=200_000, eval_every=5_000)
    off_x, off_y, _ = run_offpolicy_wis(episodes=200_000, eval_every=5_000)

    save_line(es_x, es_y, "Exploring Starts — Learning Curve", "es_learning_curve.png")
    save_line(on_x, on_y, "On-Policy FV MC — Learning Curve", "onpolicy_learning_curve.png")
    save_line(
        off_x, off_y, "Off-Policy EV MC (WIS) — Learning Curve", "offpolicy_wis_learning_curve.png"
    )

    # Combined figure for quick comparison
    plt.figure()
    plt.plot(es_x, es_y, label="Exploring Starts")
    plt.plot(on_x, on_y, label="On-Policy FV")
    plt.plot(off_x, off_y, label="Off-Policy EV (WIS)")
    plt.xlabel("Episodes")
    plt.ylabel("Average return (greedy evaluation)")
    plt.title("Blackjack — MC Control Learning Curves")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.savefig("mc_control_learning_curves.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
