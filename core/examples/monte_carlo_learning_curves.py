from collections import defaultdict
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from core.algorithms.monte_carlo.common import evaluate_policy, generate_episode
from core.algorithms.monte_carlo.policies import greedy_policy, make_epsilon_greedy
from core.utils.plotting import plot_learning_curve, plot_multiple_learning_curves

State = Tuple[int, int, bool]


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
            pi_greedy = greedy_policy(Q, nA)
            avg_ret = evaluate_policy(env_id, pi_greedy, eval_episodes)
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
        policy = make_epsilon_greedy(Q, nA, eps)

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
            pi_greedy = greedy_policy(Q, nA)
            avg_ret = evaluate_policy(env_id, pi_greedy, eval_episodes)
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
            a = make_epsilon_greedy(Q, nA, behavior_eps)(s)
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
            pi_greedy = greedy_policy(Q, nA)
            avg_ret = evaluate_policy(env_id, pi_greedy, eval_episodes)
            xs.append(ep)
            ys.append(avg_ret)

    env.close()
    return xs, ys, Q


def main():
    # You can shorten episodes/eval_every while testing.
    es_x, es_y, _ = run_exploring_starts(episodes=150_000, eval_every=5_000)
    on_x, on_y, _ = run_onpolicy_fv(episodes=200_000, eval_every=5_000)
    off_x, off_y, _ = run_offpolicy_wis(episodes=200_000, eval_every=5_000)

    plot_learning_curve(
        es_x,
        es_y,
        "Exploring Starts — Learning Curve",
        path="es_learning_curve.png",
        ylabel="Average return (greedy evaluation)",
        show=False,
    )
    plot_learning_curve(
        on_x,
        on_y,
        "On-Policy FV MC — Learning Curve",
        path="onpolicy_learning_curve.png",
        ylabel="Average return (greedy evaluation)",
        show=False,
    )
    plot_learning_curve(
        off_x,
        off_y,
        "Off-Policy EV MC (WIS) — Learning Curve",
        path="offpolicy_wis_learning_curve.png",
        ylabel="Average return (greedy evaluation)",
        show=False,
    )

    plot_multiple_learning_curves(
        [
            ("Exploring Starts", es_x, es_y),
            ("On-Policy FV", on_x, on_y),
            ("Off-Policy EV (WIS)", off_x, off_y),
        ],
        title="Blackjack — MC Control Learning Curves",
        path="mc_control_learning_curves.png",
        ylabel="Average return (greedy evaluation)",
        show=False,
    )


if __name__ == "__main__":
    main()
