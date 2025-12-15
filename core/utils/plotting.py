from typing import Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

BlackjackState = Tuple[int, int, bool]


def _finalize(fig, path: str | None, show: bool):
    if path:
        fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _grid_for_blackjack(V_or_Q: Mapping[BlackjackState, np.ndarray | float], is_q: bool):
    # Build two 10x10 grids: rows=player 12..21, cols=dealer 1..10
    ps_range = range(12, 22)
    dc_range = range(1, 11)

    def build(usable: bool):
        Z = np.full((len(ps_range), len(dc_range)), np.nan, dtype=np.float64)
        for i, ps in enumerate(ps_range):
            for j, dc in enumerate(dc_range):
                state = (ps, dc, usable)
                if state in V_or_Q:
                    if is_q:
                        Z[i, j] = np.max(V_or_Q[state])
                    else:
                        Z[i, j] = V_or_Q[state]
        return Z

    return build(True), build(False), ps_range, dc_range


def plot_value_heatmaps(
    V: Mapping[BlackjackState, float],
    title_prefix: str = "V",
    fname_prefix: str | None = None,
    show: bool = True,
):
    Z_ace, Z_noace, ps_range, dc_range = _grid_for_blackjack(V, is_q=False)
    for name, Z in [("usable_ace", Z_ace), ("no_usable_ace", Z_noace)]:
        fig, ax = plt.subplots()
        ax.imshow(Z, origin="lower", extent=[1, 10, 12, 21], aspect="auto")
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_title(f"{title_prefix} — {name}")
        fig.colorbar(ax.images[0], ax=ax, label="Value")
        path = f"{fname_prefix}_{name}.png" if fname_prefix else None
        _finalize(fig, path, show)


def plot_policy_from_Q(
    Q: Mapping[BlackjackState, np.ndarray],
    title: str = "Greedy policy (0=stick,1=hit)",
    fname: str | None = None,
    show: bool = True,
):
    Z_ace, Z_noace, ps_range, dc_range = _grid_for_blackjack(Q, is_q=True)

    def build_action(usable: bool):
        A = np.full((10, 10), np.nan)
        for i, ps in enumerate(range(12, 22)):
            for j, dc in enumerate(range(1, 11)):
                state = (ps, dc, usable)
                if state in Q:
                    A[i, j] = int(np.argmax(Q[state]))
        return A

    action_grids = [("usable_ace", build_action(True)), ("no_usable_ace", build_action(False))]

    for name, A in action_grids:
        fig, ax = plt.subplots()
        ax.imshow(A, origin="lower", extent=[1, 10, 12, 21], aspect="auto")
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_title(f"{title} — {name}")
        fig.colorbar(ax.images[0], ax=ax, label="Action (0=stick, 1=hit)")
        path = f"{fname}_{name}.png" if fname else None
        _finalize(fig, path, show)


def plot_episode_rewards(
    rewards: Sequence[float],
    per_episode: int = 1,
    title: str | None = None,
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    path: str | None = None,
    show: bool = True,
) -> List[float]:
    """Plot average reward per block of episodes."""
    if per_episode < 1:
        raise ValueError("per_episode must be >= 1")

    averaged = [
        float(np.mean(rewards[i : i + per_episode])) for i in range(0, len(rewards), per_episode)
    ]
    xs = list(range(per_episode, per_episode * len(averaged) + 1, per_episode))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, averaged)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    _finalize(fig, path, show)
    return averaged


def plot_q_table(
    q_table: Mapping[Tuple[object, object], float],
    states: Sequence[object],
    actions: Sequence[object],
    title: str,
    path: str | None = None,
    show: bool = True,
    cmap: str = "coolwarm",
    highlight_max: bool = True,
):
    """Visualize a Q-table for discrete state/action spaces."""
    state_list = list(states)
    action_list = list(actions)
    q_values = np.zeros((len(state_list), len(action_list)))

    for (state, action), value in q_table.items():
        try:
            state_index = state_list.index(state)
            action_index = action_list.index(action)
        except ValueError:
            continue
        q_values[state_index, action_index] = value

    fig, ax = plt.subplots(figsize=(10, 7))
    mesh = ax.pcolor(q_values, cmap=cmap, vmin=np.min(q_values), vmax=np.max(q_values))

    if highlight_max:
        for i in range(len(states)):
            max_index = int(np.argmax(q_values[i]))
            ax.scatter(max_index + 0.5, i + 0.5, marker="o", color="lime", s=100, zorder=10)

    ax.set_title(title)
    ax.set_xlabel("Actions")
    ax.set_ylabel("States")
    ax.set_xticks(np.arange(len(action_list)) + 0.5)
    ax.set_yticks(np.arange(len(state_list)) + 0.5)
    ax.set_xticklabels([str(a) for a in action_list])
    ax.set_yticklabels([str(s) for s in state_list])
    fig.colorbar(mesh, ax=ax, label="Q-value")
    ax.invert_yaxis()
    ax.grid(True)
    _finalize(fig, path, show)


def plot_learning_curve(
    xs: Sequence[float],
    ys: Sequence[float],
    title: str,
    path: str | None = None,
    xlabel: str = "Episodes",
    ylabel: str = "Average return",
    show: bool = True,
):
    """Simple line plot helper for learning curves."""
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    _finalize(fig, path, show)


def plot_multiple_learning_curves(
    series: Iterable[Tuple[str, Sequence[float], Sequence[float]]],
    title: str,
    path: str | None = None,
    xlabel: str = "Episodes",
    ylabel: str = "Average return",
    show: bool = True,
):
    """Plot several labeled curves on the same axes."""
    fig, ax = plt.subplots()
    for label, xs, ys in series:
        ax.plot(xs, ys, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    _finalize(fig, path, show)
