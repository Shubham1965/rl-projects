# visualize.py
import matplotlib.pyplot as plt
import numpy as np


def _grid_for_blackjack(V_or_Q, is_q: bool):
    # Build two 10x10 grids: rows=player 12..21, cols=dealer 1..10
    ps_range = range(12, 22)
    dc_range = range(1, 11)

    def build(usable):
        Z = np.full((len(ps_range), len(dc_range)), np.nan, dtype=np.float64)
        for i, ps in enumerate(ps_range):
            for j, dc in enumerate(dc_range):
                s = (ps, dc, usable)
                if is_q:
                    if s in V_or_Q:
                        Z[i, j] = np.max(V_or_Q[s])
                else:
                    if s in V_or_Q:
                        Z[i, j] = V_or_Q[s]
        return Z

    return build(True), build(False), ps_range, dc_range


def plot_value_heatmaps(V, title_prefix="V", fname_prefix=None):
    Z_ace, Z_noace, ps_range, dc_range = _grid_for_blackjack(V, is_q=False)
    for name, Z in [("usable_ace", Z_ace), ("no_usable_ace", Z_noace)]:
        plt.figure()
        plt.imshow(Z, origin="lower", extent=[1, 10, 12, 21], aspect="auto")
        plt.colorbar(label="Value")
        plt.xlabel("Dealer showing")
        plt.ylabel("Player sum")
        plt.title(f"{title_prefix} — {name}")
        if fname_prefix:
            plt.savefig(f"{fname_prefix}_{name}.png", bbox_inches="tight")
        plt.show()


def plot_policy_from_Q(Q, title="Greedy policy (0=stick,1=hit)", fname=None):
    # 0=stick, 1=hit based on argmax_a Q(s,a)
    Z_ace, Z_noace, ps_range, dc_range = _grid_for_blackjack(Q, is_q=True)

    # convert value grids to action grids: we need the action, not the max value
    def build_action(usable):
        A = np.full((10, 10), np.nan)
        for i, ps in enumerate(range(12, 22)):
            for j, dc in enumerate(range(1, 11)):
                s = (ps, dc, usable)
                if s in Q:
                    A[i, j] = int(np.argmax(Q[s]))
        return A

    A_ace = build_action(True)
    A_noace = build_action(False)

    for name, A in [("usable_ace", A_ace), ("no_usable_ace", A_noace)]:
        plt.figure()
        plt.imshow(A, origin="lower", extent=[1, 10, 12, 21], aspect="auto")
        plt.colorbar(label="Action (0=stick, 1=hit)")
        plt.xlabel("Dealer showing")
        plt.ylabel("Player sum")
        plt.title(f"{title} — {name}")
        if fname:
            plt.savefig(f"{fname}_{name}.png", bbox_inches="tight")
        plt.show()
