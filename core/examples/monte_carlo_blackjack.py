import numpy as np

from core.algorithms.monte_carlo.control import (
    mc_control_exploring_starts,
    offpolicy_every_visit_mc_control_is,
    onpolicy_first_visit_mc_control,
)
from core.algorithms.monte_carlo.mc_prediction import mc_state_value_prediction
from core.utils.plotting import plot_policy_from_Q, plot_value_heatmaps


def random_policy_blackjack(state):  # 50/50 hit/stick
    return np.random.randint(2)


def main():
    # 1) MC prediction (First-Visit)
    V_fv = mc_state_value_prediction(
        "Blackjack-v1", random_policy_blackjack, gamma=1.0, episodes=200_000, first_visit=True
    )
    print(f"[MC Prediction FV] states learned: {len(V_fv)}")

    # 2) MC prediction (Every-Visit)
    V_ev = mc_state_value_prediction(
        "Blackjack-v1", random_policy_blackjack, gamma=1.0, episodes=200_000, first_visit=False
    )
    print(f"[MC Prediction EV] states learned: {len(V_ev)}")

    # 3) MC Control — Exploring Starts
    Q_es, pi_es = mc_control_exploring_starts("Blackjack-v1", gamma=1.0, episodes=300_000)
    print(f"[MC Control ES] Q states: {len(Q_es)}")

    # 4) On-policy First-Visit MC Control (ε-greedy)
    Q_on = onpolicy_first_visit_mc_control("Blackjack-v1", gamma=1.0, episodes=500_000, epsilon=0.1)
    print(f"[On-Policy FV Control] Q states: {len(Q_on)}")

    # 5) Off-policy Every-Visit MC Control with Weighted IS
    Q_off = offpolicy_every_visit_mc_control_is(
        "Blackjack-v1", gamma=1.0, episodes=500_000, behavior_epsilon=0.2, weighted=True
    )
    print(f"[Off-Policy EV Control (WIS)] Q states: {len(Q_off)}")

    # For prediction (V)
    plot_value_heatmaps(V_fv, title_prefix="First-Visit MC V", fname_prefix="../../results/fv_V")
    plot_value_heatmaps(V_ev, title_prefix="Every-Visit MC V", fname_prefix="../../results/fv_V")

    # For control (Q → policy)
    plot_policy_from_Q(Q_es, title="Exploring Starts Greedy Policy", fname="../../results/es_policy")
    plot_policy_from_Q(Q_on, title="On-Policy FV MC Greedy Policy", fname="../../results/onpolicy_policy")
    plot_policy_from_Q(
        Q_off, title="Off-Policy EV MC (WIS) Greedy Policy", fname="../../results/offpolicy_wis_policy"
    )


if __name__ == "__main__":
    main()
