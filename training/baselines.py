"""Random and rule-based baselines (used to justify why RL is needed)."""
import numpy as np

from environment.custom_env import (A_CLINIC, A_REFER, A_RESTOCK, A_SCREEN, A_SUPP, CLINIC,
                                    GRID, HOUSES, MAX_STOCK, N_ACTIONS, N_HH)


def greedy_policy(obs: np.ndarray) -> int:
    """'Always visit the child who looks worst', screening when stale, restocking when empty."""
    kz, since = obs[:N_HH], obs[N_HH:2 * N_HH]
    pos = (round(obs[3 * N_HH] * (GRID - 1)), round(obs[3 * N_HH + 1] * (GRID - 1)))
    stock = round(obs[3 * N_HH + 2] * MAX_STOCK)
    if stock == 0:
        return A_RESTOCK if pos == CLINIC else A_CLINIC
    t = int(np.argmin(kz))
    if pos != HOUSES[t]:
        return t
    if since[t] > 2 / 24:
        return A_SCREEN
    if kz[t] * 6 - 4 < -3.0:
        return A_REFER
    return A_SUPP


def random_policy(obs, rng=np.random.default_rng(0)):
    return int(rng.integers(0, N_ACTIONS))
