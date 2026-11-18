import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.custom_env import A_RESTOCK, A_SCREEN, A_SUPP, HOUSES, OBS_DIM, NutriBridgeEnv


def test_gym_api():
    check_env(NutriBridgeEnv(), skip_render_check=True)


def test_obs_shape_and_range():
    env = NutriBridgeEnv(); o, _ = env.reset(seed=0)
    assert o.shape == (OBS_DIM,) and o.min() >= 0 and o.max() <= 1


def test_restock_only_at_clinic():
    env = NutriBridgeEnv(); env.reset(seed=0)
    env.step(A_SUPP)          # supplement at clinic: no household -> penalty
    env.step(0)               # travel to H0
    _, r, *_ = env.step(A_RESTOCK)  # restock away from clinic: invalid
    assert r < 0


def test_episode_terminates():
    env = NutriBridgeEnv(); env.reset(seed=1); done = False; n = 0
    while not done:
        _, _, done, _, _ = env.step(A_SCREEN); n += 1
    assert n <= 240


def test_action_changes_state():
    env = NutriBridgeEnv(); env.reset(seed=2)
    env.pos = list(HOUSES[0]); s0 = env.supp[0]
    env.step(A_SUPP)
    assert env.supp[0] > s0 and env.stock == 5
