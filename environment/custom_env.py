"""NutriBridge CHW Planner: a custom Gymnasium environment.

A community health worker (CHW) supports 8 households, each with one child under 5,
over a 24-week season on a 7x7 village grid. The agent must trade off travel time,
a limited supplement stock, slow-but-lasting education, and costly clinic referrals,
while a lean season (weeks 10-15) pushes household food security down.

NOTE: dynamics are SIMULATED, with parameters inspired by WHO child-growth
standards (z < -2 moderate, z < -3 severe). They are not fitted to real data.
"""
from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
from gymnasium import spaces

GRID = 7
CLINIC = (3, 3)
HOUSES = [(0, 1), (1, 5), (2, 2), (4, 0), (5, 4), (6, 2), (3, 6), (0, 6)]
N_HH = len(HOUSES)
WEEK_TICKS = 10            # one week = 10 half-day work slots ("ticks")
N_WEEKS = 24
MAX_TICKS = WEEK_TICKS * N_WEEKS
MAX_STOCK = 6              # supplement packs the CHW can carry
LEAN_WEEKS = range(10, 16)
Z_MIN, Z_MAX = -4.0, 2.0
MODERATE, SEVERE = -2.0, -3.0

LOCATIONS = HOUSES + [CLINIC]            # index 0-7 = households, 8 = health centre
ACTION_NAMES = [f"Travel to H{i}" for i in range(N_HH)] + [
    "Travel to clinic", "Screen child", "Give supplement", "Nutrition education",
    "Refer to clinic", "Restock supplies"]
N_ACTIONS = len(ACTION_NAMES)            # 14
A_CLINIC, A_SCREEN, A_SUPP, A_EDU, A_REFER, A_RESTOCK = range(N_HH, N_ACTIONS)
# time cost (ticks) of the non-travel actions; travel costs the walking distance (Manhattan)
TICK_COST = {A_SCREEN: 1, A_SUPP: 1, A_EDU: 2, A_REFER: 4, A_RESTOCK: 2}
OBS_DIM = N_HH * 3 + 5 + (N_HH + 1)  # 38


class NutriBridgeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self._renderer = None
        self.last_action = None
        self.last_reward = 0.0
        self.episode_return = 0.0

    # ------------------------------------------------------------------ helpers
    def _week(self) -> int:
        return min(self.tick // WEEK_TICKS, N_WEEKS - 1)

    def _is_lean(self) -> bool:
        return self._week() in LEAN_WEEKS

    def _house_at(self, pos) -> int:
        return HOUSES.index(tuple(pos)) if tuple(pos) in HOUSES else -1

    def _frac_ok(self) -> float:
        return float(np.mean(self.z >= MODERATE))

    def _food_now(self) -> np.ndarray:
        return np.clip(self.food_base - (0.28 if self._is_lean() else 0.0), 0.0, 1.0)

    def _get_obs(self) -> np.ndarray:
        kz = (self.known_z - Z_MIN) / (Z_MAX - Z_MIN)
        since = np.minimum(self.since, N_WEEKS) / N_WEEKS
        food = np.clip(self.food_obs, 0, 1)
        x, y = self.pos
        h = self._house_at(self.pos)
        onehot = np.zeros(N_HH + 1, dtype=np.float32)
        onehot[h if h >= 0 else N_HH] = 1.0 if (h >= 0 or tuple(self.pos) == CLINIC) else 0.0
        glob = [x / (GRID - 1), y / (GRID - 1), self.stock / MAX_STOCK,
                self._week() / (N_WEEKS - 1), float(self._is_lean())]
        return np.concatenate([kz, since, food, glob, onehot]).astype(np.float32)

    # -------------------------------------------------------------------- reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.tick = 0
        self.pos = list(CLINIC)
        self.stock = MAX_STOCK
        self.z = np.clip(rng.normal(-1.5, 0.9, N_HH), -3.4, 0.5)
        self.food_base = rng.uniform(0.3, 0.8, N_HH)
        self.edu = np.zeros(N_HH)
        self.supp = np.zeros(N_HH)
        # baseline registration data from the platform: stale and noisy
        self.known_z = np.clip(self.z + rng.normal(0, 0.5, N_HH), Z_MIN, Z_MAX)
        self.since = np.full(N_HH, float(N_WEEKS))
        self.food_obs = np.clip(self._food_now() + rng.normal(0, 0.05, N_HH), 0, 1)
        self.severe_streak = 0
        self.last_action = None
        self.last_reward = 0.0
        self.episode_return = 0.0
        self.last_event = "Season begins at the clinic"
        return self._get_obs(), self._info()

    def _info(self):
        return {"week": self._week(), "frac_ok": self._frac_ok(),
                "n_severe": int(np.sum(self.z < SEVERE)), "stock": int(self.stock),
                "mean_z": float(self.z.mean())}

    # --------------------------------------------------------- weekly dynamics
    def _weekly_update(self) -> float:
        rng = self.np_random
        z_old = self.z.copy()
        food = self._food_now()
        feff = np.clip(food + 0.25 * self.edu, 0, 1)
        shock = (rng.random(N_HH) < 0.03) * -0.5          # illness episodes
        dz = 0.45 * (feff - 0.5) + 0.30 * self.supp + shock + rng.normal(0, 0.04, N_HH)
        self.z = np.clip(self.z + dz, Z_MIN, Z_MAX)
        self.supp *= 0.5                                   # supplement effect fades
        self.edu *= 0.97                                   # knowledge decays slowly
        self.since += 1
        self.food_obs = np.clip(self._food_now() + rng.normal(0, 0.05, N_HH), 0, 1)

        r = 3.0 * float(self.z.mean() - z_old.mean())
        r += 1.0 * int(np.sum((z_old < MODERATE) & (self.z >= MODERATE)))
        r -= 1.0 * int(np.sum((z_old >= MODERATE) & (self.z < MODERATE)))
        r -= 2.0 * int(np.sum((z_old >= SEVERE) & (self.z < SEVERE)))
        n_sev = int(np.sum(self.z < SEVERE))
        self.severe_streak = self.severe_streak + 1 if n_sev >= 2 else 0
        return r

    # -------------------------------------------------------------------- step
    def step(self, action: int):
        action = int(action)
        self.last_action = action
        r = 0.0
        h = self._house_at(self.pos)
        at_clinic = tuple(self.pos) == CLINIC
        event = ACTION_NAMES[action]

        cost = TICK_COST.get(action, 1)
        if action <= A_CLINIC:                         # travel: time cost = walking distance
            tx, ty = LOCATIONS[action]
            dist = abs(tx - self.pos[0]) + abs(ty - self.pos[1])
            if dist == 0:
                r -= 0.02; event += " (already here)"
            else:
                self.pos = [tx, ty]; cost = dist
        elif action == A_RESTOCK:
            if at_clinic and self.stock < MAX_STOCK:
                self.stock = MAX_STOCK
            else:
                r -= 0.05; event += " (invalid)"
        elif h < 0:
            r -= 0.05; event += " (no household here)"
        elif action == A_SCREEN:
            self.known_z[h] = np.clip(self.z[h] + self.np_random.normal(0, 0.1), Z_MIN, Z_MAX)
            self.since[h] = 0
            event += f" H{h}"
        elif action == A_SUPP:
            if self.stock > 0:
                self.stock -= 1
                self.supp[h] = min(1.0, self.supp[h] + 0.7)
                r -= 0.02
                event += f" H{h}"
            else:
                r -= 0.05; event += " (no stock)"
        elif action == A_EDU:
            self.edu[h] = min(1.0, self.edu[h] + 0.34)
            event += f" H{h}"
        elif action == A_REFER:
            if self.z[h] < -2.5:                       # clinic treatment resets severe case
                self.z[h] = -1.5
                self.supp[h] = 0.5
                self.known_z[h] = self.z[h]
                self.since[h] = 0
                event += f" H{h} (treated)"
            else:
                r -= 0.1; event += f" H{h} (not needed)"

        # advance the clock, applying weekly dynamics at each week boundary
        terminated = False
        for _ in range(cost):
            self.tick += 1
            if self.tick % WEEK_TICKS == 0:
                r += self._weekly_update()
            if self.tick >= MAX_TICKS:
                break
        if self.severe_streak >= 3:                    # sustained crisis: early failure
            r -= 5.0
            terminated = True
            event = "FAILURE: sustained severe malnutrition"
        elif self.tick >= MAX_TICKS:                   # season over
            r += 5.0 * self._frac_ok()
            terminated = True
            event = f"Season over: {self._frac_ok():.0%} of children z >= -2"

        self.last_reward = r
        self.episode_return += r
        self.last_event = event
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(r), terminated, False, self._info()

    # ------------------------------------------------------ integration / export
    def export_state(self) -> dict:
        """JSON-serialisable snapshot; consumed by a web (Three.js) / mobile front end."""
        return {
            "tick": int(self.tick), "week": self._week(), "lean_season": self._is_lean(),
            "chw": {"x": int(self.pos[0]), "y": int(self.pos[1])},
            "clinic": {"x": CLINIC[0], "y": CLINIC[1]}, "stock": int(self.stock),
            "households": [{"id": i, "x": HOUSES[i][0], "y": HOUSES[i][1],
                            "z_true": round(float(self.z[i]), 3),
                            "z_known": round(float(self.known_z[i]), 3),
                            "weeks_since_screen": float(self.since[i]),
                            "food_security": round(float(self.food_base[i]), 3),
                            "status": ("ok" if self.z[i] >= MODERATE else
                                       "moderate" if self.z[i] >= SEVERE else "severe")}
                           for i in range(N_HH)],
            "last_action": None if self.last_action is None else ACTION_NAMES[self.last_action],
            "event": self.last_event, "reward": round(self.last_reward, 3),
            "episode_return": round(self.episode_return, 3),
        }

    def export_json(self, path: str, frames: list[dict]):
        with open(path, "w") as f:
            json.dump({"grid": GRID, "frames": frames}, f)

    # ---------------------------------------------------------------- rendering
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import Renderer
            self._renderer = Renderer(headless=self.render_mode == "rgb_array")
        return self._renderer.draw(self, return_array=self.render_mode == "rgb_array")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
