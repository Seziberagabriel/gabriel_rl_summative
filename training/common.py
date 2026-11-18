"""Shared helpers: TorchRL env factory, evaluation, run tracking, checkpoint I/O."""
from __future__ import annotations

import json
import os
import time
import warnings

from pathlib import Path

import numpy as np
import torch
from torch import nn
try:  # TorchRL >= 0.15 renamed SyncDataCollector -> Collector
    from torchrl.collectors import Collector as SyncDataCollector
except ImportError:
    from torchrl.collectors import SyncDataCollector
from torchrl.envs import Compose, GymWrapper, RewardSum, SerialEnv, TransformedEnv
from torchrl.modules import MLP

from environment.custom_env import N_ACTIONS, OBS_DIM, NutriBridgeEnv

warnings.filterwarnings("ignore", module="torchrl")
os.environ.setdefault("RL_WARNINGS", "False")
ROOT = Path(__file__).resolve().parents[1]
LOGS, MODELS = ROOT / "logs", ROOT / "models"
PERIODIC_EVAL_SEEDS = list(range(5000, 5010))    # model selection during training
FINAL_EVAL_SEEDS = list(range(10000, 10030))     # held-out seeds, never seen in training


def make_env(n_envs: int = 1, seed: int = 0):
    """Same NutriBridge env for every algorithm, wrapped for TorchRL."""
    def _mk():
        return GymWrapper(NutriBridgeEnv(), categorical_action_encoding=True)
    base = _mk() if n_envs == 1 else SerialEnv(n_envs, _mk)
    env = TransformedEnv(base, Compose(RewardSum()))
    env.set_seed(seed)
    return env


def make_mlp(out_features: int, hidden: int = 128, layers: int = 2) -> nn.Module:
    return MLP(in_features=OBS_DIM, out_features=out_features,
               num_cells=[hidden] * layers, activation_class=nn.ReLU)


def episode_returns(batch) -> list[float]:
    done = batch["next", "done"].squeeze(-1)
    return batch["next", "episode_reward"].squeeze(-1)[done].tolist()


@torch.no_grad()
def evaluate(net: nn.Module, seeds, stochastic: bool = False) -> dict:
    """Rollouts on a raw Gym env. DQN: greedy argmax over Q. Policy gradient: sample from the policy."""
    env = NutriBridgeEnv()
    rets, succ, fails = [], [], []
    for s in seeds:
        o, _ = env.reset(seed=int(s))
        torch.manual_seed(int(s))
        done, G = False, 0.0
        while not done:
            out = net(torch.as_tensor(o, dtype=torch.float32))
            a = int(torch.distributions.Categorical(logits=out).sample()) if stochastic else int(out.argmax())
            o, r, done, _, info = env.step(a)
            G += r
        rets.append(G); succ.append(info["frac_ok"]); fails.append(float(env.severe_streak >= 3))
    return {"mean_return": float(np.mean(rets)), "std_return": float(np.std(rets)),
            "success": float(np.mean(succ)), "fail_rate": float(np.mean(fails))}


class Tracker:
    """Collects curves + handles periodic evaluation and best-checkpoint keeping."""

    def __init__(self, net, total_frames, n_evals=10, stochastic=False):
        self.net, self.total, self.stochastic = net, total_frames, stochastic
        self.train, self.loss, self.entropy, self.evals = [], [], [], []
        self.frames, self.t0 = 0, time.time()
        self.next_eval = total_frames // n_evals
        self.step_eval = total_frames // n_evals
        self.best_score, self.best_state = -1e9, None

    def add_batch(self, batch, n_frames, loss=None, entropy=None):
        self.frames += n_frames
        self.train += [(self.frames, r) for r in episode_returns(batch)]
        if loss is not None: self.loss.append((self.frames, float(loss)))
        if entropy is not None: self.entropy.append((self.frames, float(entropy)))
        if self.frames >= self.next_eval:
            ev = evaluate(self.net, PERIODIC_EVAL_SEEDS, self.stochastic)
            self.evals.append((self.frames, ev["mean_return"], ev["success"]))
            if ev["mean_return"] > self.best_score:
                self.best_score = ev["mean_return"]
                self.best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            self.next_eval += self.step_eval

    def finish(self, algo, run_id, cfg) -> dict:
        last = evaluate(self.net, FINAL_EVAL_SEEDS, self.stochastic)
        last_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        if self.best_state is None:
            self.best_state = last_state
        self.net.load_state_dict(self.best_state)
        best = evaluate(self.net, FINAL_EVAL_SEEDS, self.stochastic)
        sub = "dqn" if algo == "dqn" else "pg"
        (MODELS / sub).mkdir(parents=True, exist_ok=True)
        path = MODELS / sub / f"{algo}_run_{run_id:02d}.pt"
        torch.save({"algo": algo, "state_dict": self.best_state, "cfg": cfg,
                    "hidden": cfg["hidden"], "layers": cfg["layers"], "stochastic": self.stochastic}, path)
        res = {"algo": algo, "run_id": run_id, "cfg": cfg, "frames": self.frames,
               "train": self.train, "loss": self.loss, "entropy": self.entropy,
               "evals": self.evals, "final_weights_eval": last, "best_ckpt_eval": best,
               "model_path": str(path.relative_to(ROOT)), "wall_seconds": time.time() - self.t0}
        (LOGS / algo).mkdir(parents=True, exist_ok=True)
        with open(LOGS / algo / f"run_{run_id:02d}.json", "w") as f:
            json.dump(res, f)
        return res


def load_net(path: str | Path) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = make_mlp(N_ACTIONS, ck["hidden"], ck["layers"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck
