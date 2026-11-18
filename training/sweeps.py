"""Hyperparameter sweeps: 10 configurations per algorithm, same env, seed and frame budget.

Usage:
    uv run python -m training.sweeps --algo all --frames 100000 --workers 4
    uv run python -m training.sweeps --algo ppo --frames 100000
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from training.baselines import greedy_policy, random_policy
from training.common import FINAL_EVAL_SEEDS, LOGS, MODELS, ROOT

# Each entry changes ONE (or a clearly named pair of) settings vs. the default in the training file.
SWEEPS = {
    "dqn": [
        {"name": "default"},
        {"name": "lr=3e-4", "lr": 3e-4},
        {"name": "lr=3e-3", "lr": 3e-3},
        {"name": "gamma=0.95", "gamma": 0.95},
        {"name": "gamma=0.999", "gamma": 0.999},
        {"name": "fast eps decay (20%, end 0.01)", "eps_decay_frac": 0.2, "eps_end": 0.01},
        {"name": "slow eps decay (80%, end 0.10)", "eps_decay_frac": 0.8, "eps_end": 0.10},
        {"name": "small replay buffer (5k)", "buffer": 5_000},
        {"name": "hard target update /500", "target": "hard", "hard_interval": 500},
        {"name": "double DQN + 256 units", "double": True, "hidden": 256},
    ],
    "reinforce": [
        {"name": "default"},
        {"name": "lr=3e-4", "lr": 3e-4},
        {"name": "lr=3e-3", "lr": 3e-3},
        {"name": "no baseline (reward-to-go)", "baseline": False},
        {"name": "gamma=0.95", "gamma": 0.95},
        {"name": "entropy=0", "entropy": 0.0},
        {"name": "entropy=0.05", "entropy": 0.05},
        {"name": "batch=1200 frames", "frames_per_batch": 1200},
        {"name": "batch=4800 frames", "frames_per_batch": 4800},
        {"name": "no advantage normalisation", "norm_adv": False},
    ],
    "ppo": [
        {"name": "default"},
        {"name": "lr=1e-3", "lr": 1e-3},
        {"name": "lr=1e-4", "lr": 1e-4},
        {"name": "clip=0.1", "clip": 0.1},
        {"name": "clip=0.3", "clip": 0.3},
        {"name": "GAE lambda=0.8", "lmbda": 0.8},
        {"name": "GAE lambda=0.99", "lmbda": 0.99},
        {"name": "entropy=0", "entropy": 0.0},
        {"name": "entropy=0.05", "entropy": 0.05},
        {"name": "10 epochs/batch", "epochs": 10},
    ],
    "a2c": [
        {"name": "default"},
        {"name": "lr=3e-4", "lr": 3e-4},
        {"name": "lr=3e-3", "lr": 3e-3},
        {"name": "gamma=0.95", "gamma": 0.95},
        {"name": "GAE lambda=1.0", "lmbda": 1.0},
        {"name": "GAE lambda=0.8", "lmbda": 0.8},
        {"name": "entropy=0", "entropy": 0.0},
        {"name": "entropy=0.05", "entropy": 0.05},
        {"name": "16 envs x 32 steps", "n_envs": 16, "frames_per_batch": 512},
        {"name": "4 envs x 8 steps", "n_envs": 4, "frames_per_batch": 32},
    ],
}


def run_one(args):
    algo, i, cfg, frames = args
    from training.dqn_training import run_dqn
    from training.pg_training import run_pg
    import torch
    torch.set_num_threads(1)
    cfg = dict(cfg)
    res = run_dqn(i, cfg, frames) if algo == "dqn" else run_pg(algo, i, cfg, frames)
    f = res["final_weights_eval"]
    print(f"[{algo} #{i:02d}] {cfg['name']:<34} return={f['mean_return']:+.2f} success={f['success']:.2f} ({res['wall_seconds']:.0f}s)", flush=True)
    return algo, i


def baselines():
    from environment.custom_env import NutriBridgeEnv
    out = {}
    for name, pol in [("random", random_policy), ("greedy_rule", greedy_policy)]:
        env, R, S, F = NutriBridgeEnv(), [], [], []
        for s in FINAL_EVAL_SEEDS:
            o, _ = env.reset(seed=s); done, G = False, 0.0
            while not done:
                o, r, done, _, info = env.step(pol(o)); G += r
            R.append(G); S.append(info["frac_ok"]); F.append(float(env.severe_streak >= 3))
        out[name] = {"mean_return": float(np.mean(R)), "std_return": float(np.std(R)),
                     "success": float(np.mean(S)), "fail_rate": float(np.mean(F))}
    json.dump(out, open(LOGS / "baselines.json", "w"), indent=2)
    return out


def summarise():
    rows = []
    for p in sorted(LOGS.glob("*/run_*.json")):
        r = json.load(open(p))
        tr = np.array([x[1] for x in r["train"]]) if r["train"] else np.zeros(1)
        k = max(5, len(tr) // 5)
        rows.append({"algo": r["algo"], "run": r["run_id"], "config": r["cfg"].get("name", ""),
                     "train_last20pct_mean": tr[-k:].mean(), "train_last20pct_std": tr[-k:].std(),
                     "train_best_ma20": float(np.convolve(tr, np.ones(min(20, len(tr))) / min(20, len(tr)), "valid").max()),
                     "eval_return_final": r["final_weights_eval"]["mean_return"],
                     "eval_success_final": r["final_weights_eval"]["success"],
                     "eval_fail_rate_final": r["final_weights_eval"]["fail_rate"],
                     "eval_return_bestckpt": r["best_ckpt_eval"]["mean_return"],
                     "eval_success_bestckpt": r["best_ckpt_eval"]["success"],
                     "frames": r["frames"], "wall_s": round(r["wall_seconds"]), "model": r["model_path"]})
    df = pd.DataFrame(rows)
    df.to_csv(LOGS / "summary.csv", index=False)
    if len(df):
        top = df.sort_values("eval_return_bestckpt", ascending=False).iloc[0]
        json.dump({"algo": top["algo"], "model": top["model"], "config": top["config"],
                   "eval_return": top["eval_return_bestckpt"], "eval_success": top["eval_success_bestckpt"]},
                  open(MODELS / "best.json", "w"), indent=2)
        print(df.round(2).to_string(index=False)); print("\nBest overall ->", top["algo"], top["config"])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="all", choices=["all", *SWEEPS])
    ap.add_argument("--frames", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", type=int, nargs="*", help="run only these config indices")
    a = ap.parse_args()
    jobs = [(al, i, c, a.frames) for al in (SWEEPS if a.algo == "all" else [a.algo])
            for i, c in enumerate(SWEEPS[al]) if a.only is None or i in a.only]
    print(f"{len(jobs)} runs, {a.frames} frames each, {a.workers} worker(s)")
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(run_one, jobs))
    else:
        for j in jobs:
            run_one(j)
    print("Baselines:", baselines())
    summarise()
