"""Generate the report figures from logs/: uv run python -m training.plot_results"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.common import LOGS, ROOT

ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
COL = {"dqn": "tab:blue", "reinforce": "tab:orange", "ppo": "tab:green", "a2c": "tab:red"}
OUT = ROOT / "assets"


def load(algo):
    return [json.load(open(p)) for p in sorted((LOGS / algo).glob("run_*.json"))]


def curve(pairs, grid, w=20):
    """Moving average of episode returns, interpolated on a common frame grid."""
    if len(pairs) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    f, r = np.array(pairs).T
    k = min(w, len(r)); ma = np.convolve(r, np.ones(k) / k, "valid")
    return np.interp(grid, f[k - 1:], ma, left=np.nan)


def main():
    OUT.mkdir(exist_ok=True)
    runs = {a: load(a) for a in ALGOS}
    base = json.load(open(LOGS / "baselines.json")) if (LOGS / "baselines.json").exists() else {}
    F = max(r["frames"] for a in ALGOS for r in runs[a])
    grid = np.linspace(F * 0.02, F, 200)
    thr = base.get("greedy_rule", {}).get("mean_return")

    # 1. comparative reward curves (default config bold, other sweep runs faint; mean of all runs as band)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a in ALGOS:
        cs = np.array([curve(r["train"], grid) for r in runs[a]])
        d = [c for r, c in zip(runs[a], cs) if r["cfg"].get("name") == "default"]
        for c in cs: ax[0].plot(grid, c, color=COL[a], alpha=0.12, lw=0.8)
        if d: ax[0].plot(grid, d[0], color=COL[a], lw=2, label=a.upper() + " (default)")
        ax[1].plot(grid, np.nanmean(cs, 0), color=COL[a], lw=2, label=a.upper())
        ax[1].fill_between(grid, np.nanmin(cs, 0), np.nanmax(cs, 0), color=COL[a], alpha=0.12)
    for x in ax:
        if thr is not None: x.axhline(thr, color="k", ls="--", lw=1, label="greedy rule baseline")
        x.set_xlabel("Environment frames"); x.set_ylabel("Episode return (20-ep moving avg)"); x.grid(alpha=.3)
    ax[0].set_title("Default config (bold) vs. all sweep runs (faint)"); ax[1].set_title("Mean and min-max band over 10 sweep runs")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig1_comparative_rewards.png", dpi=150); plt.close(fig)

    # 2. DQN objective (TD loss)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in runs["dqn"]:
        if r["loss"]:
            f, l = np.array(r["loss"]).T
            k = max(1, len(l) // 40); ax.plot(f[k - 1:], np.convolve(l, np.ones(k) / k, "valid"), alpha=.7, lw=1,
                                              label=r["cfg"]["name"] if r["cfg"]["name"] == "default" else None)
    ax.set_yscale("log"); ax.set_xlabel("Environment frames"); ax.set_ylabel("DQN smooth-L1 TD loss (smoothed)")
    ax.set_title("DQN objective, all 10 runs"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig2_dqn_loss.png", dpi=150); plt.close(fig)

    # 3. policy-gradient entropy (max entropy = ln 14 = 2.639 for a uniform policy)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for x, a in zip(ax, ["reinforce", "ppo", "a2c"]):
        for r in runs[a]:
            if r["entropy"]:
                f, e = np.array(r["entropy"]).T
                k = max(1, len(e) // 15)
                x.plot(f[k - 1:], np.convolve(e, np.ones(k) / k, "valid"), lw=2 if r["cfg"]["name"] == "default" else .8,
                       alpha=1 if r["cfg"]["name"] == "default" else .5, label=r["cfg"]["name"] if r["cfg"]["name"] == "default" else None)
        x.axhline(np.log(14), color="k", ls=":", lw=1); x.set_title(a.upper()); x.set_xlabel("Environment frames"); x.grid(alpha=.3)
    ax[0].set_ylabel("Policy entropy (nats)"); ax[0].legend(); fig.tight_layout()
    fig.savefig(OUT / "fig3_pg_entropy.png", dpi=150); plt.close(fig)

    # 4. convergence: frames until 20-ep moving average first reaches the greedy-rule return
    fig, ax = plt.subplots(figsize=(7, 4)); conv = {}
    for i, a in enumerate(ALGOS):
        v = []
        for r in runs[a]:
            c = curve(r["train"], grid); idx = np.where(c >= thr)[0] if thr is not None else []
            v.append(grid[idx[0]] if len(idx) else np.nan)
        conv[a] = v
        ax.scatter([i] * len(v), [x if not np.isnan(x) else F * 1.05 for x in v], color=COL[a], alpha=.7)
        ax.text(i, F * 1.12, f"{int(np.sum(~np.isnan(v)))}/{len(v)} reached", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels([a.upper() for a in ALGOS]); ax.set_ylim(0, F * 1.2)
    ax.axhline(F * 1.05, color="gray", ls=":", lw=1); ax.set_ylabel("Frames to reach greedy-rule return")
    ax.set_title("Convergence speed (points at the dotted line = never reached)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig4_convergence.png", dpi=150); plt.close(fig)

    # 5. generalisation: 30 held-out seeds, best checkpoint per algorithm (+ baselines)
    best = {a: max(runs[a], key=lambda r: r["best_ckpt_eval"]["mean_return"]) for a in ALGOS}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = [a.upper() for a in ALGOS] + ["Greedy rule", "Random"]
    rets = [best[a]["best_ckpt_eval"]["mean_return"] for a in ALGOS] + [base.get("greedy_rule", {}).get("mean_return", 0), base.get("random", {}).get("mean_return", 0)]
    errs = [best[a]["best_ckpt_eval"]["std_return"] for a in ALGOS] + [base.get("greedy_rule", {}).get("std_return", 0), base.get("random", {}).get("std_return", 0)]
    succ = [best[a]["best_ckpt_eval"]["success"] for a in ALGOS] + [base.get("greedy_rule", {}).get("success", 0), base.get("random", {}).get("success", 0)]
    cols = [COL[a] for a in ALGOS] + ["gray", "lightgray"]
    ax[0].bar(names, rets, yerr=errs, color=cols, capsize=3); ax[0].set_ylabel("Mean return, 30 unseen seeds (± std)")
    ax[1].bar(names, succ, color=cols); ax[1].set_ylabel("Children with z >= -2 at week 24"); ax[1].set_ylim(0, 1)
    for x in ax: x.tick_params(axis="x", rotation=25); x.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig5_generalisation.png", dpi=150); plt.close(fig)

    json.dump({a: {"best_run": best[a]["run_id"], "config": best[a]["cfg"]["name"], **best[a]["best_ckpt_eval"]} for a in ALGOS} | {"frames_to_threshold": {a: [None if np.isnan(x) else float(x) for x in conv[a]] for a in ALGOS}},
              open(LOGS / "figure_numbers.json", "w"), indent=2)
    print("Figures written to", OUT)


if __name__ == "__main__":
    main()
