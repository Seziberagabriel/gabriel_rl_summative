"""Run the best trained agent in the NutriBridge CHW Planner with the pygame GUI.

    uv run main.py                      # best model overall (models/best.json), 3 episodes
    uv run main.py --algo ppo           # best saved model of one algorithm
    uv run main.py --episodes 5 --seed 42 --export-json assets/episode.json
    uv run main.py --no-gui             # terminal only (e.g. headless server)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def pick_model(algo: str | None) -> Path:
    best = ROOT / "models" / "best.json"
    if algo is None:
        if not best.exists():
            raise SystemExit("No trained models found. Run: uv run python -m training.sweeps --algo all")
        return ROOT / json.load(open(best))["model"]
    summary = ROOT / "logs" / "summary.csv"
    import pandas as pd
    df = pd.read_csv(summary)
    df = df[df.algo == algo].sort_values("eval_return_bestckpt", ascending=False)
    return ROOT / df.iloc[0]["model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--export-json", help="save every frame's state as JSON for a web/Three.js front end")
    ap.add_argument("--fps", type=int, default=6)
    a = ap.parse_args()

    from environment.custom_env import ACTION_NAMES, NutriBridgeEnv
    from training.common import load_net

    path = pick_model(a.algo)
    net, ck = load_net(path)
    print(f"Loaded {ck['algo'].upper()} agent  [{ck['cfg'].get('name')}]  from {path.relative_to(ROOT)}\n")

    torch.manual_seed(a.seed)
    env = NutriBridgeEnv(render_mode=None if a.no_gui else "human")
    env.metadata["render_fps"] = a.fps
    frames = []
    for ep in range(a.episodes):
        obs, info = env.reset(seed=a.seed + ep)
        done, G, t = False, 0.0, 0
        print(f"=== Episode {ep + 1}  (seed {a.seed + ep}) ===")
        if not a.no_gui: env.render()
        while not done:
            with torch.no_grad():
                out = net(torch.as_tensor(obs))
                act = (int(torch.distributions.Categorical(logits=out).sample())
                       if ck.get("stochastic") else int(out.argmax()))
            obs, r, done, _, info = env.step(act)
            G += r; t += 1
            print(f"t={t:3d} wk={info['week'] + 1:2d} | {ACTION_NAMES[act]:<20} r={r:+.2f} | stock={info['stock']} "
                  f"severe={info['n_severe']} z>=-2: {info['frac_ok']:.0%} | {env.last_event}")
            if a.export_json: frames.append(env.export_state())
        print(f"--> Episode {ep + 1} return {G:+.2f} | children with z >= -2 at end: {info['frac_ok']:.0%}\n")
    if a.export_json:
        env.export_json(a.export_json, frames); print("Exported", len(frames), "frames to", a.export_json)
    env.close()


if __name__ == "__main__":
    main()
