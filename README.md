# NutriBridge CHW Planner: DQN vs REINFORCE vs PPO vs A2C (TorchRL)

**Student:** Tuyisingize Sezibera Gabriel · **Mission:** Health & Nutrition · **Capstone:** NutriBridge Community

A community health worker (CHW) supports 8 households with one child under 5 each, over a 24-week
season on a 7x7 village map. The agent must trade off walking time, a limited supplement stock,
slow-but-lasting nutrition education, and costly clinic referrals while a **lean season** (weeks 11-16)
lowers household food security. Child status uses WHO z-score thresholds (z < -2 moderate, z < -3 severe).

> The dynamics are **simulated** (parameters inspired by WHO standards, not fitted to community data).

## Run it (only `uv` is needed)

```bash
git clone <your-repo-url> && cd <repo>
uv sync
uv run main.py                         # best trained agent, pygame GUI + verbose terminal output
uv run main.py --algo ppo --episodes 5 # best saved model of one algorithm
uv run main.py --no-gui                # terminal only
uv run main.py --export-json assets/episode.json   # JSON frames for a web / Three.js / mobile front end
```

## Reproduce the experiments

```bash
uv run python -m training.sweeps --algo all --frames 300000 --workers 4   # 10 runs x 4 algorithms
uv run python -m training.plot_results                                    # figures -> assets/
uv run pytest                                                             # environment tests
```

Outputs: `logs/<algo>/run_XX.json` (curves), `logs/summary.csv` (one row per run, use for the report tables),
`logs/baselines.json`, `models/dqn|pg/*.pt`, `models/best.json` (used by `main.py`), figures in `assets/`.

## Environment summary

| | |
|---|---|
| Actions (14) | Travel to H0-H7, Travel to clinic, Screen child, Give supplement, Nutrition education, Refer to clinic, Restock supplies |
| Observation (38) | per household: last known z (norm.), weeks since screening, food security; global: x, y, stock, week, lean flag; one-hot location |
| Reward | + 3 x change in mean z, +1 per child back to z >= -2, -1 per child falling below -2, -2 per child falling into severe (z < -3), -0.02 per supplement, -0.02 to -0.1 for invalid or wasted actions, +5 x share of children with z >= -2 at week 24, -5 for early failure |
| Start | CHW at the clinic with full stock; z-scores and food security randomised per episode |
| Terminal | Week 24 reached, or early failure: >= 2 children severe for 3 consecutive weeks |

## Evaluation protocol

All algorithms use the same environment, same frame budget, same seed per configuration. Model selection uses
10 evaluation seeds (5000-5009); reported results use 30 held-out seeds (10000-10029). DQN is evaluated greedily;
the policy-gradient agents are evaluated by sampling from their learned policy (the standard setting; a
deterministic argmax of an on-policy agent can loop on a single action).

## Layout

```
main.py  pyproject.toml  uv.lock
environment/  custom_env.py (Gymnasium env + JSON export)  rendering.py (pygame)
training/     dqn_training.py  pg_training.py (REINFORCE, PPO, A2C)  sweeps.py  plot_results.py  baselines.py  common.py
models/ dqn/ pg/   logs/   assets/   tests/
```
