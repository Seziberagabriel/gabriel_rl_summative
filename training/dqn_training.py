"""DQN (value-based) with TorchRL: EGreedyModule + ReplayBuffer + DQNLoss + target updates."""
import torch
from tensordict.nn import TensorDictSequential
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from training.common import SyncDataCollector, N_ACTIONS, Tracker, make_env, make_mlp

DEFAULTS = dict(seed=0, lr=1e-3, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.5,
                buffer=50_000, batch=64, target="soft", tau=0.01, hard_interval=500,
                double=False, hidden=128, layers=2, frames_per_batch=100, utd=0.5,
                warmup=1000, grad_clip=10.0)


def run_dqn(run_id: int, cfg: dict, total_frames: int) -> dict:
    cfg = {**DEFAULTS, **cfg, "total_frames": total_frames}
    torch.manual_seed(cfg["seed"])
    env = make_env(1, cfg["seed"])
    net = make_mlp(N_ACTIONS, cfg["hidden"], cfg["layers"])
    qnet = QValueActor(net, in_keys=["observation"], spec=env.action_spec)
    explore = EGreedyModule(env.action_spec, eps_init=cfg["eps_start"], eps_end=cfg["eps_end"],
                            annealing_num_steps=int(cfg["eps_decay_frac"] * total_frames))
    policy = TensorDictSequential(qnet, explore)
    collector = SyncDataCollector(env, policy, frames_per_batch=cfg["frames_per_batch"],
                                  total_frames=total_frames, init_random_frames=cfg["warmup"])
    loss_mod = DQNLoss(qnet, action_space=env.action_spec, loss_function="smooth_l1",
                       delay_value=True, double_dqn=cfg["double"])
    loss_mod.make_value_estimator(gamma=cfg["gamma"])
    updater = (SoftUpdate(loss_mod, tau=cfg["tau"]) if cfg["target"] == "soft"
               else HardUpdate(loss_mod, value_network_update_interval=cfg["hard_interval"]))
    optim = torch.optim.Adam(loss_mod.parameters(), lr=cfg["lr"])
    rb = ReplayBuffer(storage=LazyTensorStorage(cfg["buffer"]), batch_size=cfg["batch"])
    tracker = Tracker(net, total_frames)
    n_updates = max(1, int(cfg["frames_per_batch"] * cfg["utd"]))

    for batch in collector:
        n = batch.numel()
        rb.extend(batch.reshape(-1))
        explore.step(n)
        losses = []
        if tracker.frames + n >= cfg["warmup"]:
            for _ in range(n_updates):
                td = rb.sample()
                loss = loss_mod(td)["loss"]
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), cfg["grad_clip"])
                optim.step(); updater.step()
                losses.append(loss.item())
        tracker.add_batch(batch, n, loss=sum(losses) / len(losses) if losses else None)
    collector.shutdown()
    return tracker.finish("dqn", run_id, cfg)
