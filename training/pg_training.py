"""Policy-gradient methods with TorchRL: REINFORCE, PPO (ClipPPOLoss), A2C (A2CLoss)."""
import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE
from torchrl.objectives.value.functional import reward2go

from training.common import SyncDataCollector, N_ACTIONS, Tracker, make_env, make_mlp
from environment.custom_env import OBS_DIM

BASE = dict(seed=0, lr=1e-3, gamma=0.99, hidden=128, layers=2, entropy=0.01, critic_coeff=0.5,
            max_grad_norm=1.0, lmbda=0.95, n_envs=1)
DEFAULTS = {
    "reinforce": dict(BASE, frames_per_batch=2400, baseline=True, norm_adv=True),
    "ppo": dict(BASE, lr=3e-4, frames_per_batch=2048, clip=0.2, epochs=4, minibatch=128, critic_coeff=1.0),
    "a2c": dict(BASE, lr=7e-4, n_envs=8, frames_per_batch=128),
}


def build(cfg):
    actor_net = make_mlp(N_ACTIONS, cfg["hidden"], cfg["layers"])
    critic_net = make_mlp(1, cfg["hidden"], cfg["layers"])
    env = make_env(cfg["n_envs"], cfg["seed"])
    policy = ProbabilisticActor(TensorDictModule(actor_net, ["observation"], ["logits"]),
                                in_keys=["logits"], spec=env.action_spec,
                                distribution_class=Categorical, return_log_prob=True)
    critic = ValueOperator(critic_net, in_keys=["observation"])
    return env, actor_net, policy, critic


def run_pg(algo: str, run_id: int, cfg: dict, total_frames: int) -> dict:
    cfg = {**DEFAULTS[algo], **cfg, "total_frames": total_frames}
    torch.manual_seed(cfg["seed"])
    env, actor_net, policy, critic = build(cfg)
    collector = SyncDataCollector(env, policy, frames_per_batch=cfg["frames_per_batch"],
                                  total_frames=total_frames)
    params = list(policy.parameters()) + list(critic.parameters())
    optim = torch.optim.Adam(params, lr=cfg["lr"])
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic)

    if algo == "ppo":
        loss_mod = ClipPPOLoss(policy, critic, clip_epsilon=cfg["clip"], entropy_bonus=True,
                               entropy_coeff=cfg["entropy"], critic_coeff=cfg["critic_coeff"],
                               loss_critic_type="smooth_l1", normalize_advantage=True)
    elif algo == "a2c":
        loss_mod = A2CLoss(policy, critic, entropy_bonus=True, entropy_coeff=cfg["entropy"],
                           critic_coeff=cfg["critic_coeff"], loss_critic_type="smooth_l1")
    else:
        loss_mod = ReinforceLoss(policy, critic, loss_critic_type="smooth_l1")
        gae = GAE(gamma=cfg["gamma"], lmbda=1.0, value_network=critic)   # lambda=1: Monte-Carlo return - V(s)

    tracker = Tracker(actor_net, total_frames, stochastic=True)

    def step(td):
        out = loss_mod(td)
        if algo == "reinforce":
            ent = policy.get_dist(td).entropy().mean()
            total = out["loss_actor"] - cfg["entropy"] * ent
            if cfg["baseline"]:
                total = total + cfg["critic_coeff"] * out["loss_value"]
        else:
            ent = out["entropy"].detach() if "entropy" in out.keys() else policy.get_dist(td).entropy().mean().detach()
            total = sum(v for k, v in out.items() if k.startswith("loss_"))
        optim.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
        optim.step()
        return total.item(), float(ent.detach())

    for batch in collector:
        n = batch.numel()
        with torch.no_grad():
            if algo == "reinforce" and not cfg["baseline"]:
                ret = reward2go(batch["next", "reward"], batch["next", "done"], cfg["gamma"])
                batch.set("advantage", ret); batch.set("value_target", ret)
            else:
                gae(batch)
            if algo == "reinforce" and cfg["norm_adv"]:
                a = batch["advantage"]
                batch.set("advantage", (a - a.mean()) / (a.std() + 1e-8))
        data = batch.reshape(-1)
        losses, ents = [], []
        if algo == "ppo":
            for _ in range(cfg["epochs"]):
                perm = torch.randperm(data.shape[0])
                for i in range(0, data.shape[0], cfg["minibatch"]):
                    l, e = step(data[perm[i:i + cfg["minibatch"]]])
                    losses.append(l); ents.append(e)
        else:
            l, e = step(batch if algo == "a2c" else data)
            losses.append(l); ents.append(e)
        tracker.add_batch(batch, n, loss=sum(losses) / len(losses), entropy=sum(ents) / len(ents))
    collector.shutdown()
    return tracker.finish(algo, run_id, cfg)
