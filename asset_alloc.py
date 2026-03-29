"""
MSBD5021 Assignment 1: Multi-Asset Allocation with REINFORCE
Extends Section 8.4 to n risky assets + cash with 10% rebalancing constraint.

Usage:
    uv run python asset_alloc.py configs/n3_T5.json
    uv run python asset_alloc.py configs/n4_T9.json
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

@dataclass
class Config:
    # Problem parameters
    n_risky: int
    means: list[float]
    variances: list[float]
    r: float
    a: float
    T: int
    init_wealth: float
    init_proportions: list[float]
    max_adjustment: float

    # Training hyperparameters
    lr: float = 3e-4
    num_episodes: int = 10000
    batch_size: int = 32
    hidden_size: int = 64
    print_every: int = 1000
    eval_episodes: int = 1000

    @property
    def stdevs(self) -> list[float]:
        return [math.sqrt(v) for v in self.variances]

    @staticmethod
    def from_json(path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return Config(**d)


# ──────────────────────────────────────────────
# Environment (MDP)
# ──────────────────────────────────────────────

class MultiAssetEnv:
    """
    MDP for multi-asset allocation with rebalancing constraint.

    State: [t/T, W_t, p_0, p_1, ..., p_n]
    Action: [Δp_1, ..., Δp_n] adjustments to risky asset proportions
    Transition: W_{t+1} = W_t * Σ_k p'_k * (1 + Y_k)
    Reward: 0 for t < T;  U(W_T) = -(1/a)*exp(-a*W_T) at terminal
    Constraint: |Δp_k| ≤ max_adjustment for all k (including cash)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n = cfg.n_risky
        self.state_dim = 2 + self.n + 1
        self.action_dim = self.n
        self._stdevs = np.array(cfg.stdevs)
        self._means = np.array(cfg.means)

    def reset(self) -> np.ndarray:
        self.t = 0
        self.wealth = self.cfg.init_wealth
        self.proportions = np.array(self.cfg.init_proportions, dtype=np.float64)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.concatenate([
            [self.t / self.cfg.T, self.wealth],
            self.proportions,
        ])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        cfg = self.cfg
        max_adj = cfg.max_adjustment

        delta_risky = np.clip(action, -max_adj, max_adj)
        delta_cash = -delta_risky.sum()

        if abs(delta_cash) > max_adj:
            scale = max_adj / abs(delta_cash)
            delta_risky *= scale
            delta_cash = -delta_risky.sum()

        new_props = self.proportions.copy()
        new_props[0] += delta_cash
        new_props[1:] += delta_risky

        new_props = np.maximum(new_props, 0.0)
        total = new_props.sum()
        if total > 0:
            new_props /= total

        returns_risky = np.random.normal(self._means, self._stdevs)

        growth = new_props[0] * (1 + cfg.r) + \
                 (new_props[1:] * (1 + returns_risky)).sum()
        new_wealth = self.wealth * growth

        if new_wealth > 0:
            natural_props = np.empty_like(new_props)
            natural_props[0] = new_props[0] * (1 + cfg.r) / growth
            natural_props[1:] = new_props[1:] * (1 + returns_risky) / growth
        else:
            natural_props = new_props

        self.t += 1
        self.wealth = new_wealth
        self.proportions = natural_props

        done = (self.t >= cfg.T)
        reward = -math.exp(-cfg.a * new_wealth) / cfg.a if done else 0.0

        return self._get_state(), reward, done


# ──────────────────────────────────────────────
# Policy Network
# ──────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int,
                 max_adj: float, policy_stdev: float,
                 learn_policy_stdev: bool = False,
                 min_policy_stdev: float = 1e-3,
                 max_policy_stdev: float = 0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )
        self.max_adj = max_adj
        self.learn_policy_stdev = learn_policy_stdev
        self.fixed_policy_stdev = policy_stdev
        self.min_policy_stdev = min_policy_stdev
        self.max_policy_stdev = max_policy_stdev

        if learn_policy_stdev:
            self.sigma_head = nn.Linear(hidden_size, action_dim)
            nn.init.zeros_(self.sigma_head.weight)
            nn.init.constant_(
                self.sigma_head.bias,
                self._initial_sigma_logit(policy_stdev),
            )

    def _initial_sigma_logit(self, init_sigma: float) -> float:
        span = self.max_policy_stdev - self.min_policy_stdev
        clipped = min(max(init_sigma, self.min_policy_stdev), self.max_policy_stdev)
        ratio = (clipped - self.min_policy_stdev) / span if span > 0 else 0.5
        ratio = min(max(ratio, 1e-4), 1 - 1e-4)
        return math.log(ratio / (1 - ratio))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        mu = self.mu_head(features) * self.max_adj

        if self.learn_policy_stdev:
            sigma_span = self.max_policy_stdev - self.min_policy_stdev
            raw_sigma = self.sigma_head(features)
            sigma = self.min_policy_stdev + sigma_span * torch.sigmoid(raw_sigma)
        else:
            sigma = torch.full_like(mu, self.fixed_policy_stdev)

        return mu, sigma

    def set_fixed_policy_stdev(self, policy_stdev: float) -> None:
        self.fixed_policy_stdev = policy_stdev


# ──────────────────────────────────────────────
# REINFORCE Training
# ──────────────────────────────────────────────

def get_manual_sigma(update: int, num_updates: int, policy_sigma: float, min_policy_sigma: float) -> float:
    if num_updates <= 1:
        return min_policy_sigma
    progress = (update - 1) / (num_updates - 1)
    return policy_sigma + progress * (min_policy_sigma - policy_sigma)

def train(cfg: Config, policy_sigma_type: str, policy_sigma: float, min_policy_sigma: float, max_policy_sigma: float) -> tuple[PolicyNetwork, list[float], list[float]]:
    env = MultiAssetEnv(cfg)
    learn_policy_stdev = (policy_sigma_type == "learn")
    manual_sigma_decay = (policy_sigma_type == "manual")
    policy = PolicyNetwork(
        env.state_dim, env.action_dim, cfg.hidden_size,
        cfg.max_adjustment, policy_sigma,
        learn_policy_stdev, min_policy_sigma, max_policy_sigma
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    num_updates = cfg.num_episodes // cfg.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_updates, eta_min=cfg.lr * 0.01
    )

    baseline = 0.0
    utilities: list[float] = []
    batch_sigma_means: list[float] = []

    for update in range(1, num_updates + 1):
        if manual_sigma_decay:
            policy.set_fixed_policy_stdev(get_manual_sigma(update, num_updates, policy_sigma, min_policy_sigma))

        batch_log_probs: list[list[torch.Tensor]] = []
        batch_returns: list[float] = []
        sigma_samples: list[float] = []

        for _ in range(cfg.batch_size):
            state = env.reset()
            log_probs: list[torch.Tensor] = []

            done = False
            while not done:
                s_tensor = torch.tensor(state, dtype=torch.float32)
                mu, sigma = policy(s_tensor)
                sigma_samples.append(float(sigma.mean().item()))
                dist = Independent(Normal(mu, sigma), 1)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                state, reward, done = env.step(action.detach().numpy())

            batch_log_probs.append(log_probs)
            batch_returns.append(reward)

        utilities.extend(batch_returns)
        batch_sigma_means.append(sum(sigma_samples) / len(sigma_samples))
        batch_mean = sum(batch_returns) / cfg.batch_size
        baseline = baseline * 0.99 + batch_mean * 0.01

        loss = torch.tensor(0.0)
        for lps, G in zip(batch_log_probs, batch_returns):
            advantage = G - baseline
            for lp in lps:
                loss = loss - lp * advantage
        loss = loss / cfg.batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        episodes_done = update * cfg.batch_size
        if episodes_done % cfg.print_every == 0:
            recent = utilities[-cfg.print_every:]
            avg_u = sum(recent) / len(recent)
            lr_now = optimizer.param_groups[0]['lr']
            sigma_now = batch_sigma_means[-1]
            print(
                f"  Episode {episodes_done:6d} | Avg Utility: {avg_u:.4f} | "
                f"Avg Sigma: {sigma_now:.4f} | LR: {lr_now:.2e}"
            )

    return policy, utilities, batch_sigma_means


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate(policy: PolicyNetwork, cfg: Config) -> tuple[float, float]:
    env = MultiAssetEnv(cfg)
    wealth_records = []
    prop_records: list[list[np.ndarray]] = [[] for _ in range(cfg.T + 1)]
    action_records: list[list[np.ndarray]] = [[] for _ in range(cfg.T)]

    for _ in range(cfg.eval_episodes):
        state = env.reset()
        prop_records[0].append(env.proportions.copy())
        for t in range(cfg.T):
            s_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                mu, _ = policy(s_tensor)
            action = mu.numpy()
            action_records[t].append(action.copy())
            state, _, _ = env.step(action)
            prop_records[t + 1].append(env.proportions.copy())
        wealth_records.append(env.wealth)

    avg_wealth = float(np.mean(wealth_records))
    avg_utility = float(np.mean(
        [-math.exp(-cfg.a * w) / cfg.a for w in wealth_records]
    ))

    # Print table header
    print(f"\n  {'Time':>4s} | {'Cash':>7s}", end="")
    for k in range(cfg.n_risky):
        print(f" | {'Asset'+str(k+1):>7s}", end="")
    print(f" | {'Δ Cash':>7s}", end="")
    for k in range(cfg.n_risky):
        print(f" | {'Δp_'+str(k+1):>7s}", end="")
    print()
    print("  " + "-" * (16 + 10 * cfg.n_risky * 2 + 10))

    for t in range(cfg.T):
        props = np.array(prop_records[t]).mean(axis=0)
        acts = np.array(action_records[t]).mean(axis=0)
        delta_cash = -acts.sum()
        print(f"  {t:>4d} |", end="")
        print(f" {props[0]:>7.1%}", end="")
        for k in range(cfg.n_risky):
            print(f" | {props[k+1]:>7.1%}", end="")
        print(f" | {delta_cash:>+7.4f}", end="")
        for k in range(cfg.n_risky):
            print(f" | {acts[k]:>+7.4f}", end="")
        print()

    props_T = np.array(prop_records[cfg.T]).mean(axis=0)
    print(f"  {cfg.T:>4d} |", end="")
    print(f" {props_T[0]:>7.1%}", end="")
    for k in range(cfg.n_risky):
        print(f" | {props_T[k+1]:>7.1%}", end="")
    print(f" |  (terminal)")

    print(f"\n  Avg Terminal Wealth:  {avg_wealth:.4f}")
    print(f"  Avg Terminal Utility: {avg_utility:.4f}")

    return avg_wealth, avg_utility


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_training(utilities: list[float], batch_sigma_means: list[float], cfg: Config,
                  filename: str = "training_curve.png",
                  window: int = 500) -> None:
    smoothed = []
    for i in range(len(utilities)):
        start = max(0, i - window + 1)
        smoothed.append(sum(utilities[start:i + 1]) / (i - start + 1))

    updates = np.arange(1, len(batch_sigma_means) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    axes[0].plot(smoothed)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Utility (smoothed)")
    axes[0].set_title(f"REINFORCE Training: n={cfg.n_risky} assets, T={cfg.T} periods")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(updates, batch_sigma_means)
    axes[1].set_xlabel("Batch Update")
    axes[1].set_ylabel("Mean Sigma")

    sigma_title = "Mean sampled sigma per batch"
    # 判断 sigma 类型
    import inspect
    frame = inspect.currentframe()
    outer = frame.f_back
    args = outer.f_locals.get('args', None)
    sigma_type = getattr(args, 'policy_sigma_type', None) if args else None
    if sigma_type is not None:
        if sigma_type == "fixed":
            sigma_title += " (fixed policy sigma)"
        elif sigma_type == "manual":
            sigma_title += " (manual decay)"
        elif sigma_type == "learn":
            sigma_title += " (learned sigma)"
    axes[1].set_title(sigma_title)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Training curve saved to {filename}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Asset Allocation with REINFORCE (Policy Gradient)")
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("-o", "--output", default="training_curve.png",
                        help="Output filename for training curve (default: training_curve.png)")
    parser.add_argument("--policy-sigma-type", choices=["fixed", "manual", "learn"], default="learn",
                        help="Policy sigma type: fixed, manual, or learn (default: learn)")
    parser.add_argument("--policy-sigma", type=float, default=0.08,
                        help="Initial policy sigma (for fixed/manual, default: 0.08)")
    parser.add_argument("--min-policy-sigma", type=float, default=0.01,
                        help="Minimum policy sigma (for manual/learn, default: 0.01)")
    parser.add_argument("--max-policy-sigma", type=float, default=0.2,
                        help="Maximum policy sigma (for manual/learn, default: 0.2)")
    args = parser.parse_args()

    cfg = Config.from_json(args.config)

    print("=" * 60)
    print("Multi-Asset Allocation with REINFORCE")
    print("=" * 60)
    print(f"  Config:           {args.config}")
    print(f"  Risky assets:     {cfg.n_risky}")
    print(f"  Time horizon:     {cfg.T}")
    print(f"  Means a(k):       {cfg.means}")
    print(f"  Variances s(k):   {cfg.variances}")
    print(f"  Stdevs σ(k):      {[f'{s:.4f}' for s in cfg.stdevs]}")
    print(f"  Riskless rate r:  {cfg.r}")
    print(f"  CARA coeff a:     {cfg.a}")
    print(f"  Init proportions: {cfg.init_proportions}")
    print(f"  Max adjustment:   {cfg.max_adjustment}")
    print(f"  Policy sigma type:{args.policy_sigma_type}")
    if args.policy_sigma_type == "learn":
        print(f"  Sigma range:      [{args.min_policy_sigma}, {args.max_policy_sigma}]")
    elif args.policy_sigma_type == "manual":
        print(f"  Sigma schedule:   linear decay {args.policy_sigma} -> {args.min_policy_sigma}")
    else:
        print(f"  Fixed sigma:      {args.policy_sigma}")
    print(f"  Episodes:         {cfg.num_episodes}")
    print()

    print("Training...")
    policy, utilities, batch_sigma_means = train(
        cfg,
        args.policy_sigma_type,
        args.policy_sigma,
        args.min_policy_sigma,
        args.max_policy_sigma
    )

    print("\nEvaluation (deterministic policy):")
    evaluate(policy, cfg)

    plot_training(utilities, batch_sigma_means, cfg, filename=args.output)
    print("=" * 60)


if __name__ == "__main__":
    main()
