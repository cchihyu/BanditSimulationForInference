from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from .contextual_bsi import contextual_bandit_exp_runner
from .simulation import (
    collect_offline_data,
    context_sampler,
    default_params,
    make_policy_builder,
    make_reward_model,
)


PAIR_MAP = {
    "uni_ts": ("uniform", "contextual_ts"),
    "ts_ts": ("contextual_ts", "contextual_ts"),
    "eps_eps": ("contextual_epsilon", "contextual_epsilon"),
}


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate the inner Monte Carlo size M needed for stable contextual BSI widths."
    )
    parser.add_argument("--envs", default="linear_gaussian,logistic_bernoulli")
    parser.add_argument("--pairs", default="uni_ts,ts_ts,eps_eps")
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--T_offline_grid", type=parse_int_list, default=parse_int_list("100,200,500,1000,2000,3000"))
    parser.add_argument("--context_dim", type=int, default=2)
    parser.add_argument("--n_actions", type=int, default=3)
    parser.add_argument("--epsilon_policy", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--rel_eps", type=float, default=0.1)
    parser.add_argument("--m0", type=int, default=1000)
    parser.add_argument("--B", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--ts_prob_mc", type=int, default=1000)
    parser.add_argument("--context_var", type=float, default=1.0)
    parser.add_argument("--param_scenario", default="default")
    parser.add_argument("--rep_idx", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("selected_inner_reps.csv"))
    return parser.parse_args()


def make_run_args(args: argparse.Namespace, env: str, pi0: str, pi1: str, toff: int) -> argparse.Namespace:
    return argparse.Namespace(
        env=env,
        n_actions=args.n_actions,
        context_dim=args.context_dim,
        T=args.T,
        T_offline=toff,
        outer_reps=1,
        rep_idx=args.rep_idx,
        inner_reps=args.m0,
        truth_reps=1,
        conf_level=0.95,
        pi0=pi0,
        pi1=pi1,
        epsilon0=args.epsilon_policy,
        epsilon1=args.epsilon_policy,
        obs_sigma=1.0,
        reward_sigma=1.0,
        ts_prob_mc=args.ts_prob_mc,
        param_scenario=args.param_scenario,
        context_var=args.context_var,
        policy_explore_untried=False,
        adaptive_behavior="auto",
        include_elfcb=False,
        truth_value=None,
        seed=args.seed,
        save_dir=Path("."),
        tag="m_needed",
    )


def per_trajectory_gradients(reward_model, sim_result: dict, lambda_hat: np.ndarray) -> np.ndarray:
    contexts = np.asarray(sim_result["all_contexts"], dtype=np.float64)
    actions = np.asarray(sim_result["all_actions"], dtype=np.int64)
    rewards = np.asarray(sim_result["all_rewards"], dtype=np.float64)
    n_reps, T = actions.shape
    n_actions = reward_model.n_actions
    p = contexts.shape[2] + 1
    lambda_hat = np.asarray(lambda_hat, dtype=np.float64)
    n_beta = n_actions * p
    beta = lambda_hat[:n_beta].reshape(n_actions, p)
    x_design = np.concatenate(
        [np.ones((*contexts.shape[:2], 1), dtype=np.float64), contexts],
        axis=2,
    )
    future_returns = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
    grads = np.zeros((n_reps, lambda_hat.shape[0]), dtype=np.float64)

    if reward_model.__class__.__name__ == "ContextualLinearGaussianRewardModel":
        sigma = (
            float(np.exp(lambda_hat[n_beta]))
            if lambda_hat.shape[0] == n_beta + 1
            else float(reward_model.sigma)
        )
        means = np.einsum("mtp,mtp->mt", x_design, beta[actions])
        residuals = rewards - means
        scale = residuals * future_returns / (T * sigma**2)
        if lambda_hat.shape[0] == n_beta + 1:
            eta_score = -1.0 + residuals**2 / (sigma**2)
            grads[:, n_beta] = np.sum(eta_score * future_returns / T, axis=1)
    else:
        logits = np.einsum("mtp,mtp->mt", x_design, beta[actions])
        probs = expit(np.clip(logits, -35.0, 35.0))
        scale = (rewards - probs) * future_returns / T

    for action in range(n_actions):
        mask = actions == action
        contrib = x_design * (scale * mask)[:, :, None]
        grads[:, action * p : (action + 1) * p] = contrib.sum(axis=1)
    return grads


def width_from_gradient(g: np.ndarray, Sigma: np.ndarray) -> float:
    return float(np.sqrt(max(float(g @ Sigma @ g), 0.0)))


def estimate_m_from_pilot(
    grads: np.ndarray,
    Sigma: np.ndarray,
    *,
    B: int,
    tau: float,
    rel_eps: float,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    m0 = grads.shape[0]
    g_bar = grads.mean(axis=0)
    pilot_width = width_from_gradient(g_bar, Sigma)
    widths = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, m0, size=m0)
        widths[b] = width_from_gradient(grads[idx].mean(axis=0), Sigma)
    mean_boot_width = float(widths.mean())
    var_boot_width = float(widths.var(ddof=1))
    denom = max(mean_boot_width**2, np.finfo(float).tiny)
    rho_hat = float(m0 * var_boot_width / denom)
    z = float(norm.ppf(1.0 - tau / 2.0))
    m_req = int(np.ceil((z**2) * rho_hat / (rel_eps**2)))
    achieved_delta_m0 = float(z * np.sqrt(rho_hat / m0))
    achieved_delta_20k = float(z * np.sqrt(rho_hat / 20000.0))
    return {
        "pilot_width_kernel": pilot_width,
        "boot_width_mean_kernel": mean_boot_width,
        "boot_width_sd_kernel": float(widths.std(ddof=1)),
        "rho_hat": rho_hat,
        "m_req": m_req,
        "m_star": max(m0, m_req),
        "achieved_delta_m0": achieved_delta_m0,
        "achieved_delta_20k": achieved_delta_20k,
    }


def main() -> None:
    args = parse_args()
    envs = [part.strip() for part in args.envs.split(",") if part.strip()]
    pairs = [part.strip() for part in args.pairs.split(",") if part.strip()]
    rows = []

    for env in envs:
        true_params = default_params(
            env,
            args.n_actions,
            args.context_dim,
            args.seed,
            args.param_scenario,
        )
        for pair in pairs:
            pi0, pi1 = PAIR_MAP[pair]
            for toff in args.T_offline_grid:
                print(f"env={env} pair={pair} T_offline={toff}", flush=True)
                run_args = make_run_args(args, env, pi0, pi1, toff)
                adaptive_behavior = pi0 != "uniform"
                offline = collect_offline_data(run_args, true_params, args.rep_idx)
                reward_model = make_reward_model(run_args, adaptive_behavior=adaptive_behavior)
                lambda_hat, Sigma = reward_model.fit(
                    contexts=offline["contexts"],
                    actions=offline["actions"],
                    rewards=offline["rewards"],
                    behavior_probs=offline["behavior_probs"],
                )
                sim = contextual_bandit_exp_runner(
                    reward_model=reward_model,
                    eval_policy_builder=make_policy_builder(run_args, pi1, args.epsilon_policy, true_params),
                    context_sampler=context_sampler(args.context_dim, args.context_var),
                    T=args.T,
                    n_reps=args.m0,
                    lambda_params=lambda_hat,
                    algo_seed=args.seed + 700000 + args.rep_idx + 17 * toff,
                    context_seed=args.seed + 800000 + args.rep_idx + 19 * toff,
                    table_renew=True,
                )
                grads = per_trajectory_gradients(reward_model, sim, lambda_hat)
                est = estimate_m_from_pilot(
                    grads,
                    Sigma,
                    B=args.B,
                    tau=args.tau,
                    rel_eps=args.rel_eps,
                    seed=args.seed + 31 * toff + 101 * len(rows),
                )
                row = {
                    "env": env,
                    "pair": pair,
                    "pi0": pi0,
                    "pi1": pi1,
                    "T": args.T,
                    "T_offline": toff,
                    "rep_idx": args.rep_idx,
                    "m0": args.m0,
                    "B": args.B,
                    "tau": args.tau,
                    "rel_eps": args.rel_eps,
                    "epsilon_policy": args.epsilon_policy,
                    "adaptive_behavior": adaptive_behavior,
                    **est,
                }
                rows.append(row)
                print(
                    f"  rho={est['rho_hat']:.4g} m_req={est['m_req']} "
                    f"delta@20k={est['achieved_delta_20k']:.4g}",
                    flush=True,
                )
                with args.out.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
