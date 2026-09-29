from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm

from environments import BernoulliRewardEnv, NormalRewardEnv
from inference import (
    bandit_exp_runner_vectorized,
    bernoulli_rollout_gradients,
    compute_arm_estimates_adaptive,
    compute_arm_mean_std,
    normal_rollout_gradients,
)
from run import collect_offline_data, finalize_args, make_builder, make_env

from contextual.contextual_bsi import contextual_bandit_exp_runner
from contextual.select_inner_reps import (
    PAIR_MAP,
    estimate_m_from_pilot,
    per_trajectory_gradients as contextual_per_trajectory_gradients,
)
from contextual.simulation import (
    collect_offline_data as collect_contextual_offline_data,
    context_sampler,
    default_params,
    make_policy_builder as make_contextual_policy_builder,
    make_reward_model,
)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str | None) -> list[float] | None:
    if text is None:
        return None
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pilot-bootstrap selector for the BSI inner Monte Carlo size M. "
            "Supports original MAB, sub-Gaussian MAB, and contextual BSI."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["mab", "subgaussian", "contextual"],
        required=True,
        help="Use 'mab' for normal/bernoulli MAB, 'subgaussian' for beta MAB, or 'contextual'.",
    )
    parser.add_argument("--env", default=None)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument(
        "--T_offline_grid",
        type=parse_int_list,
        default=parse_int_list("100,200,500,1000,2000,3000"),
    )
    parser.add_argument("--m0", type=int, default=1000, help="Pilot inner rollouts.")
    parser.add_argument("--B", type=int, default=1000, help="Bootstrap resamples.")
    parser.add_argument("--Mmax", type=int, default=10000, help="Upper budget for selected M.")
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument(
        "--rel_eps",
        type=float,
        default=0.05,
        help="Target relative half-width for the Monte Carlo error of the CI-width kernel.",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--rep_idx", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("selected_M.csv"))

    # Original MAB / sub-Gaussian MAB options.
    parser.add_argument("--mus", type=parse_float_list, default=None)
    parser.add_argument("--sigmas", type=parse_float_list, default=None)
    parser.add_argument("--beta_alphas", type=parse_float_list, default=None)
    parser.add_argument("--beta_betas", type=parse_float_list, default=None)
    parser.add_argument(
        "--var_estimation_beta",
        choices=["hoeffding", "empirical_variance"],
        default="hoeffding",
    )
    parser.add_argument(
        "--pi0",
        default=None,
        choices=[
            "uniform",
            "epsilon_greedy",
            "etc",
            "batch_greedy",
            "ts_normal",
            "ts_bernoulli",
        ],
    )
    parser.add_argument(
        "--pi1",
        default=None,
        choices=[
            "uniform",
            "epsilon_greedy",
            "etc",
            "batch_greedy",
            "ucb",
            "ts_normal",
            "ts_bernoulli",
        ],
    )
    parser.add_argument("--behavior_policy", type=parse_float_list, default=None)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--ucb_c", type=float, default=np.sqrt(2.0))
    parser.add_argument("--prior_mean", type=float, default=0.0)
    parser.add_argument("--prior_var", type=float, default=1.0)
    parser.add_argument("--obs_sigma", type=float, default=1.0)
    parser.add_argument("--prior_alpha", type=float, default=1.0)
    parser.add_argument("--prior_beta", type=float, default=1.0)
    parser.add_argument("--estimate_sigma", action="store_true", default=False)

    # Contextual options.
    parser.add_argument(
        "--contextual_envs",
        default=None,
        help="Comma-separated contextual envs. Defaults to --env if given, otherwise both.",
    )
    parser.add_argument("--pairs", default="uni_ts,ts_ts,eps_eps")
    parser.add_argument("--context_dim", type=int, default=2)
    parser.add_argument("--n_actions", type=int, default=3)
    parser.add_argument("--epsilon_policy", type=float, default=0.1)
    parser.add_argument("--ts_prob_mc", type=int, default=1000)
    parser.add_argument("--context_var", type=float, default=1.0)
    parser.add_argument("--param_scenario", default="default")
    parser.add_argument("--policy_explore_untried", action="store_true", default=False)
    return parser


def _as_run_args(args: argparse.Namespace, toff: int) -> argparse.Namespace:
    env = args.env
    if args.mode == "subgaussian":
        env = "beta"
    elif env is None:
        env = "normal"

    pi0 = args.pi0
    pi1 = args.pi1
    if pi0 is None and args.behavior_policy is None:
        pi0 = "uniform"

    ns = argparse.Namespace(
        env=env,
        mus=args.mus,
        beta_alphas=args.beta_alphas,
        beta_betas=args.beta_betas,
        sigmas=args.sigmas,
        behavior_policy=args.behavior_policy,
        pi0=pi0,
        pi1=pi1,
        T=args.T,
        T_offline=toff,
        offline_reps=1,
        infer_reps=args.m0,
        n_policy_value_mc=1,
        n_eval_prob_mc=1000,
        alphas=[0.1],
        var_estimation_beta=args.var_estimation_beta,
        estimate_sigma=args.estimate_sigma,
        obs_sigma=args.obs_sigma,
        epsilon=args.epsilon,
        m=args.m,
        batch_size=args.batch_size,
        ucb_c=args.ucb_c,
        prior_mean=args.prior_mean,
        prior_var=args.prior_var,
        prior_alpha=args.prior_alpha,
        prior_beta=args.prior_beta,
        algo_seed=args.seed,
        table_seed=args.seed + 137,
        n_jobs=1,
        run_cadr_rescaled=False,
        run_weighted_t_test=False,
        dr_bootstrap_reps=1000,
        weight_mode=None,
        weight_modes=["one_step"],
        cadr_min_samples=30,
        rep_idx=args.rep_idx,
        save_dir=".",
        tag="find_M",
        show_progress=False,
        save_logged_data=False,
        baselines_only=False,
        run_naive_t_test=False,
    )
    return finalize_args(ns)


def _fit_mab_simulator_and_covariance(
    run_args: argparse.Namespace,
    offline_data: dict,
    is_adaptive_pi0: bool,
) -> tuple[object, np.ndarray, np.ndarray, bool]:
    env = make_env(run_args)
    actions = np.asarray(offline_data["all_actions"][0], dtype=np.int64)
    rewards = np.asarray(offline_data["all_rewards"][0], dtype=np.float64)
    n_actions = env.n_actions

    if is_adaptive_pi0:
        probs = np.asarray(offline_data["all_probs"][0], dtype=np.float64)
        if run_args.env == "bernoulli":
            hat_mu, _, Sigma = compute_arm_estimates_adaptive(
                actions, rewards, probs, n_actions, env="Bernoulli"
            )
            hat_mu = np.clip(hat_mu, 1e-6, 1.0 - 1e-6)
            return BernoulliRewardEnv(mus=hat_mu), hat_mu, Sigma, False

        sigma_env = None if run_args.estimate_sigma else env.sigmas
        hat_mu, hat_sigma, Sigma = compute_arm_estimates_adaptive(
            actions,
            rewards,
            probs,
            n_actions,
            estimate_sigma=run_args.estimate_sigma,
            sigma_env=sigma_env,
            env="Gaussian",
        )
        hat_sigma2 = hat_sigma**2
        lambda_hat = (
            np.stack([hat_mu, hat_sigma2], axis=1).ravel()
            if run_args.estimate_sigma
            else hat_mu
        )
        return NormalRewardEnv(mus=hat_mu, sigma=hat_sigma), lambda_hat, Sigma, run_args.estimate_sigma

    arm_summary = compute_arm_mean_std(
        all_actions=offline_data["all_actions"],
        all_rewards=offline_data["all_rewards"],
        n_actions=n_actions,
    )
    hat_mu = np.asarray(arm_summary["arm_means"][0], dtype=np.float64)
    if np.any(np.isnan(hat_mu)):
        raise ValueError("Some arm was never selected in the pilot offline data.")

    counts = np.bincount(actions, minlength=n_actions).astype(np.float64)
    if np.any(counts == 0.0):
        raise ValueError("Some arm has zero count in the pilot offline data.")

    if run_args.env == "bernoulli":
        hat_mu = np.clip(hat_mu, 1e-6, 1.0 - 1e-6)
        Sigma = np.diag((offline_data["T"] / counts) * hat_mu * (1.0 - hat_mu))
        return BernoulliRewardEnv(mus=hat_mu), hat_mu, Sigma, False

    if run_args.estimate_sigma:
        hat_std = np.asarray(arm_summary["arm_std"][0], dtype=np.float64)
        hat_sigma2 = np.where(
            np.isnan(hat_std) | (hat_std == 0.0),
            env.sigmas**2,
            hat_std**2,
        )
    else:
        hat_sigma2 = env.sigmas**2

    if run_args.estimate_sigma:
        repeated_counts = counts.repeat(2)
        sigma_diag = (
            np.stack([hat_sigma2, 2.0 * hat_sigma2**2], axis=1).ravel()
            * offline_data["T"]
            / repeated_counts
        )
        lambda_hat = np.stack([hat_mu, hat_sigma2], axis=1).ravel()
    else:
        sigma_diag = hat_sigma2 * offline_data["T"] / counts
        lambda_hat = hat_mu
    Sigma = np.diag(sigma_diag)
    return NormalRewardEnv(mus=hat_mu, sigma=np.sqrt(hat_sigma2)), lambda_hat, Sigma, run_args.estimate_sigma


def run_mab_find_m(args: argparse.Namespace) -> list[dict]:
    rows = []
    for toff in args.T_offline_grid:
        run_args = _as_run_args(args, toff)
        args_dict = vars(run_args).copy()
        print(
            f"mode={args.mode} env={run_args.env} pi0={run_args.pi0 or 'static'} "
            f"pi1={run_args.pi1} T_offline={toff}",
            flush=True,
        )
        offline_data, _, is_adaptive_pi0 = collect_offline_data(args_dict, args.rep_idx)
        imagined_env, lambda_hat, Sigma, estimate_sigma = _fit_mab_simulator_and_covariance(
            run_args, offline_data, is_adaptive_pi0
        )
        pi1_builder = make_builder(run_args.pi1, imagined_env.n_actions, args_dict)
        sim = bandit_exp_runner_vectorized(
            env=imagined_env,
            algo_builder=pi1_builder,
            T=run_args.T,
            n_reps=args.m0,
            base_exp_seed=args.seed + 700000 + args.rep_idx + 17 * toff,
            table_seed=args.seed + 800000 + args.rep_idx + 19 * toff,
            table_renew=True,
        )

        sim_actions = np.asarray(sim["all_actions"], dtype=np.int64)
        sim_rewards = np.asarray(sim["all_rewards"], dtype=np.float64)
        if run_args.env == "bernoulli":
            grads = bernoulli_rollout_gradients(
                sim_actions,
                sim_rewards,
                np.asarray(lambda_hat, dtype=np.float64),
                average=False,
            )
        else:
            if estimate_sigma:
                lam = np.asarray(lambda_hat, dtype=np.float64).reshape(imagined_env.n_actions, 2)
                hat_mu = lam[:, 0]
                hat_sigma2 = lam[:, 1]
            else:
                hat_mu = np.asarray(lambda_hat, dtype=np.float64)
                hat_sigma2 = imagined_env.sigmas**2
            grads = normal_rollout_gradients(
                sim_actions,
                sim_rewards,
                hat_mu,
                hat_sigma2,
                estimate_sigma=estimate_sigma,
                average=False,
            )

        est = estimate_m_from_pilot(
            grads,
            Sigma,
            B=args.B,
            tau=args.tau,
            rel_eps=args.rel_eps,
            seed=args.seed + 31 * toff + 101 * len(rows),
        )
        m_raw = int(est["m_star"])
        m_selected = min(m_raw, int(args.Mmax))
        exceeds_budget = m_raw > int(args.Mmax)
        if exceeds_budget:
            warnings.warn(
                f"Selected M={m_raw} exceeds Mmax={args.Mmax}; using M={m_selected}.",
                RuntimeWarning,
            )
        row = {
            "mode": args.mode,
            "env": run_args.env,
            "pi0": run_args.pi0 or "static",
            "pi1": run_args.pi1,
            "T": run_args.T,
            "T_offline": toff,
            "rep_idx": args.rep_idx,
            "m0": args.m0,
            "B": args.B,
            "Mmax": args.Mmax,
            "m_selected": m_selected,
            "m_exceeds_budget": exceeds_budget,
            "tau": args.tau,
            "rel_eps": args.rel_eps,
            "estimate_sigma": estimate_sigma,
            "adaptive_behavior": bool(is_adaptive_pi0),
            **est,
        }
        rows.append(row)
        print(
            f"  M={row['m_star']} selected={row['m_selected']} rho={row['rho_hat']:.4g} "
            f"delta@m0={row['achieved_delta_m0']:.4g}",
            flush=True,
        )
    return rows


def make_contextual_run_args(
    args: argparse.Namespace,
    env: str,
    pi0: str,
    pi1: str,
    toff: int,
) -> argparse.Namespace:
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
        policy_explore_untried=args.policy_explore_untried,
        adaptive_behavior="auto",
        include_elfcb=False,
        truth_value=None,
        seed=args.seed,
        save_dir=Path("."),
        tag="find_M",
    )


def run_contextual_find_m(args: argparse.Namespace) -> list[dict]:
    env_text = args.contextual_envs or args.env or "linear_gaussian,logistic_bernoulli"
    envs = [part.strip() for part in env_text.split(",") if part.strip()]
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
            if pair not in PAIR_MAP:
                raise ValueError(f"Unknown contextual pair '{pair}'. Choices: {sorted(PAIR_MAP)}")
            pi0, pi1 = PAIR_MAP[pair]
            for toff in args.T_offline_grid:
                print(f"mode=contextual env={env} pair={pair} T_offline={toff}", flush=True)
                run_args = make_contextual_run_args(args, env, pi0, pi1, toff)
                adaptive_behavior = pi0 != "uniform"
                offline = collect_contextual_offline_data(run_args, true_params, args.rep_idx)
                reward_model = make_reward_model(run_args, adaptive_behavior=adaptive_behavior)
                lambda_hat, Sigma = reward_model.fit(
                    contexts=offline["contexts"],
                    actions=offline["actions"],
                    rewards=offline["rewards"],
                    behavior_probs=offline["behavior_probs"],
                )
                sim = contextual_bandit_exp_runner(
                    reward_model=reward_model,
                    eval_policy_builder=make_contextual_policy_builder(
                        run_args, pi1, args.epsilon_policy, true_params
                    ),
                    context_sampler=context_sampler(args.context_dim, args.context_var),
                    T=args.T,
                    n_reps=args.m0,
                    lambda_params=lambda_hat,
                    algo_seed=args.seed + 700000 + args.rep_idx + 17 * toff,
                    context_seed=args.seed + 800000 + args.rep_idx + 19 * toff,
                    table_renew=True,
                )
                grads = contextual_per_trajectory_gradients(reward_model, sim, lambda_hat)
                est = estimate_m_from_pilot(
                    grads,
                    Sigma,
                    B=args.B,
                    tau=args.tau,
                    rel_eps=args.rel_eps,
                    seed=args.seed + 31 * toff + 101 * len(rows),
                )
                m_raw = int(est["m_star"])
                m_selected = min(m_raw, int(args.Mmax))
                exceeds_budget = m_raw > int(args.Mmax)
                if exceeds_budget:
                    warnings.warn(
                        f"Selected M={m_raw} exceeds Mmax={args.Mmax}; using M={m_selected}.",
                        RuntimeWarning,
                    )
                row = {
                    "mode": "contextual",
                    "env": env,
                    "pair": pair,
                    "pi0": pi0,
                    "pi1": pi1,
                    "T": args.T,
                    "T_offline": toff,
                    "context_dim": args.context_dim,
                    "n_actions": args.n_actions,
                    "rep_idx": args.rep_idx,
                    "m0": args.m0,
                    "B": args.B,
                    "Mmax": args.Mmax,
                    "m_selected": m_selected,
                    "m_exceeds_budget": exceeds_budget,
                    "tau": args.tau,
                    "rel_eps": args.rel_eps,
                    "epsilon_policy": args.epsilon_policy,
                    "adaptive_behavior": adaptive_behavior,
                    **est,
                }
                rows.append(row)
                print(
                    f"  M={row['m_star']} selected={row['m_selected']} rho={row['rho_hat']:.4g} "
                    f"delta@m0={row['achieved_delta_m0']:.4g}",
                    flush=True,
                )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows were produced.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "contextual":
        rows = run_contextual_find_m(args)
    else:
        rows = run_mab_find_m(args)
    write_rows(args.out, rows)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
