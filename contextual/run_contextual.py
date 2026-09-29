from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contextual_bsi import ContextualParametricBSI
from .simulation import (
    collect_offline_data,
    context_sampler,
    default_params,
    make_policy_builder,
    make_reward_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one small contextual BSI experiment.")
    parser.add_argument("--env", choices=["linear_gaussian", "logistic_bernoulli"], default="linear_gaussian")
    parser.add_argument("--pi0", choices=["uniform", "contextual_epsilon", "contextual_ts"], default="uniform")
    parser.add_argument("--pi1", choices=["uniform", "contextual_epsilon", "contextual_ts"], default="contextual_ts")
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--T_offline", type=int, default=100)
    parser.add_argument("--inner_reps", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_actions", type=int, default=3)
    parser.add_argument("--context_dim", type=int, default=2)
    parser.add_argument("--context_var", type=float, default=1.0)
    parser.add_argument("--epsilon0", type=float, default=0.1)
    parser.add_argument("--epsilon1", type=float, default=0.1)
    parser.add_argument("--obs_sigma", type=float, default=1.0)
    parser.add_argument("--reward_sigma", type=float, default=1.0)
    parser.add_argument("--ts_prob_mc", type=int, default=500)
    parser.add_argument("--param_scenario", default="default")
    parser.add_argument("--policy_explore_untried", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--rep_idx", type=int, default=0)
    parser.add_argument("--save_path", type=Path, default=None)
    return parser


def to_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    return value


def main() -> None:
    args = build_parser().parse_args()
    true_params = default_params(
        args.env,
        args.n_actions,
        args.context_dim,
        args.seed,
        args.param_scenario,
    )
    adaptive_behavior = args.pi0 != "uniform"
    offline = collect_offline_data(args, true_params, args.rep_idx)
    reward_model = make_reward_model(args, adaptive_behavior=adaptive_behavior)
    bsi = ContextualParametricBSI(
        reward_model=reward_model,
        eval_policy_builder=make_policy_builder(args, args.pi1, args.epsilon1, true_params),
        context_sampler=context_sampler(args.context_dim, args.context_var),
        T=args.T,
        algo_seed=args.seed + 700000 + args.rep_idx,
        context_seed=args.seed + 800000 + args.rep_idx,
    )
    result = bsi.run(offline, alphas=[args.alpha], n_reps=args.inner_reps)
    lower = result.center - result.ci_width[args.alpha]
    upper = result.center + result.ci_width[args.alpha]
    proj_lower = result.center - result.proj_ci_width[args.alpha]
    proj_upper = result.center + result.proj_ci_width[args.alpha]

    payload = {
        "config": vars(args),
        "center": result.center,
        "center_se": result.center_se,
        "ci": [lower, upper],
        "proj_ci": [proj_lower, proj_upper],
        "ci_width": result.ci_width,
        "proj_ci_width": result.proj_ci_width,
        "se": result.se,
        "lambda_hat": result.lambda_hat,
        "gradient": result.gradient,
        "Sigma": result.Sigma,
    }
    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        with args.save_path.open("w") as f:
            json.dump(to_serializable(payload), f, indent=2)

    print(f"theta_hat: {result.center:.6f}")
    print(f"{100 * (1 - args.alpha):.0f}% BSI CI: ({lower:.6f}, {upper:.6f})")
    print(f"{100 * (1 - args.alpha):.0f}% BSI-Projection CI: ({proj_lower:.6f}, {proj_upper:.6f})")
    print(f"gradient norm: {np.linalg.norm(result.gradient):.6f}")
    if args.save_path is not None:
        print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
