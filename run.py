import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm.auto import tqdm

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from algorithms import (
    BatchExploreThenGreedy,
    EpsilonGreedy,
    ExploreThenCommit,
    TSBernoulli,
    TSNormal,
    UCB,
    UniformSampling,
)
from baselines import (
    compute_all_intervals,
    estimate_policy_average_value_mc,
    naive_t_test_interval,
    simulate_offline_dataset,
    validate_static_bandit_inputs,
)
from environments import BernoulliRewardEnv, BetaRewardEnv, NormalRewardEnv
from inference import (
    AdaptiveBernoulliBSI,
    AdaptiveNormalBSI,
    BernoulliBSI,
    NormalBSI,
    bandit_exp_runner,
    compute_arm_mean_std,
)


ADAPTIVE_LOGGING_POLICIES = {"ts_normal", "epsilon_greedy", "ts_bernoulli"}
BETA_VARIANCE_ESTIMATION_OPTIONS = ["hoeffding", "empirical_variance"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run BSI and OPE baseline experiments."
    )
    parser.add_argument("--env", type=str, choices=["normal", "bernoulli", "beta"], default="normal")
    parser.add_argument("--mus", type=float, nargs="+", required=False, default=None,
                        help="Arm means. Required for env=normal/bernoulli; derived from beta params for env=beta.")
    parser.add_argument("--beta_alphas", type=float, nargs="+", default=None,
                        help="Beta distribution alpha params (one per arm). Required for env=beta.")
    parser.add_argument("--beta_betas", type=float, nargs="+", default=None,
                        help="Beta distribution beta params (one per arm). Required for env=beta.")
    parser.add_argument("--sigmas", type=float, nargs="+", default=None)
    parser.add_argument(
        "--behavior_policy",
        type=float,
        nargs="+",
        default=None,
        help="Static logging policy probabilities. Mutually exclusive with --pi0.",
    )
    parser.add_argument(
        "--pi0",
        type=str,
        default=None,
        choices=["uniform", "etc", "batch_greedy", "ts_normal", "epsilon_greedy", "ts_bernoulli"],
        help="Logging policy. If omitted, a static --behavior_policy is used.",
    )
    parser.add_argument(
        "--pi1",
        type=str,
        default=None,
        choices=["uniform", "etc", "batch_greedy", "ucb", "epsilon_greedy", "ts_normal", "ts_bernoulli"],
        help="Target policy to evaluate. Defaults to Thompson Sampling matching the environment.",
    )
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--T_offline", type=int, default=100)
    parser.add_argument("--offline_reps", type=int, default=100)
    parser.add_argument("--n_rep", dest="offline_reps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05])
    parser.add_argument("--conf_levels", dest="alphas", type=float, nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--ucb_c", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--prior_mean", type=float, default=0.0)
    parser.add_argument("--prior_var", type=float, default=1.0)
    parser.add_argument("--obs_sigma", type=float, default=1.0)
    parser.add_argument("--prior_alpha", type=float, default=1.0)
    parser.add_argument("--prior_beta", type=float, default=1.0)
    parser.add_argument("--infer_reps", type=int, default=200)
    parser.add_argument("--estimate_sigma", action="store_true", default=False)
    # input variance estimation methods
    parser.add_argument( 
        "--var_estimation_beta",
        type=str,
        choices=BETA_VARIANCE_ESTIMATION_OPTIONS,
        default=None,
        help=(
            "Variance rule for beta-normal BSI inference only: "
            "hoeffding uses variance 1/4, i.e. sigma 0.5; "
            "empirical_variance treats variance as unknown and estimates it as in normal-normal. "
            "If omitted for env=beta, hoeffding is used."
        ),
    )
    parser.add_argument("--n_policy_value_mc", type=int, default=10000)
    parser.add_argument("--n_ts_value_mc", dest="n_policy_value_mc", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--n_eval_prob_mc", type=int, default=2000)
    parser.add_argument("--n_ts_prob_mc", dest="n_eval_prob_mc", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--algo_seed", type=int, default=2026)
    parser.add_argument("--seed", dest="algo_seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--table_seed", type=int, default=1013)
    parser.add_argument(
        "--baselines_only",
        action="store_true",
        default=False,
        help="Run only the OPE baselines and skip BSI.",
    )
    parser.add_argument(
        "--run_weighted_t_test",
        action="store_true",
        default=False,
        help="Include an IPW-style t-based interval built from weighted rewards.",
    )
    parser.add_argument(
        "--run_naive_t_test",
        action="store_true",
        default=False,
        help="Include a naive iid t-based interval built directly from raw rewards.",
    )
    parser.add_argument(
        "--run_cadr_rescaled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the previous variance-rescaled CADR interval as cadr_rescaled.",
    )
    parser.add_argument(
        "--dr_bootstrap_reps",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for the DR interval.",
    )
    parser.add_argument(
        "--weight_mode",
        type=str,
        choices=["one_step", "cumulative"],
        default=None,
        help="Single weighting rule used by off-policy baselines. Deprecated in favor of --weight_modes.",
    )
    parser.add_argument(
        "--weight_modes",
        type=str,
        nargs="+",
        choices=["one_step", "cumulative"],
        default=None,
        help="One or more weighting rules to evaluate on the same logged data.",
    )
    parser.add_argument("--cadr_min_samples", type=int, default=30)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--rep_idx", type=int, default=None,
                        help="Run only this 0-based replication index and save a partial result. Used with PBS array jobs.")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--tag", type=str, default="compare")
    parser.add_argument("--show_progress", action="store_true", default=False)
    parser.add_argument(
        "--save_logged_data",
        dest="save_logged_data",
        action="store_true",
        default=True,
        help="Save the offline logged trajectories for each replication (default: on).",
    )
    parser.add_argument(
        "--no-save_logged_data",
        dest="save_logged_data",
        action="store_false",
        help="Do not save the offline logged trajectories for each replication.",
    )
    return parser


def to_serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: to_serializable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_serializable(v) for v in x]
    return x


def contains(interval, theta_true):
    return float(interval[0] <= theta_true <= interval[1])


def beta_variance_filename_suffix(args):
    if getattr(args, "env", None) != "beta":
        return ""
    return f"_var-{args.var_estimation_beta}"


def wilson_95_band(p_hat, n, z=1.96):
    if n <= 0:
        return float(p_hat), float(p_hat)
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2.0 * n)) / denom
    radius = z * np.sqrt((p_hat * (1.0 - p_hat) + z**2 / (4.0 * n)) / n) / denom
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def mean_95_band(xs, z=1.96):
    xs = np.asarray(xs, dtype=float)
    mean_x = float(xs.mean())
    if xs.size < 2:
        return mean_x, mean_x
    se_x = float(xs.std(ddof=1) / np.sqrt(xs.size))
    half = z * se_x
    return float(mean_x - half), float(mean_x + half)


def enabled_weighted_baseline_methods(args_like):
    methods = ["elfcb", "ipw", "cadr", "dr"]
    if getattr(args_like, "run_cadr_rescaled", False):
        methods.append("cadr_rescaled")
    if getattr(args_like, "run_weighted_t_test", False):
        methods.append("weighted_t_test")
    return methods


def summarize_intervals(intervals, centers, theta_true, theta_true_se):
    valid_intervals = [
        (float(lo), float(hi))
        for lo, hi in intervals
        if np.isfinite(lo) and np.isfinite(hi)
    ]
    valid_centers = [
        float(center)
        for center in centers
        if np.isfinite(center)
    ]
    n_total = len(intervals)
    n_valid = len(valid_intervals)
    n_failed = n_total - n_valid

    if n_valid == 0:
        return {
            "theta_star": float(theta_true),
            "theta_star_se": float(theta_true_se),
            "theta_hat": float("nan"),
            "theta_hat_se": float("nan"),
            "coverage": float("nan"),
            "coverage_ci_95": [float("nan"), float("nan")],
            "coverage_ci_band": float("nan"),
            "mean_width": float("nan"),
            "mean_width_ci_95": [float("nan"), float("nan")],
            "mean_width_ci_band": float("nan"),
            "n_valid": 0,
            "n_failed": n_failed,
        }

    coverage_hits = np.asarray([contains(iv, theta_true) for iv in valid_intervals], dtype=float)
    widths = np.asarray([hi - lo for lo, hi in valid_intervals], dtype=float)
    centers_arr = np.asarray(valid_centers, dtype=float)

    coverage = float(coverage_hits.mean())
    coverage_lo, coverage_hi = wilson_95_band(coverage, coverage_hits.size)
    mean_interval_width = float(widths.mean())
    width_lo, width_hi = mean_95_band(widths)
    theta_hat = float(centers_arr.mean())
    theta_hat_se = float(centers_arr.std(ddof=1) / np.sqrt(centers_arr.size)) if centers_arr.size > 1 else 0.0

    return {
        "theta_star": float(theta_true),
        "theta_star_se": float(theta_true_se),
        "theta_hat": theta_hat,
        "theta_hat_se": theta_hat_se,
        "coverage": coverage,
        "coverage_ci_95": [coverage_lo, coverage_hi],
        "coverage_ci_band": float(max(coverage - coverage_lo, coverage_hi - coverage)),
        "mean_width": mean_interval_width,
        "mean_width_ci_95": [width_lo, width_hi],
        "mean_width_ci_band": float(max(mean_interval_width - width_lo, width_hi - mean_interval_width)),
        "n_valid": n_valid,
        "n_failed": n_failed,
    }


def finalize_args(args):
    if not hasattr(args, "run_cadr_rescaled"):
        args.run_cadr_rescaled = False
    if not hasattr(args, "dr_bootstrap_reps"):
        args.dr_bootstrap_reps = 1000
    if args.dr_bootstrap_reps < 2:
        raise ValueError("--dr_bootstrap_reps must be at least 2.")
    if args.pi0 is not None and args.behavior_policy is not None:
        raise ValueError("Use either --pi0 or --behavior_policy, not both.")
    if args.pi0 is None and args.behavior_policy is None:
        raise ValueError("Provide one of --pi0 or --behavior_policy.")
    if args.pi1 is None:
        args.pi1 = "ts_bernoulli" if args.env == "bernoulli" else "ts_normal"
    if args.env == "bernoulli" and args.pi0 in {"ts_normal", "epsilon_greedy"}:
        raise ValueError(
            f"Adaptive logging policy '{args.pi0}' is not supported with env='bernoulli'."
        )
    if args.env == "beta":
        if args.beta_alphas is None or args.beta_betas is None:
            raise ValueError("For env='beta', provide --beta_alphas and --beta_betas.")
        beta_alphas = np.asarray(args.beta_alphas, dtype=float)
        beta_betas = np.asarray(args.beta_betas, dtype=float)
        if beta_alphas.shape != beta_betas.shape:
            raise ValueError("beta_alphas and beta_betas must have the same length.")
        if np.any(beta_alphas <= 0) or np.any(beta_betas <= 0):
            raise ValueError("beta_alphas and beta_betas must be strictly positive.")
        mus = beta_alphas / (beta_alphas + beta_betas)
        sigmas = np.full(len(mus), 0.5) # start with 0.5 (hoeffding)
        args.beta_alphas = beta_alphas.tolist()
        args.beta_betas = beta_betas.tolist()
        args.estimate_sigma = args.var_estimation_beta == "empirical_variance"
        if args.var_estimation_beta == "empirical_variance":
            print(
                "Warning: --var_estimation_beta empirical_variance forces --estimate_sigma=True "
                "for beta-normal BSI inference.",
                file=sys.stderr,
            )
        elif "--estimate_sigma" in sys.argv:
            print(
                "Warning: --estimate_sigma is ignored for env='beta' when "
                "--var_estimation_beta is hoeffding; forcing --estimate_sigma=False.",
                file=sys.stderr,
            )
        args.obs_sigma = 0.5  # sub-Gaussian bound; used for DR reward model
        if args.behavior_policy is not None:
            behavior_policy = np.asarray(args.behavior_policy, dtype=float)
            if behavior_policy.shape[0] != mus.shape[0]:
                raise ValueError("behavior_policy length must match number of arms.")
            if np.any(behavior_policy <= 0.0):
                raise ValueError("behavior_policy must be strictly positive.")
            if not np.isclose(behavior_policy.sum(), 1.0, atol=1e-10):
                raise ValueError("behavior_policy must sum to 1.")
            args.behavior_policy = behavior_policy.tolist()
    elif args.behavior_policy is not None:
        if args.mus is None:
            raise ValueError("For env='normal'/'bernoulli', you must provide --mus.")
        mus, sigmas, behavior_policy = validate_static_bandit_inputs(
            args.mus,
            args.behavior_policy,
            env_type=args.env,
            reward_stds=args.sigmas,
        )
        args.behavior_policy = behavior_policy.tolist()
    else:
        if args.mus is None:
            raise ValueError("For env='normal'/'bernoulli', you must provide --mus.")
        mus = np.asarray(args.mus, dtype=float)
        if args.env == "bernoulli":
            sigmas = np.sqrt(np.clip(mus * (1.0 - mus), 0.0, None))
        else:
            if args.sigmas is None:
                raise ValueError("For env='normal', you must provide --sigmas.")
            sigmas = np.asarray(args.sigmas, dtype=float)
            if sigmas.shape[0] != mus.shape[0]:
                raise ValueError("mus and sigmas must have the same length.")

    args.mus = mus.tolist()
    args.sigmas = sigmas.tolist()
    if args.weight_modes is None:
        if args.weight_mode is not None:
            args.weight_modes = [args.weight_mode]
        else:
            args.weight_modes = ["one_step", "cumulative"]
    args.weight_modes = list(dict.fromkeys(args.weight_modes))
    return args


def make_env(args):
    if args.env == "beta":
        env = BetaRewardEnv(
            alpha_params=np.asarray(args.beta_alphas, dtype=float),
            beta_params=np.asarray(args.beta_betas, dtype=float),
        )
        env.sigmas = np.asarray(args.sigmas, dtype=float)
        return env
    mus = np.asarray(args.mus, dtype=float)
    if args.env == "bernoulli":
        return BernoulliRewardEnv(mus=mus)
    return NormalRewardEnv(mus=mus, sigma=np.asarray(args.sigmas, dtype=float))


def make_builder(policy_name, n_actions, args_dict):
    if policy_name == "uniform":
        def builder(seed):
            return UniformSampling(n_actions=n_actions, seed=seed)
    elif policy_name == "epsilon_greedy":
        def builder(seed):
            return EpsilonGreedy(n_actions=n_actions, epsilon=args_dict["epsilon"], seed=seed)
    elif policy_name == "etc":
        def builder(seed):
            return ExploreThenCommit(n_actions=n_actions, m=args_dict["m"], seed=seed)
    elif policy_name == "batch_greedy":
        def builder(seed):
            return BatchExploreThenGreedy(
                n_actions=n_actions,
                m=args_dict["m"],
                batch_size=args_dict["batch_size"],
                seed=seed,
            )
    elif policy_name == "ucb":
        def builder(seed):
            return UCB(n_actions=n_actions, c=args_dict["ucb_c"], seed=seed)
    elif policy_name == "ts_normal":
        def builder(seed):
            return TSNormal(
                n_actions=n_actions,
                seed=seed,
                prior_mean=args_dict["prior_mean"],
                prior_var=args_dict["prior_var"],
                obs_sigma=args_dict["obs_sigma"],
            )
    elif policy_name == "ts_bernoulli":
        def builder(seed):
            return TSBernoulli(
                n_actions=n_actions,
                seed=seed,
                alpha0=args_dict["prior_alpha"],
                beta0=args_dict["prior_beta"],
            )
    else:
        raise ValueError(f"Unsupported policy: {policy_name}")
    return builder


def run_theta_true_chunk(task):
    args_dict, start_idx, n_rollouts = task
    env = make_env(argparse.Namespace(**args_dict))
    pi1_builder = make_builder(args_dict["pi1"], env.n_actions, args_dict)
    result = bandit_exp_runner(
        env=env,
        algo_builder=pi1_builder,
        T=args_dict["T"],
        n_reps=n_rollouts,
        base_exp_seed=args_dict["algo_seed"] + 2 + start_idx,
        table_seed=args_dict["algo_seed"] + 2 + 7919 + start_idx,
        table_renew=True,
    )
    avg_rewards = np.asarray(result["all_rewards"], dtype=float).sum(axis=1) / args_dict["T"]
    return {
        "n": int(n_rollouts),
        "sum": float(avg_rewards.sum()),
        "sumsq": float(np.sum(avg_rewards * avg_rewards)),
    }


def combine_theta_true_chunks(chunk_results):
    n_total = sum(result["n"] for result in chunk_results)
    total = sum(result["sum"] for result in chunk_results)
    total_squares = sum(result["sumsq"] for result in chunk_results)
    theta_true = float(total / n_total)
    if n_total > 1:
        sample_var = (total_squares - n_total * theta_true * theta_true) / (n_total - 1)
        sample_var = max(float(sample_var), 0.0)
        theta_true_se = float(np.sqrt(sample_var / n_total))
    else:
        theta_true_se = 0.0
    return {
        "theta_star": theta_true,
        "theta_star_se": theta_true_se,
    }


def estimate_policy_average_value_mc_parallel(args, args_dict):
    n_rollouts = int(args.n_policy_value_mc)
    n_jobs = max(1, min(int(args.n_jobs), n_rollouts))
    if n_jobs == 1:
        env = make_env(args)
        pi1_builder = make_builder(args.pi1, env.n_actions, args_dict)
        return estimate_policy_average_value_mc(
            env=env,
            algo_builder=pi1_builder,
            t=args.T,
            n_rollouts=n_rollouts,
            seed=args.algo_seed + 2,
        )

    chunk_sizes = np.full(n_jobs, n_rollouts // n_jobs, dtype=int)
    chunk_sizes[: n_rollouts % n_jobs] += 1
    starts = np.concatenate(([0], np.cumsum(chunk_sizes)[:-1]))
    tasks = [
        (args_dict.copy(), int(start), int(size))
        for start, size in zip(starts, chunk_sizes)
        if size > 0
    ]
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            chunk_results = list(pool.map(run_theta_true_chunk, tasks))
    except PermissionError:
        print("Process-based theta_true parallelism unavailable in this environment; falling back to serial execution.")
        chunk_results = [run_theta_true_chunk(task) for task in tasks]
    return combine_theta_true_chunks(chunk_results)


def collect_offline_data(args_dict, rep_idx):
    env = make_env(argparse.Namespace(**args_dict))
    t_offline = int(args_dict["T_offline"])

    if args_dict["pi0"] is None:
        behavior_policy = np.asarray(args_dict["behavior_policy"], dtype=float)
        rng = np.random.default_rng(args_dict["algo_seed"] + 10000 * rep_idx + 17)
        if args_dict["env"] == "beta":
            # Sample actions from behavior policy, rewards from Beta distribution
            k = env.n_actions
            actions = rng.choice(k, size=t_offline, p=behavior_policy)
            rewards = rng.beta(
                env.alpha_params[actions], env.beta_params[actions]
            ).astype(np.float64)
        else:
            actions, rewards = simulate_offline_dataset(
                t_off=t_offline,
                reward_means=np.asarray(args_dict["mus"], dtype=float),
                reward_stds=np.asarray(args_dict["sigmas"], dtype=float),
                behavior_policy=behavior_policy,
                rng=rng,
                env_type=args_dict["env"],
            )
        behavior_probs = np.tile(behavior_policy, (t_offline, 1))
        offline_data = {
            "all_actions": actions.reshape(1, -1),
            "all_rewards": rewards.reshape(1, -1),
            "all_probs": behavior_probs.reshape(1, t_offline, -1),
            "T": t_offline,
        }
        return offline_data, behavior_probs, False

    pi0_builder = make_builder(args_dict["pi0"], env.n_actions, args_dict)
    offline_data = bandit_exp_runner(
        env=env,
        algo_builder=pi0_builder,
        T=t_offline,
        n_reps=1,
        base_exp_seed=args_dict["algo_seed"] + rep_idx,
        table_seed=args_dict["table_seed"] + rep_idx,
        table_renew=True,
        adaptive=True,
    )
    behavior_probs = np.asarray(offline_data["all_probs"][0], dtype=float)
    is_adaptive_pi0 = args_dict["pi0"] in ADAPTIVE_LOGGING_POLICIES
    return offline_data, behavior_probs, is_adaptive_pi0


def run_delta_method(args_dict, env, offline_data, is_adaptive_pi0, rep_idx):
    pi1_builder = make_builder(args_dict["pi1"], env.n_actions, args_dict)
    common_kwargs = dict(
        true_env=env,
        algo_builder1=lambda seed: UniformSampling(n_actions=env.n_actions, seed=seed),
        algo_builder2=pi1_builder,
        T=args_dict["T"],
        algo_seed=args_dict["algo_seed"] + rep_idx,
        table_seed=args_dict["table_seed"] + rep_idx,
    )

    if is_adaptive_pi0 and args_dict["env"] == "bernoulli":
        delta = AdaptiveBernoulliBSI(**common_kwargs)
    elif is_adaptive_pi0:
        delta = AdaptiveNormalBSI(
            **common_kwargs,
            estimate_sigma=args_dict["estimate_sigma"],
        )
    elif args_dict["env"] == "bernoulli":
        delta = BernoulliBSI(**common_kwargs)
    else:
        delta = NormalBSI(
            **common_kwargs,
            estimate_sigma=args_dict["estimate_sigma"],
        )

    delta_result = delta.run(
        offline_data=offline_data,
        alphas=args_dict["alphas"],
        n_reps=args_dict["infer_reps"],
    )
    alphas = np.asarray(args_dict["alphas"], dtype=float)
    bw     = {str(a): float(delta_result["ci_width_proj"][a]) for a in alphas}
    bw_adj = ({str(a): float(delta_result["ci_width"][a])    for a in alphas}
              if "ci_width" in delta_result else bw)

    return {
        "bsi_center": float(delta_result["center"]),
        "bsi_center_se": float(delta_result["center_se"]),
        "bsi_se": float(delta_result["se"]),
        "bsi_bandwidths":     bw,
        "bsi_bandwidths_adj": bw_adj,
    }


def build_rep_result(args_dict, rep_idx, offline_data, behavior_probs, is_adaptive_pi0, delta_summary=None):
    args_ns = argparse.Namespace(**args_dict)
    env = make_env(args_ns)
    alphas = np.asarray(args_dict["alphas"], dtype=float)
    weighted_methods = enabled_weighted_baseline_methods(args_ns)
    actions = np.asarray(offline_data["all_actions"][0], dtype=int)
    rewards = np.asarray(offline_data["all_rewards"][0], dtype=float)
    pi1_builder = make_builder(args_dict["pi1"], env.n_actions, args_dict)

    # For beta: use empirical arm std as obs_sigma for baselines (unknown-variance approach,
    # same as normal-normal). Fall back to 0.5 for arms with fewer than 2 observations.
    if args_dict["env"] == "beta":
        arm_dict_bl = compute_arm_mean_std(
            offline_data["all_actions"], offline_data["all_rewards"], env.n_actions
        )
        arm_std_bl = np.squeeze(np.asarray(arm_dict_bl["arm_std"], dtype=float))
        baseline_obs_sigma = np.where(
            np.isnan(arm_std_bl) | (arm_std_bl == 0.0), 0.5, arm_std_bl
        )
    else:
        baseline_obs_sigma = args_dict["obs_sigma"]

    interval_map = {}
    horizon = min(args_dict["T"], args_dict["T_offline"])
    for alpha in alphas:
        interval_map[str(alpha)] = {}
        for weight_mode in args_dict["weight_modes"]:
            try:
                interval_map[str(alpha)][weight_mode] = compute_all_intervals(
                    actions=actions[:horizon],
                    rewards=rewards[:horizon],
                    behavior_probs=behavior_probs[:horizon],
                    eval_algo_builder=pi1_builder,
                    conf_level=1.0 - float(alpha),
                    n_eval_prob_mc=args_dict["n_eval_prob_mc"],
                    eval_seed=args_dict["algo_seed"] + 20000 * rep_idx + int(1000 * alpha),
                    reward_model_prior_mean=args_dict["prior_mean"],
                    reward_model_prior_var=args_dict["prior_var"],
                    reward_model_obs_sigma=baseline_obs_sigma,
                    env_type="normal" if args_dict["env"] == "beta" else args_dict["env"],
                    reward_model_prior_alpha=args_dict["prior_alpha"],
                    reward_model_prior_beta=args_dict["prior_beta"],
                    weight_mode=weight_mode,
                    prob_eps=1e-8,
                    cadr_min_samples=args_dict["cadr_min_samples"],
                    include_weighted_t_test=args_dict["run_weighted_t_test"],
                    include_cadr_rescaled=args_dict["run_cadr_rescaled"],
                    dr_bootstrap_reps=args_dict["dr_bootstrap_reps"],
                    dr_bootstrap_seed=(
                        args_dict["algo_seed"]
                        + 30000 * rep_idx
                        + int(1000 * alpha)
                        + (0 if weight_mode == "one_step" else 1000003)
                    ),
                )
            except Exception:
                interval_map[str(alpha)][weight_mode] = argparse.Namespace(
                    elfcb=(float("nan"), float("nan")),
                    ipw=(float("nan"), float("nan")),
                    weighted_t_test=(float("nan"), float("nan"))
                    if args_dict["run_weighted_t_test"]
                    else None,
                    cadr=(float("nan"), float("nan")),
                    cadr_rescaled=(float("nan"), float("nan"))
                    if args_dict["run_cadr_rescaled"]
                    else None,
                    dr=(float("nan"), float("nan")),
                )

    rep_result = {
        "logging_policy": args_dict["pi0"] or "static",
        "is_adaptive_logging": bool(is_adaptive_pi0),
        "intervals": {
            conf: {
                weight_mode: {
                    method: getattr(interval_map[conf][weight_mode], method)
                    for method in weighted_methods
                }
                for weight_mode in interval_map[conf]
            }
            for conf in interval_map
        },
    }
    if args_dict.get("run_naive_t_test"):
        rep_result["naive_t_test"] = {
            str(alpha): naive_t_test_interval(rewards[:horizon], 1.0 - float(alpha))
            for alpha in alphas
        }
    if delta_summary is not None:
        rep_result.update(delta_summary)
    if args_dict.get("save_logged_data"):
        rep_result["logged_data"] = {
            "actions": actions.tolist(),
            "rewards": rewards.tolist(),
            "behavior_probs": behavior_probs.tolist(),
            "T_offline": int(offline_data["T"]),
        }
    return rep_result


def run_one_rep(task):
    rep_idx, args_dict = task
    args_ns = argparse.Namespace(**args_dict)
    env = make_env(args_ns)
    offline_data, behavior_probs, is_adaptive_pi0 = collect_offline_data(args_dict, rep_idx)
    delta_summary = None
    if not args_dict.get("baselines_only", False):
        delta_summary = run_delta_method(args_dict, env, offline_data, is_adaptive_pi0, rep_idx)
    return build_rep_result(
        args_dict,
        rep_idx,
        offline_data,
        behavior_probs,
        is_adaptive_pi0,
        delta_summary=delta_summary,
    )


def build_summary(args, per_rep, theta_true, theta_true_se):
    weighted_methods = enabled_weighted_baseline_methods(args)
    summary = {
        "config": vars(args),
        "theta_true": theta_true,
        "theta_true_se": theta_true_se,
        "var_estimation_beta": args.var_estimation_beta,
        "results_by_alpha": {},
        "per_rep": per_rep,
    }

    for alpha in args.alphas:
        key = str(alpha)
        baseline_intervals = {
            f"{method}_{weight_mode}": []
            for weight_mode in args.weight_modes
            for method in weighted_methods
        }
        baseline_centers = {
            f"{method}_{weight_mode}": []
            for weight_mode in args.weight_modes
            for method in weighted_methods
        }
        naive_t_test_intervals = []
        naive_t_test_centers = []

        is_adaptive = per_rep[0].get("is_adaptive_logging", False) if per_rep else False
        pfx = "adaptive" if is_adaptive else "bsi"
        if is_adaptive:
            bsi_variant_keys = [f"{pfx}_proj", pfx]
        else:
            bsi_variant_keys = [pfx]
        bsi_variant_intervals = {k: [] for k in bsi_variant_keys}
        bsi_variant_centers = {k: [] for k in bsi_variant_keys}

        for rep in per_rep:
            if not args.baselines_only:
                if is_adaptive:
                    bw_map = {
                        f"{pfx}_proj": rep["bsi_bandwidths"][key],
                        pfx:           rep["bsi_bandwidths_adj"][key],
                    }
                else:
                    bw_map = {pfx: rep["bsi_bandwidths"][key]}
                ctr_map = {k: rep["bsi_center"] for k in bw_map}
                for k, bw in bw_map.items():
                    ctr = ctr_map[k]
                    bsi_variant_intervals[k].append((ctr - bw, ctr + bw))
                    bsi_variant_centers[k].append(ctr)

            for weight_mode in args.weight_modes:
                for method in weighted_methods:
                    interval = tuple(rep["intervals"][key][weight_mode][method])
                    baseline_intervals[f"{method}_{weight_mode}"].append(interval)
                    baseline_centers[f"{method}_{weight_mode}"].append(
                        float((interval[0] + interval[1]) / 2.0)
                    )
            if args.run_naive_t_test:
                interval = tuple(rep["naive_t_test"][key])
                naive_t_test_intervals.append(interval)
                naive_t_test_centers.append(float((interval[0] + interval[1]) / 2.0))

        result_metrics = {}
        if not args.baselines_only:
            for k in bsi_variant_intervals:
                result_metrics[k] = summarize_intervals(
                    bsi_variant_intervals[k],
                    bsi_variant_centers[k],
                    theta_true,
                    theta_true_se,
                )
        for metric_key, intervals in baseline_intervals.items():
            result_metrics[metric_key] = summarize_intervals(
                intervals,
                baseline_centers[metric_key],
                theta_true,
                theta_true_se,
            )
        if args.run_naive_t_test:
            result_metrics["naive_t_test"] = summarize_intervals(
                naive_t_test_intervals,
                naive_t_test_centers,
                theta_true,
                theta_true_se,
            )
        summary["results_by_alpha"][key] = result_metrics
    return summary


def run_config(args, theta_true=None, theta_true_se=None, finalize=True):
    if finalize:
        args = finalize_args(args)
    args_dict = vars(args).copy()
    weighted_methods = enabled_weighted_baseline_methods(args)

    if theta_true is None or theta_true_se is None:
        theta_true_result = estimate_policy_average_value_mc_parallel(args, args_dict)
        theta_true = float(theta_true_result["theta_star"])
        theta_true_se = float(theta_true_result["theta_star_se"])
    # ── Array-job mode: run a single replication and save a partial file ─────
    if args.rep_idx is not None:
        rep_result = run_one_rep((args.rep_idx, args_dict))
        partial = {
            "config": vars(args),
            "theta_true": theta_true,
            "theta_true_se": theta_true_se,
            "var_estimation_beta": args.var_estimation_beta,
            "rep_idx": args.rep_idx,
            "rep_result": rep_result,
        }
        os.makedirs(args.save_dir, exist_ok=True)
        beta_var_suffix = beta_variance_filename_suffix(args)
        filename = (
            f"{args.tag}_{args.pi1}"
            f"{beta_var_suffix}"
            f"_T-{args.T}"
            f"_Toff-{args.T_offline}"
            f"_reps-{args.offline_reps}"
            f"_seed-{args.algo_seed}"
            f"_rep-{args.rep_idx}.json"
        )
        save_path = os.path.join(args.save_dir, filename)
        with open(save_path, "w") as f:
            json.dump(to_serializable(partial), f, indent=2)
        print(f"Rep {args.rep_idx}: saved to {save_path}")
        return

    tasks = [(i, args_dict.copy()) for i in range(args.offline_reps)]
    if args.n_jobs == 1:
        iterator = tasks
        if args.show_progress:
            iterator = tqdm(tasks, desc="Replications", leave=False)
        per_rep = [run_one_rep(task) for task in iterator]
    else:
        try:
            with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
                iterator = pool.map(run_one_rep, tasks)
                if args.show_progress:
                    iterator = tqdm(iterator, total=len(tasks), desc="Replications", leave=False)
                per_rep = list(iterator)
        except PermissionError:
            print("Process-based parallelism unavailable in this environment; falling back to serial execution.")
            iterator = tasks
            if args.show_progress:
                iterator = tqdm(tasks, desc="Replications", leave=False)
            per_rep = [run_one_rep(task) for task in iterator]

    summary = build_summary(args, per_rep, theta_true, theta_true_se)

    os.makedirs(args.save_dir, exist_ok=True)
    beta_var_suffix = beta_variance_filename_suffix(args)
    filename = (
        f"{args.tag}_{args.pi1}"
        f"{beta_var_suffix}"
        f"_T-{args.T}"
        f"_Toff-{args.T_offline}"
        f"_reps-{args.offline_reps}"
        f"_seed-{args.algo_seed}.json"
    )
    save_path = os.path.join(args.save_dir, filename)
    with open(save_path, "w") as f:
        json.dump(to_serializable(summary), f, indent=2)

    print(f"theta_true: {theta_true:.6f} (se: {theta_true_se:.6f})")
    for alpha, result in summary["results_by_alpha"].items():
        print(f"\nalpha={alpha}")
        ordered_methods = []
        if not args.baselines_only:
            ordered_methods.extend([k for k in result
                                     if k.startswith("bsi") or k.startswith("adaptive")])
        ordered_methods.extend(
            [
                f"{method}_{weight_mode}"
                for weight_mode in args.weight_modes
                for method in weighted_methods
            ]
        )
        if args.run_naive_t_test:
            ordered_methods.append("naive_t_test")
        for method in ordered_methods:
            stats = result[method]
            cov_lo, cov_hi = stats["coverage_ci_95"]
            width_lo, width_hi = stats["mean_width_ci_95"]
            print(
                f"  {method:16s} "
                f"theta_hat={stats['theta_hat']:.6f} "
                f"theta_hat_se={stats['theta_hat_se']:.6f} "
                f"coverage={stats['coverage']:.4f} +/- {stats['coverage_ci_band']:.4f} "
                f"(95% CI [{cov_lo:.4f}, {cov_hi:.4f}]) "
                f"width={stats['mean_width']:.6f} +/- {stats['mean_width_ci_band']:.6f} "
                f"(95% CI [{width_lo:.6f}, {width_hi:.6f}]) "
                f"valid={stats['n_valid']} failed={stats['n_failed']}"
            )
    print(f"\nSaved to {save_path}")


def main():
    raw_args = build_parser().parse_args()
    if raw_args.var_estimation_beta is None:
        raw_args.var_estimation_beta = "hoeffding"
    run_config(raw_args)


if __name__ == "__main__":
    main()
