from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.special import expit
from scipy.stats import chi2, norm

from .algorithms import ContextualEpsilonGreedyPolicy, ContextualTSPolicy
from .environments import ContextualLinearGaussianRewardModel, ContextualLogisticBernoulliRewardModel


@dataclass
class ContextualBSIResult:
    center: float
    center_se: float
    ci_width: dict[float, float]
    proj_ci_width: dict[float, float]
    se: float
    lambda_hat: np.ndarray
    gradient: np.ndarray
    Sigma: np.ndarray
    pi1_img: dict[str, np.ndarray | float | int]


class ContextualParametricBSI:
    """
    Simulation-based inference for contextual bandits.

    This implements the correctly specified parametric BSI extension:
        theta_hat = f_T(lambda_hat, pi1)
        CI half-width = z / sqrt(T_offline) * sqrt(g^T Sigma g)
    where g is estimated by Monte Carlo using the score/future-return identity.
    """

    def __init__(
        self,
        reward_model: Any,
        eval_policy_builder: Callable[[int], Any],
        context_sampler: Callable[[np.random.Generator, int], np.ndarray],
        T: int,
        algo_seed: int = 2026,
        context_seed: int = 1013,
    ):
        if T <= 0:
            raise ValueError("T must be positive.")
        self.reward_model = reward_model
        self.eval_policy_builder = eval_policy_builder
        self.context_sampler = context_sampler
        self.T = int(T)
        self.algo_seed = int(algo_seed)
        self.context_seed = int(context_seed)

    def run(
        self,
        offline_data: dict[str, np.ndarray | int],
        alphas: list[float] | np.ndarray,
        n_reps: int = 500,
    ) -> ContextualBSIResult:
        contexts, actions, rewards, behavior_probs, T_offline = _unpack_offline_data(offline_data)
        lambda_hat, Sigma = self.reward_model.fit(
            contexts=contexts,
            actions=actions,
            rewards=rewards,
            behavior_probs=behavior_probs,
        )

        pi1_img = contextual_bandit_exp_runner(
            reward_model=self.reward_model,
            eval_policy_builder=self.eval_policy_builder,
            context_sampler=self.context_sampler,
            T=self.T,
            n_reps=n_reps,
            lambda_params=lambda_hat,
            algo_seed=self.algo_seed,
            context_seed=self.context_seed,
            table_renew=True,
        )

        gradient = estimate_contextual_bsi_gradient(
            reward_model=self.reward_model,
            sim_result=pi1_img,
            lambda_params=lambda_hat,
        )
        se = float(np.sqrt(max(gradient @ Sigma @ gradient, 0.0)))
        center = float(pi1_img["mean_avg_reward"])
        center_se = float(pi1_img["se_avg_reward"])
        ci_width = {
            float(alpha): float(norm.ppf(1.0 - float(alpha) / 2.0) * se / np.sqrt(T_offline))
            for alpha in alphas
        }
        proj_ci_width = {
            float(alpha): float(np.sqrt(chi2.ppf(1.0 - float(alpha), df=lambda_hat.shape[0])) * se / np.sqrt(T_offline))
            for alpha in alphas
        }
        return ContextualBSIResult(
            center=center,
            center_se=center_se,
            ci_width=ci_width,
            proj_ci_width=proj_ci_width,
            se=se,
            lambda_hat=lambda_hat,
            gradient=gradient,
            Sigma=Sigma,
            pi1_img=pi1_img,
        )


def contextual_bandit_exp_runner(
    reward_model: Any,
    eval_policy_builder: Callable[[int], Any],
    context_sampler: Callable[[np.random.Generator, int], np.ndarray],
    T: int,
    n_reps: int,
    lambda_params: np.ndarray,
    algo_seed: int = 2026,
    context_seed: int = 1013,
    table_renew: bool = True,
) -> dict[str, np.ndarray | float | int]:
    if T <= 0 or n_reps <= 0:
        raise ValueError("T and n_reps must be positive.")

    is_linear = isinstance(reward_model, ContextualLinearGaussianRewardModel)
    is_logistic = isinstance(reward_model, ContextualLogisticBernoulliRewardModel)
    probe_policy = eval_policy_builder(algo_seed)
    can_vectorize_epsilon = (
        (is_linear or is_logistic)
        and reward_model.feature_map is None
        and not (is_linear and reward_model.sigma is None)
        and isinstance(probe_policy, ContextualEpsilonGreedyPolicy)
        and probe_policy.reward_type == ("linear_gaussian" if is_linear else "logistic_bernoulli")
        and probe_policy.include_intercept
    )
    can_vectorize_ts = (
        os.environ.get("CONTEXTUAL_BSI_DISABLE_TS_VECTORIZE", "").lower() not in {"1", "true", "yes"}
        and (is_linear or is_logistic)
        and reward_model.feature_map is None
        and not (is_linear and reward_model.sigma is None)
        and isinstance(probe_policy, ContextualTSPolicy)
        and probe_policy.reward_type == ("linear_gaussian" if is_linear else "logistic_bernoulli")
        and probe_policy.include_intercept
    )

    if can_vectorize_ts:
        n_actions = probe_policy.n_actions
        context_dim = probe_policy.context_dim
        p = context_dim + 1
        lambda_params = np.asarray(lambda_params, dtype=np.float64)
        if lambda_params.shape[0] == n_actions * p:
            first_rng = np.random.default_rng(context_seed)
            first_contexts = _as_2d_contexts(context_sampler(first_rng, T))
            if first_contexts.shape != (T, context_dim):
                raise ValueError(f"context_sampler must return shape {(T, context_dim)} every time.")

            all_contexts = np.zeros((n_reps, T, context_dim), dtype=np.float64)
            for rep in range(n_reps):
                if table_renew:
                    ctx_rng = np.random.default_rng(context_seed + rep)
                    contexts = _as_2d_contexts(context_sampler(ctx_rng, T))
                else:
                    contexts = first_contexts
                if contexts.shape != (T, context_dim):
                    raise ValueError(f"context_sampler must return shape {(T, context_dim)} every time.")
                all_contexts[rep] = contexts

            all_x = _add_policy_intercept(all_contexts)
            all_actions = np.zeros((n_reps, T), dtype=np.int64)
            all_rewards = np.zeros((n_reps, T), dtype=np.float64)
            all_probs = np.zeros((n_reps, T, n_actions), dtype=np.float64)
            beta = lambda_params.reshape(n_actions, p)

            action_rng = np.random.default_rng(algo_seed)
            policy_rng = np.random.default_rng(algo_seed + 1_000_003)
            reward_rng = np.random.default_rng(context_seed + 100_000)
            mc_chunk = _ts_mc_chunk(n_reps, n_actions, p, probe_policy.n_prob_mc)

            if is_linear:
                prior_precision_matrix = np.eye(p, dtype=np.float64) / probe_policy.prior_var
                prior_info = np.full(p, probe_policy.prior_mean, dtype=np.float64) / probe_policy.prior_var
                precision = np.broadcast_to(
                    prior_precision_matrix,
                    (n_reps, n_actions, p, p),
                ).copy()
                info = np.broadcast_to(prior_info, (n_reps, n_actions, p)).copy()
                sigma = float(reward_model.sigma)
            else:
                coefs = np.zeros((n_reps, n_actions, p), dtype=np.float64)
                covs = _logistic_ts_prior_cov(
                    n_reps=n_reps,
                    n_actions=n_actions,
                    p=p,
                    prior_precision=probe_policy.prior_precision,
                )
                dirty_actions = np.full(n_reps, -1, dtype=np.int64)
                sigma = None

            for step in range(T):
                x_step = all_x[:, step, :]
                if is_linear:
                    probs = _batched_linear_ts_action_probs_from_precision(
                        precision=precision,
                        info=info,
                        x_step=x_step,
                        n_prob_mc=probe_policy.n_prob_mc,
                        pi_clip=probe_policy.pi_clip,
                        rng=policy_rng,
                        mc_chunk=mc_chunk,
                    )
                else:
                    if step > 0:
                        for action in range(n_actions):
                            idx = np.where(dirty_actions == action)[0]
                            if idx.size == 0:
                                continue
                            x_history = all_x[idx, :step, :]
                            action_history = all_actions[idx, :step]
                            reward_history = all_rewards[idx, :step]
                            sample_mask = (action_history == action).astype(np.float64)
                            active = np.any(sample_mask > 0.0, axis=1)
                            beta_action = np.zeros((idx.size, p), dtype=np.float64)
                            precision_action = np.broadcast_to(
                                np.eye(p, dtype=np.float64) * probe_policy.prior_precision,
                                (idx.size, p, p),
                            ).copy()
                            if probe_policy.include_intercept:
                                precision_action[:, 0, 0] = 0.0
                            if np.any(active):
                                active_idx = np.where(active)[0]
                                beta_active = np.zeros((active_idx.size, p), dtype=np.float64)
                                xa_all = x_history[active]
                                ya_all = reward_history[active]
                                ma_all = sample_mask[active]
                                still_active = np.ones(active_idx.size, dtype=bool)
                                for _ in range(probe_policy.max_iter):
                                    if not np.any(still_active):
                                        break
                                    xa = xa_all[still_active]
                                    ya = ya_all[still_active]
                                    ma = ma_all[still_active]
                                    beta_now = beta_active[still_active]
                                    eta = np.clip(np.einsum("mtp,mp->mt", xa, beta_now), -35.0, 35.0)
                                    p_hat = expit(eta)
                                    weights = np.maximum(p_hat * (1.0 - p_hat), 1e-10)
                                    gradient = np.einsum("mt,mtp->mp", ma * (ya - p_hat), xa)
                                    hessian = np.einsum("mt,mtp,mtq->mpq", ma * weights, xa, xa)
                                    step_beta = _batched_solve_or_pinv(hessian, gradient)
                                    beta_next = beta_now + step_beta
                                    active_positions = np.where(still_active)[0]
                                    beta_active[active_positions] = beta_next
                                    converged = np.linalg.norm(step_beta, axis=1) <= probe_policy.tol * (
                                        1.0 + np.linalg.norm(beta_now, axis=1)
                                    )
                                    still_active[active_positions[converged]] = False
                                beta_action[active_idx] = beta_active
                                eta = np.clip(np.einsum("mtp,mp->mt", xa_all, beta_active), -35.0, 35.0)
                                p_hat = expit(eta)
                                weights = np.maximum(p_hat * (1.0 - p_hat), 1e-10)
                                hessian = np.einsum("mt,mtp,mtq->mpq", ma_all * weights, xa_all, xa_all)
                                precision_obs = hessian + probe_policy.prior_precision * np.eye(
                                    p, dtype=np.float64
                                )
                                if probe_policy.include_intercept:
                                    precision_obs[:, 0, 0] -= probe_policy.prior_precision
                                precision_action[active_idx] = precision_obs
                            coefs[idx, action, :] = beta_action
                            covs[idx, action, :, :] = _batched_inv_or_pinv(
                                precision_action + 1e-8 * np.eye(p, dtype=np.float64)
                            )
                        dirty_actions = np.full(n_reps, -1, dtype=np.int64)
                    probs = _batched_ts_action_probs(
                        means=coefs,
                        covs=covs,
                        x_step=x_step,
                        n_prob_mc=probe_policy.n_prob_mc,
                        pi_clip=probe_policy.pi_clip,
                        posterior_scale=probe_policy.posterior_scale,
                        reward_type="logistic_bernoulli",
                        rng=policy_rng,
                        mc_chunk=mc_chunk,
                    )

                all_probs[:, step, :] = probs
                actions = _sample_rows_from_probs(probs, action_rng)
                reward_means = np.einsum("mp,mp->m", x_step, beta[actions])
                if is_linear:
                    rewards = reward_rng.normal(reward_means, sigma, size=n_reps)
                    selected_x = x_step
                    precision[np.arange(n_reps), actions] += (
                        selected_x[:, :, None] * selected_x[:, None, :]
                    ) / (sigma**2)
                    info[np.arange(n_reps), actions] += selected_x * rewards[:, None] / (sigma**2)
                else:
                    reward_probs = expit(np.clip(reward_means, -35.0, 35.0))
                    rewards = reward_rng.binomial(1, reward_probs, size=n_reps).astype(np.float64)
                    dirty_actions = actions.copy()

                all_actions[:, step] = actions
                all_rewards[:, step] = rewards

            avg_rewards = all_rewards.mean(axis=1)
            return {
                "context_seed": context_seed,
                "algo_seed": algo_seed,
                "n_rep": n_reps,
                "T": T,
                "all_contexts": all_contexts,
                "all_actions": all_actions,
                "all_rewards": all_rewards,
                "all_probs": all_probs,
                "mean_avg_reward": float(avg_rewards.mean()),
                "se_avg_reward": float(avg_rewards.std(ddof=1) / np.sqrt(n_reps)) if n_reps > 1 else 0.0,
                "std_avg_reward": float(avg_rewards.std(ddof=1)) if n_reps > 1 else 0.0,
            }

    if can_vectorize_epsilon:
        n_actions = probe_policy.n_actions
        context_dim = probe_policy.context_dim
        p = context_dim + 1
        lambda_params = np.asarray(lambda_params, dtype=np.float64)
        if lambda_params.shape[0] == n_actions * p:
            first_rng = np.random.default_rng(context_seed)
            first_contexts = _as_2d_contexts(context_sampler(first_rng, T))
            if first_contexts.shape != (T, context_dim):
                raise ValueError(f"context_sampler must return shape {(T, context_dim)} every time.")

            all_contexts = np.zeros((n_reps, T, context_dim), dtype=np.float64)
            for rep in range(n_reps):
                if table_renew:
                    ctx_rng = np.random.default_rng(context_seed + rep)
                    contexts = _as_2d_contexts(context_sampler(ctx_rng, T))
                else:
                    contexts = first_contexts
                if contexts.shape != (T, context_dim):
                    raise ValueError(f"context_sampler must return shape {(T, context_dim)} every time.")
                all_contexts[rep] = contexts

            all_x = _add_policy_intercept(all_contexts)
            all_actions = np.zeros((n_reps, T), dtype=np.int64)
            all_rewards = np.zeros((n_reps, T), dtype=np.float64)
            all_probs = np.zeros((n_reps, T, n_actions), dtype=np.float64)

            if is_linear:
                gram = np.zeros((n_reps, n_actions, p, p), dtype=np.float64)
                rhs = np.zeros((n_reps, n_actions, p), dtype=np.float64)
            else:
                gram = None
                rhs = None
                policy_coefs = np.zeros((n_reps, n_actions, p), dtype=np.float64)
                dirty_actions = np.full(n_reps, -1, dtype=np.int64)

            counts = np.zeros((n_reps, n_actions), dtype=np.int64)
            beta = lambda_params.reshape(n_actions, p)
            sigma = float(reward_model.sigma) if is_linear else None
            action_rngs = [np.random.default_rng(algo_seed + rep) for rep in range(n_reps)]
            reward_rngs = [np.random.default_rng(context_seed + 100000 + rep) for rep in range(n_reps)]

            for step in range(T):
                probs = np.empty((n_reps, n_actions), dtype=np.float64)
                if is_logistic and step > 0:
                    for action in range(n_actions):
                        idx = np.where(dirty_actions == action)[0]
                        if idx.size == 0:
                            continue
                        x_history = all_x[idx, :step, :]
                        action_history = all_actions[idx, :step]
                        reward_history = all_rewards[idx, :step]
                        beta_action = np.zeros((idx.size, p), dtype=np.float64)
                        sample_mask = (action_history == action).astype(np.float64)
                        active = np.any(sample_mask > 0.0, axis=1)
                        for _ in range(probe_policy.max_iter):
                            if not np.any(active):
                                break
                            xa = x_history[active]
                            ya = reward_history[active]
                            ma = sample_mask[active]
                            beta_active = beta_action[active]
                            eta = np.clip(np.einsum("mtp,mp->mt", xa, beta_active), -35.0, 35.0)
                            p_hat = expit(eta)
                            weights = np.maximum(p_hat * (1.0 - p_hat), 1e-10)
                            gradient = np.einsum("mt,mtp->mp", ma * (ya - p_hat), xa)
                            hessian = np.einsum("mt,mtp,mtq->mpq", ma * weights, xa, xa)
                            try:
                                step_beta = np.linalg.solve(hessian, gradient[..., None])[..., 0]
                            except np.linalg.LinAlgError:
                                step_beta = np.stack(
                                    [
                                        np.linalg.pinv(hessian_i) @ gradient_i
                                        for hessian_i, gradient_i in zip(hessian, gradient)
                                    ]
                                )
                            beta_next = beta_active + step_beta
                            active_idx = np.where(active)[0]
                            beta_action[active_idx] = beta_next
                            converged = np.linalg.norm(step_beta, axis=1) <= probe_policy.tol * (
                                1.0 + np.linalg.norm(beta_active, axis=1)
                            )
                            active[active_idx[converged]] = False
                        policy_coefs[idx, action, :] = beta_action

                if probe_policy.explore_untried:
                    untried = counts == 0
                    has_untried = np.any(untried, axis=1)
                else:
                    has_untried = np.zeros(n_reps, dtype=bool)

                if np.any(has_untried):
                    probs[has_untried] = untried[has_untried] / np.sum(
                        untried[has_untried], axis=1, keepdims=True
                    )

                fit_rows = ~has_untried
                if np.any(fit_rows):
                    if is_linear:
                        gram_fit = gram[fit_rows]
                        rhs_fit = rhs[fit_rows]
                        counts_fit = counts[fit_rows]
                        coefs = np.zeros((gram_fit.shape[0], n_actions, p), dtype=np.float64)
                        mask = counts_fit.reshape(-1) > 0
                        if np.any(mask):
                            lhs = gram_fit.reshape(-1, p, p)[mask]
                            b = rhs_fit.reshape(-1, p)[mask]
                            try:
                                solved = np.linalg.solve(lhs, b[..., None])[..., 0]
                            except np.linalg.LinAlgError:
                                solved = np.stack(
                                    [np.linalg.pinv(lhs_i) @ b_i for lhs_i, b_i in zip(lhs, b)]
                                )
                            coefs.reshape(-1, p)[mask] = solved
                    else:
                        coefs = policy_coefs[fit_rows]
                    means_policy = np.einsum("mkp,mp->mk", coefs, all_x[fit_rows, step, :])
                    if is_logistic:
                        means_policy = expit(np.clip(means_policy, -35.0, 35.0))
                    best = np.argmax(means_policy, axis=1)
                    probs_fit = np.full(
                        (int(np.sum(fit_rows)), n_actions),
                        probe_policy.epsilon / n_actions,
                        dtype=np.float64,
                    )
                    probs_fit[np.arange(probs_fit.shape[0]), best] += 1.0 - probe_policy.epsilon
                    probs[fit_rows] = probs_fit

                all_probs[:, step, :] = probs
                actions = np.array(
                    [int(action_rngs[rep].choice(n_actions, p=probs[rep])) for rep in range(n_reps)],
                    dtype=np.int64,
                )
                reward_means = np.einsum("mp,mp->m", all_x[:, step, :], beta[actions])
                if is_linear:
                    rewards = np.array(
                        [reward_rngs[rep].normal(reward_means[rep], sigma) for rep in range(n_reps)],
                        dtype=np.float64,
                    )
                else:
                    reward_probs = expit(np.clip(reward_means, -35.0, 35.0))
                    rewards = np.array(
                        [reward_rngs[rep].binomial(1, reward_probs[rep]) for rep in range(n_reps)],
                        dtype=np.float64,
                    )

                all_actions[:, step] = actions
                all_rewards[:, step] = rewards

                x_step = all_x[:, step, :]
                if is_linear:
                    for action in range(n_actions):
                        idx = np.where(actions == action)[0]
                        if idx.size == 0:
                            continue
                        xa = x_step[idx]
                        gram[idx, action] += xa[:, :, None] * xa[:, None, :]
                        rhs[idx, action] += xa * rewards[idx, None]
                        counts[idx, action] += 1
                else:
                    for action in range(n_actions):
                        counts[actions == action, action] += 1
                    dirty_actions = actions.copy()

            avg_rewards = all_rewards.mean(axis=1)
            return {
                "context_seed": context_seed,
                "algo_seed": algo_seed,
                "n_rep": n_reps,
                "T": T,
                "all_contexts": all_contexts,
                "all_actions": all_actions,
                "all_rewards": all_rewards,
                "all_probs": all_probs,
                "mean_avg_reward": float(avg_rewards.mean()),
                "se_avg_reward": float(avg_rewards.std(ddof=1) / np.sqrt(n_reps)) if n_reps > 1 else 0.0,
                "std_avg_reward": float(avg_rewards.std(ddof=1)) if n_reps > 1 else 0.0,
            }

    first_rng = np.random.default_rng(context_seed)
    first_contexts = _as_2d_contexts(context_sampler(first_rng, T))
    context_dim = first_contexts.shape[1]
    all_contexts = np.zeros((n_reps, T, context_dim), dtype=np.float64)
    all_actions = np.zeros((n_reps, T), dtype=np.int64)
    all_rewards = np.zeros((n_reps, T), dtype=np.float64)
    all_probs = None

    for rep in range(n_reps):
        ctx_rng = np.random.default_rng(context_seed + rep if table_renew else context_seed)
        contexts = first_contexts if rep == 0 and not table_renew else _as_2d_contexts(context_sampler(ctx_rng, T))
        if contexts.shape != (T, context_dim):
            raise ValueError(f"context_sampler must return shape {(T, context_dim)} every time.")
        policy = eval_policy_builder(algo_seed + rep)
        action_rng = np.random.default_rng(algo_seed + rep)
        reward_rng = np.random.default_rng(context_seed + 100000 + rep)
        probs_rep = []

        for step in range(T):
            history = {
                "contexts": all_contexts[rep, :step],
                "actions": all_actions[rep, :step],
                "rewards": all_rewards[rep, :step],
            }
            probs = _policy_action_probs(policy, contexts[step], history)
            action = int(action_rng.choice(len(probs), p=probs))
            reward = reward_model.sample(
                contexts[step],
                action,
                rng=reward_rng,
                params=lambda_params,
            )
            _maybe_update_policy(policy, contexts[step], action, reward)

            all_contexts[rep, step] = contexts[step]
            all_actions[rep, step] = action
            all_rewards[rep, step] = reward
            probs_rep.append(probs)

        probs_rep = np.asarray(probs_rep, dtype=np.float64)
        if all_probs is None:
            all_probs = np.zeros((n_reps, T, probs_rep.shape[1]), dtype=np.float64)
        all_probs[rep] = probs_rep

    avg_rewards = all_rewards.mean(axis=1)
    return {
        "context_seed": context_seed,
        "algo_seed": algo_seed,
        "n_rep": n_reps,
        "T": T,
        "all_contexts": all_contexts,
        "all_actions": all_actions,
        "all_rewards": all_rewards,
        "all_probs": all_probs,
        "mean_avg_reward": float(avg_rewards.mean()),
        "se_avg_reward": float(avg_rewards.std(ddof=1) / np.sqrt(n_reps)) if n_reps > 1 else 0.0,
        "std_avg_reward": float(avg_rewards.std(ddof=1)) if n_reps > 1 else 0.0,
    }


def estimate_contextual_bsi_gradient(
    reward_model: Any,
    sim_result: dict[str, np.ndarray | float | int],
    lambda_params: np.ndarray,
) -> np.ndarray:
    contexts = np.asarray(sim_result["all_contexts"], dtype=np.float64)
    actions = np.asarray(sim_result["all_actions"], dtype=np.int64)
    rewards = np.asarray(sim_result["all_rewards"], dtype=np.float64)
    n_reps, T = actions.shape
    lambda_params = np.asarray(lambda_params, dtype=np.float64)

    if (
        isinstance(reward_model, ContextualLinearGaussianRewardModel)
        and reward_model.feature_map is None
        and reward_model.sigma is not None
        and contexts.ndim == 3
    ):
        n_actions = reward_model.n_actions
        p = contexts.shape[2] + 1
        if lambda_params.shape[0] == n_actions * p:
            x_design = _add_policy_intercept(contexts)
            beta = lambda_params.reshape(n_actions, p)
            residuals = rewards - np.einsum("mtp,mtp->mt", x_design, beta[actions])
            future_returns = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
            scale = residuals * future_returns / (T * float(reward_model.sigma) ** 2)
            gradient = np.zeros(n_actions * p, dtype=np.float64)
            for action in range(n_actions):
                mask = actions == action
                gradient[action * p : (action + 1) * p] = (
                    np.sum(x_design[mask] * scale[mask, None], axis=0) / n_reps
                )
            return gradient

    if (
        isinstance(reward_model, ContextualLogisticBernoulliRewardModel)
        and reward_model.feature_map is None
        and reward_model.include_intercept
        and contexts.ndim == 3
    ):
        n_actions = reward_model.n_actions
        p = contexts.shape[2] + 1
        if lambda_params.shape[0] == n_actions * p:
            x_design = _add_policy_intercept(contexts)
            beta = lambda_params.reshape(n_actions, p)
            probs = expit(np.clip(np.einsum("mtp,mtp->mt", x_design, beta[actions]), -35.0, 35.0))
            future_returns = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
            scale = (rewards - probs) * future_returns / T
            gradient = np.zeros(n_actions * p, dtype=np.float64)
            for action in range(n_actions):
                mask = actions == action
                gradient[action * p : (action + 1) * p] = (
                    np.sum(x_design[mask] * scale[mask, None], axis=0) / n_reps
                )
            return gradient

    gradient = np.zeros(lambda_params.shape[0], dtype=np.float64)

    for rep in range(n_reps):
        future_returns = np.cumsum(rewards[rep, ::-1])[::-1]
        for step in range(T):
            score = reward_model.score(
                contexts[rep, step],
                int(actions[rep, step]),
                float(rewards[rep, step]),
                params=lambda_params,
            )
            gradient += score * future_returns[step] / T

    gradient /= n_reps
    return gradient


def _ts_mc_chunk(n_reps: int, n_actions: int, p: int, n_prob_mc: int) -> int:
    max_draw_values = 8_000_000
    per_mc = max(int(n_reps) * int(n_actions) * int(p), 1)
    return max(1, min(int(n_prob_mc), max_draw_values // per_mc))


def _batched_ts_action_probs(
    means: np.ndarray,
    covs: np.ndarray,
    x_step: np.ndarray,
    n_prob_mc: int,
    pi_clip: float,
    posterior_scale: float,
    reward_type: str,
    rng: np.random.Generator,
    mc_chunk: int,
) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)
    x_step = np.asarray(x_step, dtype=np.float64)
    n_reps, n_actions, p = means.shape
    counts = np.zeros((n_reps, n_actions), dtype=np.float64)
    factors = _batched_psd_factor(covs) * float(posterior_scale)

    done = 0
    while done < n_prob_mc:
        chunk = min(int(mc_chunk), int(n_prob_mc) - done)
        values = np.empty((n_reps, chunk, n_actions), dtype=np.float64)
        for action in range(n_actions):
            z = rng.normal(size=(n_reps, chunk, p))
            theta = means[:, None, action, :] + np.einsum(
                "rmp,rqp->rmq",
                z,
                factors[:, action, :, :],
            )
            action_values = np.einsum("rmp,rp->rm", theta, x_step)
            if reward_type == "logistic_bernoulli":
                action_values = expit(np.clip(action_values, -35.0, 35.0))
            values[:, :, action] = action_values
        winners = np.argmax(values, axis=2)
        counts += np.sum(winners[..., None] == np.arange(n_actions), axis=1)
        done += chunk

    probs = counts / float(n_prob_mc)
    if pi_clip > 0.0:
        probs = np.maximum(probs, float(pi_clip))
        probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _batched_linear_ts_action_probs_from_precision(
    precision: np.ndarray,
    info: np.ndarray,
    x_step: np.ndarray,
    n_prob_mc: int,
    pi_clip: float,
    rng: np.random.Generator,
    mc_chunk: int,
) -> np.ndarray:
    precision = np.asarray(precision, dtype=np.float64)
    info = np.asarray(info, dtype=np.float64)
    x_step = np.asarray(x_step, dtype=np.float64)
    n_reps, n_actions, p = info.shape
    counts = np.zeros((n_reps, n_actions), dtype=np.float64)
    means = _batched_solve_or_pinv(precision, info)

    chol = np.empty_like(precision)
    for action in range(n_actions):
        chol[:, action] = _batched_cholesky_or_eig_factor_precision(precision[:, action])

    done = 0
    while done < n_prob_mc:
        chunk = min(int(mc_chunk), int(n_prob_mc) - done)
        values = np.empty((n_reps, chunk, n_actions), dtype=np.float64)
        for action in range(n_actions):
            z = rng.normal(size=(n_reps, chunk, p))
            noise = np.linalg.solve(
                np.swapaxes(chol[:, action], 1, 2),
                np.swapaxes(z, 1, 2),
            )
            theta = means[:, None, action, :] + np.swapaxes(noise, 1, 2)
            values[:, :, action] = np.einsum("rmp,rp->rm", theta, x_step)
        winners = np.argmax(values, axis=2)
        counts += np.sum(winners[..., None] == np.arange(n_actions), axis=1)
        done += chunk

    probs = counts / float(n_prob_mc)
    if pi_clip > 0.0:
        probs = np.maximum(probs, float(pi_clip))
        probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _batched_cholesky_or_eig_factor_precision(precision: np.ndarray) -> np.ndarray:
    precision = np.asarray(precision, dtype=np.float64)
    eye = np.eye(precision.shape[-1], dtype=np.float64)
    for scale in (1e-10, 1e-8, 1e-6, 1e-4):
        try:
            return np.linalg.cholesky(precision + scale * eye)
        except np.linalg.LinAlgError:
            pass
    eigvals, eigvecs = np.linalg.eigh(precision)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvecs * np.sqrt(eigvals)[:, None, :]


def _sample_rows_from_probs(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cdf = np.cumsum(np.asarray(probs, dtype=np.float64), axis=1)
    cdf[:, -1] = 1.0
    uniforms = rng.random(probs.shape[0])
    return np.sum(uniforms[:, None] > cdf, axis=1).astype(np.int64)


def _logistic_ts_prior_cov(
    n_reps: int,
    n_actions: int,
    p: int,
    prior_precision: float,
) -> np.ndarray:
    precision = np.eye(p, dtype=np.float64) * float(prior_precision)
    if p > 0:
        precision[0, 0] = 0.0
    cov = _batched_inv_or_pinv(
        np.broadcast_to(
            precision + 1e-8 * np.eye(p, dtype=np.float64),
            (n_reps, n_actions, p, p),
        )
    )
    return np.asarray(cov, dtype=np.float64)


def _batched_solve_or_pinv(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    try:
        return np.linalg.solve(lhs, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return (np.linalg.pinv(lhs) @ rhs[..., None])[..., 0]


def _batched_inv_or_pinv(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def _batched_psd_factor(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs * np.sqrt(eigvals)[..., None, :]


def _add_policy_intercept(contexts: np.ndarray) -> np.ndarray:
    contexts = np.asarray(contexts, dtype=np.float64)
    return np.concatenate(
        [np.ones((*contexts.shape[:-1], 1), dtype=np.float64), contexts],
        axis=-1,
    )


def _as_2d_contexts(contexts: np.ndarray) -> np.ndarray:
    contexts = np.asarray(contexts, dtype=np.float64)
    if contexts.ndim == 1:
        contexts = contexts.reshape(-1, 1)
    if contexts.ndim != 2:
        raise ValueError("contexts must be a 1D or 2D numeric array.")
    return contexts


def _validate_logs(contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
    if actions.ndim != 1 or rewards.ndim != 1:
        raise ValueError("actions and rewards must be 1D.")
    if contexts.shape[0] != actions.shape[0] or actions.shape[0] != rewards.shape[0]:
        raise ValueError("contexts, actions, and rewards must have the same length.")


def _unpack_offline_data(
    offline_data: dict[str, np.ndarray | int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int]:
    contexts = _as_2d_contexts(np.asarray(offline_data["contexts"], dtype=np.float64))
    actions = np.asarray(offline_data["actions"], dtype=np.int64)
    rewards = np.asarray(offline_data["rewards"], dtype=np.float64)
    _validate_logs(contexts, actions, rewards)
    behavior_probs = offline_data.get("behavior_probs")
    if behavior_probs is not None:
        behavior_probs = np.asarray(behavior_probs, dtype=np.float64)
        if behavior_probs.ndim != 2 or behavior_probs.shape[0] != contexts.shape[0]:
            raise ValueError("behavior_probs must have shape (T_offline, K).")
    T_offline = int(offline_data.get("T", contexts.shape[0]))
    if T_offline != contexts.shape[0]:
        raise ValueError("offline_data['T'] must match the number of logged rows.")
    return contexts, actions, rewards, behavior_probs, T_offline


def _policy_action_probs(policy: Any, context: np.ndarray, history: dict[str, np.ndarray]) -> np.ndarray:
    if hasattr(policy, "action_probs"):
        method = policy.action_probs
        try:
            probs = method(context, history=history)
        except TypeError:
            try:
                probs = method(context, history)
            except TypeError:
                probs = method(context)
    elif callable(policy):
        try:
            probs = policy(context, history=history)
        except TypeError:
            try:
                probs = policy(context, history)
            except TypeError:
                probs = policy(context)
    else:
        raise TypeError("policy must be callable or expose action_probs(context, history=...).")
    return _normalize_probs(probs)


def _normalize_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("policy probabilities must be 1D.")
    if np.any(~np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("policy probabilities must be finite and non-negative.")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("policy probabilities must have positive mass.")
    probs = probs / total
    probs = np.maximum(probs, eps)
    return probs / probs.sum()


def _maybe_update_policy(policy: Any, context: np.ndarray, action: int, reward: float) -> None:
    if not hasattr(policy, "update"):
        return
    try:
        policy.update(context, action, reward)
    except TypeError:
        policy.update(action, reward)
