from __future__ import annotations

import argparse

import numpy as np
from scipy.special import expit, logit

from .algorithms import ContextualEpsilonGreedyPolicy, ContextualTSPolicy
from .environments import ContextualLinearGaussianRewardModel, ContextualLogisticBernoulliRewardModel


def default_params(
    env: str,
    n_actions: int,
    context_dim: int,
    seed: int,
    param_scenario: str = "default",
) -> np.ndarray:
    del seed
    if n_actions != 3:
        raise ValueError("Default parameters are provided only for n_actions=3.")
    if context_dim < 0:
        raise ValueError("context_dim must be nonnegative.")

    if param_scenario == "mablike_var01":
        def expand(base: np.ndarray) -> np.ndarray:
            intercepts = base[:, :1]
            base_slopes = base[:, 1:]
            if context_dim == 0:
                return intercepts.reshape(-1)
            reps = int(np.ceil(context_dim / base_slopes.shape[1]))
            slopes = np.tile(base_slopes, reps)[:, :context_dim]
            return np.column_stack([intercepts, slopes]).reshape(-1)

        if env == "linear_gaussian":
            return expand(
                np.array(
                    [
                        [0.1, 0.01, -0.01],
                        [0.2, -0.01, 0.01],
                        [1.0, 0.01, 0.01],
                    ],
                    dtype=float,
                )
            )
        if env == "logistic_bernoulli":
            return expand(
                np.array(
                    [
                        [logit(0.35), 0.01, -0.01],
                        [logit(0.50), -0.01, 0.01],
                        [logit(0.60), 0.01, 0.01],
                    ],
                    dtype=float,
                )
            )
        raise ValueError(f"Unknown environment {env!r}.")

    if param_scenario != "default":
        raise ValueError(f"Unknown param_scenario {param_scenario!r}.")
    if context_dim != 2:
        raise ValueError("The default parameter scenario is fixed only for context_dim=2.")
    if env in {"linear_gaussian", "logistic_bernoulli"}:
        return np.array(
            [
                [0.15, 1.00, -0.75],
                [0.15, 1.00, -0.75],
                [0.05, -0.90, 0.95],
            ],
            dtype=float,
        ).reshape(-1)
    raise ValueError(f"Unknown environment {env!r}.")


def context_sampler(context_dim: int, context_var: float = 1.0):
    if context_var <= 0.0:
        raise ValueError("context_var must be positive.")

    def sample(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.0, np.sqrt(context_var), size=(n, context_dim))

    return sample


class UniformContextualPolicy:
    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)

    def action_probs(self, context: np.ndarray, history: dict[str, np.ndarray] | None = None) -> np.ndarray:
        del context, history
        return np.full(self.n_actions, 1.0 / self.n_actions, dtype=float)


class OracleEpsilonPolicy:
    def __init__(
        self,
        env: str,
        params: np.ndarray,
        n_actions: int,
        context_dim: int,
        epsilon: float,
    ):
        self.env = env
        self.params = np.asarray(params, dtype=float).reshape(n_actions, context_dim + 1)
        self.n_actions = int(n_actions)
        self.epsilon = float(epsilon)

    def action_probs(self, context: np.ndarray, history: dict[str, np.ndarray] | None = None) -> np.ndarray:
        del history
        x = np.concatenate([[1.0], np.asarray(context, dtype=float).reshape(-1)])
        values = x @ self.params.T
        if self.env == "logistic_bernoulli":
            values = expit(np.clip(values, -35.0, 35.0))
        best = int(np.argmax(values))
        probs = np.full(self.n_actions, self.epsilon / self.n_actions, dtype=float)
        probs[best] += 1.0 - self.epsilon
        return probs


def make_policy_builder(args: argparse.Namespace, name: str, epsilon: float, true_params: np.ndarray):
    def builder(seed: int):
        if name == "uniform":
            return UniformContextualPolicy(args.n_actions)
        if name == "oracle_epsilon":
            return OracleEpsilonPolicy(args.env, true_params, args.n_actions, args.context_dim, epsilon)
        if name == "contextual_epsilon":
            reward_type = "linear_gaussian" if args.env == "linear_gaussian" else "logistic_bernoulli"
            return ContextualEpsilonGreedyPolicy(
                n_actions=args.n_actions,
                context_dim=args.context_dim,
                epsilon=epsilon,
                reward_type=reward_type,
                explore_untried=getattr(args, "policy_explore_untried", False),
                seed=seed,
            )
        if name == "contextual_ts" and args.env == "linear_gaussian":
            return ContextualTSPolicy(
                n_actions=args.n_actions,
                context_dim=args.context_dim,
                reward_type="linear_gaussian",
                obs_sigma=args.obs_sigma,
                n_prob_mc=args.ts_prob_mc,
                seed=seed,
            )
        if name == "contextual_ts" and args.env == "logistic_bernoulli":
            return ContextualTSPolicy(
                n_actions=args.n_actions,
                context_dim=args.context_dim,
                reward_type="logistic_bernoulli",
                n_prob_mc=args.ts_prob_mc,
                seed=seed,
            )
        raise ValueError(f"Unsupported policy {name!r} for env {args.env!r}.")

    return builder


def make_reward_model(args: argparse.Namespace, adaptive_behavior: bool):
    if args.env == "linear_gaussian":
        return ContextualLinearGaussianRewardModel(
            n_actions=args.n_actions,
            context_dim=args.context_dim,
            sigma=None,
            adaptive_behavior=adaptive_behavior,
        )
    if args.env == "logistic_bernoulli":
        return ContextualLogisticBernoulliRewardModel(
            n_actions=args.n_actions,
            context_dim=args.context_dim,
            adaptive_behavior=adaptive_behavior,
        )
    raise ValueError(f"Unknown environment {args.env!r}.")


def simulate_reward(
    env: str,
    params: np.ndarray,
    context: np.ndarray,
    action: int,
    reward_sigma: float,
    rng: np.random.Generator,
) -> float:
    beta = params.reshape(-1, context.shape[0] + 1)
    x = np.concatenate([[1.0], context])
    mean = float(x @ beta[int(action)])
    if env == "linear_gaussian":
        return float(rng.normal(mean, reward_sigma))
    return float(rng.binomial(1, expit(np.clip(mean, -35.0, 35.0))))


def collect_offline_data(args: argparse.Namespace, true_params: np.ndarray, rep_idx: int) -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(args.seed + 1000 * rep_idx + args.T_offline)
    action_rng = np.random.default_rng(args.seed + 2000 * rep_idx + args.T_offline)
    reward_rng = np.random.default_rng(args.seed + 3000 * rep_idx + args.T_offline)
    contexts = context_sampler(args.context_dim, args.context_var)(rng, args.T_offline)
    actions = np.zeros(args.T_offline, dtype=int)
    rewards = np.zeros(args.T_offline, dtype=float)
    behavior_probs = np.zeros((args.T_offline, args.n_actions), dtype=float)
    policy = make_policy_builder(args, args.pi0, args.epsilon0, true_params)(
        args.seed + 4000 * rep_idx + args.T_offline
    )

    for t in range(args.T_offline):
        history = {"contexts": contexts[:t], "actions": actions[:t], "rewards": rewards[:t]}
        probs = np.asarray(policy.action_probs(contexts[t], history=history), dtype=float)
        probs = probs / probs.sum()
        action = int(action_rng.choice(args.n_actions, p=probs))
        reward = simulate_reward(args.env, true_params, contexts[t], action, args.reward_sigma, reward_rng)
        behavior_probs[t] = probs
        actions[t] = action
        rewards[t] = reward
        if hasattr(policy, "update"):
            policy.update(contexts[t], action, reward)

    return {
        "contexts": contexts,
        "actions": actions,
        "rewards": rewards,
        "behavior_probs": behavior_probs,
        "T": args.T_offline,
    }
