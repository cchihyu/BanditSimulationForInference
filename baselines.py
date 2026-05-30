from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from cvxopt import matrix, solvers
from numba import njit
from scipy.optimize import brentq
from scipy.stats import f, norm, t

from bsi.inference import bandit_exp_runner


# Avoid import-time cache failures in environments where Numba cannot
# derive a persistent cache locator for this source tree.

# Fast sample mean / standard-error helper shared across Wald-style intervals.

@njit(cache=False)
def _mean_and_se_numba(xs: np.ndarray) -> tuple[float, float]:
    n = xs.shape[0]
    mean_x = 0.0
    for i in range(n):
        mean_x += xs[i]
    mean_x /= n

    if n < 2:
        return mean_x, 0.0

    var = 0.0
    for i in range(n):
        d = xs[i] - mean_x
        var += d * d
    var /= (n - 1)
    return mean_x, math.sqrt(var / n)

# Sequential scale estimate used to stabilize the CADR score sequence.

@njit(cache=False)
def _running_cadr_sigma_inv_numba(d0: np.ndarray, min_samples: int) -> np.ndarray:
    n = d0.shape[0]
    sigma_inv = np.empty(n, dtype=np.float64)

    running_sum = 0.0
    running_sumsq = 0.0

    for t in range(n):
        if t >= min_samples:
            mean_d0 = running_sum / t
            mean_d0sq = running_sumsq / t
            sigma_sq_t = mean_d0sq - mean_d0 * mean_d0
            if sigma_sq_t < 1e-12:
                sigma_sq_t = 1e-12
            sigma_inv[t] = 1.0 / math.sqrt(sigma_sq_t)
        else:
            sigma_inv[t] = 1.0

        running_sum += d0[t]
        running_sumsq += d0[t] * d0[t]

    return sigma_inv

# Fits an armwise Gaussian reward model with conjugate normal updates.

@njit(cache=False)
def _fit_arm_reward_model_numba(
    actions: np.ndarray,
    rewards: np.ndarray,
    obs_var: np.ndarray,
    prior_mean: float,
    prior_var: float,
    k: int,
) -> np.ndarray:
    counts = np.zeros(k, dtype=np.float64)
    reward_sums = np.zeros(k, dtype=np.float64)

    for i in range(actions.shape[0]):
        a = actions[i]
        counts[a] += 1.0
        reward_sums[a] += rewards[i]

    prior_prec = 1.0 / prior_var
    like_prec = counts / obs_var
    post_prec = prior_prec + like_prec
    post_mean = (prior_mean * prior_prec + reward_sums / obs_var) / post_prec
    return post_mean

# Normalizes scalar or vector scale inputs and enforces positivity.

def _as_positive_std_array(x, k: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        if arr <= 0.0:
            raise ValueError(f"{name} must be positive.")
        return np.full(k, float(arr), dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != k:
        raise ValueError(f"{name} must be scalar or length-{k}.")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    return arr


def validate_static_bandit_inputs(
    reward_means,
    behavior_policy,
    env_type: str = "normal",
    reward_stds=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reward_means = np.asarray(reward_means, dtype=np.float64)
    behavior_policy = np.asarray(behavior_policy, dtype=np.float64)

    if reward_means.ndim != 1:
        raise ValueError("reward_means must be 1D.")
    if behavior_policy.ndim != 1:
        raise ValueError("behavior_policy must be 1D.")
    if behavior_policy.shape[0] != reward_means.shape[0]:
        raise ValueError("behavior_policy must match reward_means.")
    if np.any(behavior_policy <= 0.0):
        raise ValueError("behavior_policy must be strictly positive.")
    if not np.isclose(behavior_policy.sum(), 1.0, atol=1e-10):
        raise ValueError("behavior_policy must sum to 1.")

    if env_type == "bernoulli":
        if np.any((reward_means < 0.0) | (reward_means > 1.0)):
            raise ValueError("For bernoulli env, mus must be probabilities in [0,1].")
        reward_stds_arr = np.sqrt(np.clip(reward_means * (1.0 - reward_means), 0.0, None))
        return reward_means, reward_stds_arr, behavior_policy
    if env_type != "normal":
        raise ValueError("env_type must be 'normal' or 'bernoulli'.")

    reward_stds = _as_positive_std_array(reward_stds, reward_means.shape[0], "reward_stds")
    return reward_means, reward_stds, behavior_policy

# Draws logged actions and rewards under a fixed behavior policy.

def simulate_offline_dataset(
    t_off: int,
    reward_means: np.ndarray,
    reward_stds: np.ndarray,
    behavior_policy: np.ndarray,
    rng: np.random.Generator,
    env_type: str = "normal",
) -> tuple[np.ndarray, np.ndarray]:
    k = reward_means.shape[0]
    actions = rng.choice(k, size=t_off, p=behavior_policy)
    if env_type == "bernoulli":
        rewards = rng.binomial(1, reward_means[actions]).astype(np.float64)
    else:
        rewards = rng.normal(
            loc=reward_means[actions],
            scale=reward_stds[actions],
            size=t_off,
        ).astype(np.float64)
    return actions.astype(np.int64), rewards

# Monte Carlo estimate of the target policy value via repeated online rollouts.

def estimate_policy_average_value_mc(
    env,
    algo_builder,
    t: int,
    n_rollouts: int,
    seed: int,
) -> dict[str, float]:
    result = bandit_exp_runner(
        env=env,
        algo_builder=algo_builder,
        T=t,
        n_reps=n_rollouts,
        base_exp_seed=seed,
        table_seed=seed + 7919,
        table_renew=True,
    )
    return {
        "theta_star": float(result["mean_avg_reward"]),
        "theta_star_se": float(result["se_avg_reward"]),
    }


def estimate_true_ts_average_value_mc(
    reward_means: np.ndarray,
    reward_stds: np.ndarray,
    t: int,
    n_rollouts: int,
    seed: int,
    env_type: str = "normal",
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    obs_sigma: float | np.ndarray = 1.0,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> float:
    reward_means = np.asarray(reward_means, dtype=np.float64)
    reward_stds = np.asarray(reward_stds, dtype=np.float64)
    k = reward_means.shape[0]
    rng = np.random.default_rng(seed)
    total_reward = 0.0
    if env_type == "bernoulli":
        alpha = np.full((n_rollouts, k), prior_alpha, dtype=np.float64)
        beta = np.full((n_rollouts, k), prior_beta, dtype=np.float64)
        row_idx = np.arange(n_rollouts)

        for _ in range(t):
            theta = rng.beta(alpha, beta)
            chosen = np.argmax(theta, axis=1)
            rewards = rng.binomial(1, reward_means[chosen]).astype(np.float64)
            total_reward += rewards.sum()
            alpha[row_idx, chosen] += rewards
            beta[row_idx, chosen] += 1.0 - rewards
    else:
        obs_sigma = _as_positive_std_array(obs_sigma, k, "obs_sigma")
        obs_var = obs_sigma ** 2
        post_mean = np.full((n_rollouts, k), prior_mean, dtype=np.float64)
        post_var = np.full((n_rollouts, k), prior_var, dtype=np.float64)
        row_idx = np.arange(n_rollouts)

        for _ in range(t):
            theta = rng.normal(post_mean, np.sqrt(post_var))
            chosen = np.argmax(theta, axis=1)

            rewards = rng.normal(
                loc=reward_means[chosen],
                scale=reward_stds[chosen],
                size=n_rollouts,
            ).astype(np.float64)
            total_reward += rewards.sum()

            v0 = post_var[row_idx, chosen]
            m0 = post_mean[row_idx, chosen]
            sig2 = obs_var[chosen]

            v1 = 1.0 / (1.0 / v0 + 1.0 / sig2)
            m1 = v1 * (m0 / v0 + rewards / sig2)

            post_var[row_idx, chosen] = v1
            post_mean[row_idx, chosen] = m1

    return float(total_reward / (n_rollouts * t))

# Returns evaluation-policy action probabilities, using Monte Carlo when needed.

def _policy_action_probs(algo, n_prob_mc: int) -> np.ndarray:
    if hasattr(algo, "select_action_with_probs"):
        try:
            _, probs = algo.select_action_with_probs(n_samples=n_prob_mc)
        except TypeError:
            _, probs = algo.select_action_with_probs()
        probs = np.asarray(probs, dtype=np.float64)
    else:
        action = int(algo.select_action())
        probs = np.zeros(algo.n_actions, dtype=np.float64)
        probs[action] = 1.0

    probs = np.maximum(probs, 1e-8)
    probs /= probs.sum()
    return probs


def compute_history_dependent_policy_weights(
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
    eval_algo_builder,
    n_eval_prob_mc: int,
    eval_seed: int,
    weight_mode: str = "one_step",
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    behavior_probs = np.asarray(behavior_probs, dtype=np.float64)

    if behavior_probs.ndim != 2 or behavior_probs.shape[0] != actions.shape[0]:
        raise ValueError("behavior_probs must have shape (T, K).")

    algo = eval_algo_builder(eval_seed)
    n = actions.shape[0]
    k = behavior_probs.shape[1]

    if weight_mode not in {"one_step", "cumulative"}:
        raise ValueError("weight_mode must be either 'one_step' or 'cumulative'.")

    weights = np.empty(n, dtype=np.float64)
    pi_hist = np.empty((n, k), dtype=np.float64)
    cumulative_ratio = 1.0

    for t in range(n):
        pi_t = _policy_action_probs(algo, n_eval_prob_mc)
        pi_t = (1.0 - k * eps) * pi_t + eps
        pi_t /= pi_t.sum()
        pi_hist[t] = pi_t

        a = actions[t]
        denom = max(float(behavior_probs[t, a]), eps)
        one_step_ratio = float(pi_t[a]) / denom
        if weight_mode == "one_step":
            weights[t] = one_step_ratio
        else:
            cumulative_ratio *= one_step_ratio
            weights[t] = cumulative_ratio

        algo.update(int(a), float(rewards[t]))

    return weights, pi_hist

# One-step uses the current ratio only; cumulative multiplies ratios over time.

def compute_history_dependent_ts_weights(
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_policy: np.ndarray,
    prior_mean: float,
    prior_var: float,
    obs_sigma: float | np.ndarray,
    n_ts_prob_mc: int,
    rng: np.random.Generator,
    env_type: str = "normal",
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    weight_mode: str = "one_step",
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    behavior_policy = np.asarray(behavior_policy, dtype=np.float64)

    n = actions.shape[0]
    k = behavior_policy.shape[0]
    if env_type == "bernoulli":
        alpha = np.full(k, prior_alpha, dtype=np.float64)
        beta = np.full(k, prior_beta, dtype=np.float64)
    else:
        obs_sigma = _as_positive_std_array(obs_sigma, k, "obs_sigma")
        obs_var = obs_sigma ** 2
        post_mean = np.full(k, prior_mean, dtype=np.float64)
        post_var = np.full(k, prior_var, dtype=np.float64)

    weights = np.empty(n, dtype=np.float64)
    pi_hist = np.empty((n, k), dtype=np.float64)
    cumulative_ratio = 1.0

    if weight_mode not in {"one_step", "cumulative"}:
        raise ValueError("weight_mode must be either 'one_step' or 'cumulative'.")

    for t in range(n):
        if env_type == "bernoulli":
            theta = rng.beta(alpha, beta, size=(n_ts_prob_mc, k))
        else:
            theta = rng.normal(
                loc=post_mean,
                scale=np.sqrt(post_var),
                size=(n_ts_prob_mc, k),
            )
        chosen = np.argmax(theta, axis=1)
        counts = np.bincount(chosen, minlength=k).astype(np.float64)

        pi_t = counts / n_ts_prob_mc
        pi_t = (1.0 - k * eps) * pi_t + eps
        pi_t = pi_t / pi_t.sum()

        pi_hist[t] = pi_t

        a = actions[t]
        one_step_ratio = pi_t[a] / behavior_policy[a]
        if weight_mode == "one_step":
            weights[t] = one_step_ratio
        else:
            cumulative_ratio *= one_step_ratio
            weights[t] = cumulative_ratio

        r = rewards[t]
        if env_type == "bernoulli":
            alpha[a] += r
            beta[a] += 1.0 - r
        else:
            v0 = post_var[a]
            m0 = post_mean[a]
            sig2 = obs_var[a]

            v1 = 1.0 / (1.0 / v0 + 1.0 / sig2)
            m1 = v1 * (m0 / v0 + r / sig2)

            post_var[a] = v1
            post_mean[a] = m1

    return weights, pi_hist


def _mean_and_se(xs: np.ndarray) -> tuple[float, float]:
    xs = np.asarray(xs, dtype=np.float64)
    mean_x, se = _mean_and_se_numba(xs)
    return float(mean_x), float(se)


def ipw_wald_interval(
    weights: np.ndarray,
    rewards: np.ndarray,
    conf_level: float,
) -> tuple[float, float]:
    xs = np.asarray(weights, dtype=np.float64) * np.asarray(rewards, dtype=np.float64)
    mean_x, se_x = _mean_and_se(xs)
    z = norm.ppf(0.5 + conf_level / 2.0)
    return mean_x - z * se_x, mean_x + z * se_x


def weighted_t_test_interval(
    weights: np.ndarray,
    rewards: np.ndarray,
    conf_level: float,
) -> tuple[float, float]:
    xs = np.asarray(weights, dtype=np.float64) * np.asarray(rewards, dtype=np.float64)
    mean_x, se_x = _mean_and_se(xs)
    if xs.size < 2:
        return mean_x, mean_x
    tcrit = t.ppf(0.5 + conf_level / 2.0, df=xs.size - 1)
    return mean_x - tcrit * se_x, mean_x + tcrit * se_x


def naive_t_test_interval(
    rewards: np.ndarray,
    conf_level: float,
) -> tuple[float, float]:
    xs = np.asarray(rewards, dtype=np.float64)
    mean_x, se_x = _mean_and_se(xs)
    if xs.size < 2:
        return mean_x, mean_x
    tcrit = t.ppf(0.5 + conf_level / 2.0, df=xs.size - 1)
    return mean_x - tcrit * se_x, mean_x + tcrit * se_x

def cadr_rescaled_interval(
    rewards: np.ndarray,
    weights: np.ndarray,
    conf_level: float,
    min_samples: int = 30,
) -> tuple[float, float]:
    rewards = np.asarray(rewards, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    d0 = weights * rewards
    sigma_inv = _running_cadr_sigma_inv_numba(d0, min_samples=min_samples)
    d = sigma_inv * d0

    _, se_d = _mean_and_se(d)
    z = norm.ppf(0.5 + conf_level / 2.0)

    sigma_bar = float(np.mean(1.0 / sigma_inv))
    half_width = z * se_d * sigma_bar
    center = float(np.mean(d0))
    return center - half_width, center + half_width


def cadr_interval(
    rewards: np.ndarray,
    weights: np.ndarray,
    conf_level: float,
    min_samples: int = 30,
) -> tuple[float, float]:
    rewards = np.asarray(rewards, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    d0 = weights * rewards
    sigma_inv = _running_cadr_sigma_inv_numba(d0, min_samples=min_samples)
    gamma = 1.0 / float(np.mean(sigma_inv))

    center = gamma * float(np.mean(sigma_inv * d0))
    z = norm.ppf(0.5 + conf_level / 2.0)
    half_width = z * gamma / math.sqrt(d0.shape[0])
    return center - half_width, center + half_width


def fit_arm_reward_model(
    actions: np.ndarray,
    rewards: np.ndarray,
    k: int,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    obs_sigma: float | np.ndarray = 1.0,
    env_type: str = "normal",
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if env_type == "bernoulli":
        counts = np.zeros(k, dtype=np.float64)
        successes = np.zeros(k, dtype=np.float64)
        for i in range(actions.shape[0]):
            a = actions[i]
            counts[a] += 1.0
            successes[a] += rewards[i]
        return (successes + prior_alpha) / (counts + prior_alpha + prior_beta)
    obs_sigma = _as_positive_std_array(obs_sigma, k, "obs_sigma")
    obs_var = obs_sigma ** 2

    return _fit_arm_reward_model_numba(actions, rewards, obs_var, prior_mean, prior_var, k)


def dr_scores(
    actions: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    pi_hist: np.ndarray,
    qhat: np.ndarray,
) -> np.ndarray:
    direct_part = pi_hist @ qhat
    residual_part = weights * (rewards - qhat[actions])
    return direct_part + residual_part


def compress_datagen(weights, rewards):
    counts = {}
    for w, r in zip(weights, rewards):
        key = (float(w), float(r))
        counts[key] = counts.get(key, 0) + 1

    items = [(c, w, r) for (w, r), c in counts.items()]

    def datagen():
        for c, w, r in items:
            yield c, w, r

    return datagen


def elfcb_estimate_repo(datagen, wmin, wmax, rmin=0.0, rmax=1.0):
    assert wmin >= 0.0
    assert wmin < 1.0
    assert wmax > 1.0
    assert rmax >= rmin

    num = sum(c for c, _, _ in datagen())
    if num < 1:
        raise ValueError("Need at least one observation.")

    def sumofw(beta):
        return sum(
            (c * w) / ((w - 1.0) * beta + num)
            for c, w, _ in datagen()
            if c > 0
        )

    def graddualobjective(beta):
        return sum(
            c * (w - 1.0) / ((w - 1.0) * beta + num)
            for c, w, _ in datagen()
            if c > 0
        )

    betamax = min(
        ((num - c) / (1.0 - w) for c, w, _ in datagen() if w < 1.0 and c > 0),
        default=num / (1.0 - wmin),
    )
    betamax = min(betamax, num / (1.0 - wmin))

    betamin = max(
        ((num - c) / (1.0 - w) for c, w, _ in datagen() if w > 1.0 and c > 0),
        default=num / (1.0 - wmax),
    )
    betamin = max(betamin, num / (1.0 - wmax))

    gradmin = graddualobjective(betamin)
    gradmax = graddualobjective(betamax)

    if gradmin * gradmax < 0.0:
        betastar = brentq(graddualobjective, betamin, betamax)
    elif gradmin < 0.0:
        betastar = betamin
    else:
        betastar = betamax

    remw = max(0.0, 1.0 - sumofw(betastar))
    vhat = 0.0

    for c, w, r in datagen():
        if c > 0:
            vhat += w * r * c / ((w - 1.0) * betastar + num)

    vmin = vhat + remw * rmin
    vmax = vhat + remw * rmax
    vhat += remw * (rmin + rmax) / 2.0

    qfunc = lambda c, w, r: c / (num + betastar * (w - 1.0))
    return {
        "betastar": betastar,
        "vmin": vmin,
        "vmax": vmax,
        "num": num,
        "vhat": vhat,
        "qfunc": qfunc,
    }


def elfcb_confidence_interval_repo(
    datagen,
    wmin,
    wmax,
    alpha_mis=0.05,
    rmin=0.0,
    rmax=1.0,
    show_cvxopt_progress=False,
):
    qmle = elfcb_estimate_repo(datagen, wmin=wmin, wmax=wmax, rmin=rmin, rmax=rmax)
    num = qmle["num"]
    if num < 2:
        return (rmin, rmax), (None, None)

    betamle = qmle["betastar"]
    delta = 0.5 * f.isf(q=alpha_mis, dfn=1, dfd=num - 1)

    sumwsq = sum(c * w * w for c, w, _ in datagen())
    wscale = max(1.0, np.sqrt(sumwsq / num))
    rscale = max(1.0, abs(rmin), abs(rmax))

    tiny = 1e-5
    logtiny = math.log(tiny)

    def logstar(x):
        if x > tiny:
            return math.log(x)
        xt = x / tiny
        return -1.5 + logtiny + 2.0 * xt - 0.5 * xt * xt

    def jaclogstar(x):
        if x > tiny:
            return 1.0 / x
        return (2.0 - (x / tiny)) / tiny

    def hesslogstar(x):
        if x > tiny:
            return -1.0 / (x * x)
        return -1.0 / (tiny * tiny)

    def dualobjective(p, sign):
        gamma, beta = float(p[0]), float(p[1])
        logcost = -delta
        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                mledenom = num + betamle * (w - 1.0)
                logcost += c * (logstar(denom) - logstar(mledenom))
        assert n == num
        if n > 0:
            logcost /= n
        return (-n * math.exp(logcost) + gamma + beta / wscale) / rscale

    def jacdualobjective(p, sign):
        gamma, beta = float(p[0]), float(p[1])
        logcost = -delta
        jac = np.zeros(2, dtype=np.float64)
        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                mledenom = num + betamle * (w - 1.0)
                logcost += c * (logstar(denom) - logstar(mledenom))
                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wscale)
        assert n == num
        if n > 0:
            logcost /= n
            jac /= n
        jac *= -(n / rscale) * math.exp(logcost)
        jac[0] += 1.0 / rscale
        jac[1] += 1.0 / (wscale * rscale)
        return jac

    def hessdualobjective(p, sign):
        gamma, beta = float(p[0]), float(p[1])
        logcost = -delta
        jac = np.zeros(2, dtype=np.float64)
        hess = np.zeros((2, 2), dtype=np.float64)
        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                mledenom = num + betamle * (w - 1.0)
                logcost += c * (logstar(denom) - logstar(mledenom))
                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wscale)
                hesslogcost = c * hesslogstar(denom)
                hess[0, 0] += hesslogcost
                hess[0, 1] += hesslogcost * (w / wscale)
                hess[1, 1] += hesslogcost * (w / wscale) * (w / wscale)
        assert n == num
        if n > 0:
            logcost /= n
            jac /= n
            hess /= n
        hess[1, 0] = hess[0, 1]
        hess += np.outer(jac, jac)
        hess *= -(n / rscale) * math.exp(logcost)
        return hess

    conse = np.array(
        [[1.0, w / wscale] for w in (wmin, wmax) for r in (rmin, rmax)],
        dtype=np.float64,
    )

    retvals = []
    easybounds = [
        (qmle["vmin"] <= rmin + tiny, rmin),
        (qmle["vmax"] >= rmax - tiny, rmax),
    ]

    solvers.options["show_progress"] = show_cvxopt_progress

    for what in range(2):
        if easybounds[what][0]:
            retvals.append((easybounds[what][1], None))
            continue

        sign = 1 - 2 * what
        d = np.array(
            [-sign * w * r + tiny for w in (wmin, wmax) for r in (rmin, rmax)],
            dtype=np.float64,
        )
        minsr = min(sign * rmin, sign * rmax)
        gamma0 = num - qmle["betastar"] + 2.0 * tiny
        beta0 = wscale * (qmle["betastar"] - (1.0 + 1.0 / wscale) * minsr)
        x0 = np.array([gamma0, beta0], dtype=np.float64)

        def F(x=None, z=None):
            if x is None:
                return 0, matrix(x0)
            p = np.reshape(np.array(x), -1)
            fval = dualobjective(p, sign)
            jf = jacdualobjective(p, sign)
            df = matrix(jf, (1, 2))
            if z is None:
                return fval, df
            hf = z[0] * hessdualobjective(p, sign)
            h = matrix(hf, hf.shape)
            return fval, df, h

        soln = solvers.cp(F, G=-matrix(conse, conse.shape), h=-matrix(d))

        if soln["status"] != "optimal":
            raise RuntimeError(f"cvxopt.cp failed with status={soln['status']}")

        xstar = np.reshape(np.array(soln["x"]), -1)
        fstar = float(soln["primal objective"])
        gammastar = float(xstar[0])
        betastar = float(xstar[1]) / wscale
        kappastar = (-rscale * fstar + gammastar + betastar) / num

        qfunc = lambda c, w, r, kappa=kappastar, gamma=gammastar, beta=betastar, s=sign: (
            kappa * c / (gamma + (beta + s * r) * w)
        )

        vbound = -sign * rscale * fstar
        retvals.append(
            (
                vbound,
                {
                    "gammastar": gammastar,
                    "betastar": betastar,
                    "kappastar": kappastar,
                    "qfunc": qfunc,
                },
            )
        )

    return (retvals[0][0], retvals[1][0]), (retvals[0][1], retvals[1][1])


@dataclass
class BaselineIntervals:
    elfcb: tuple[float, float]
    ipw: tuple[float, float]
    weighted_t_test: tuple[float, float] | None
    cadr: tuple[float, float]
    cadr_rescaled: tuple[float, float] | None
    dr: tuple[float, float]


def dr_interval(
    actions: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    pi_hist: np.ndarray,
    conf_level: float,
    reward_model_prior_mean: float = 0.0,
    reward_model_prior_var: float = 1.0,
    reward_model_obs_sigma: float | np.ndarray = 1.0,
    env_type: str = "normal",
    reward_model_prior_alpha: float = 1.0,
    reward_model_prior_beta: float = 1.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    pi_hist = np.asarray(pi_hist, dtype=np.float64)
    n = actions.shape[0]
    k = pi_hist.shape[1]
    qhat_full = fit_arm_reward_model(
        actions=actions,
        rewards=rewards,
        k=k,
        prior_mean=reward_model_prior_mean,
        prior_var=reward_model_prior_var,
        obs_sigma=reward_model_obs_sigma,
        env_type=env_type,
        prior_alpha=reward_model_prior_alpha,
        prior_beta=reward_model_prior_beta,
    )
    theta_hat = float(np.mean(dr_scores(actions, rewards, weights, pi_hist, qhat_full)))
    if n < 2:
        return theta_hat, theta_hat

    rng = np.random.default_rng(seed)
    boot_estimates = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        qhat_b = fit_arm_reward_model(
            actions=actions[idx],
            rewards=rewards[idx],
            k=k,
            prior_mean=reward_model_prior_mean,
            prior_var=reward_model_prior_var,
            obs_sigma=reward_model_obs_sigma,
            env_type=env_type,
            prior_alpha=reward_model_prior_alpha,
            prior_beta=reward_model_prior_beta,
        )
        phi_b = dr_scores(
            actions=actions[idx],
            rewards=rewards[idx],
            weights=weights[idx],
            pi_hist=pi_hist[idx],
            qhat=qhat_b,
        )
        boot_estimates[b] = float(np.mean(phi_b))

    se_boot = float(np.std(boot_estimates, ddof=1))
    z = norm.ppf(0.5 + conf_level / 2.0)
    return theta_hat - z * se_boot, theta_hat + z * se_boot


# End-to-end baseline evaluation: weights, ELFCB, IPW, CADR, DR, and optional t-test.

def compute_all_intervals(
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
    eval_algo_builder,
    conf_level: float,
    n_eval_prob_mc: int,
    eval_seed: int,
    reward_model_prior_mean: float,
    reward_model_prior_var: float,
    reward_model_obs_sigma: float | np.ndarray,
    env_type: str = "normal",
    reward_model_prior_alpha: float = 1.0,
    reward_model_prior_beta: float = 1.0,
    weight_mode: str = "one_step",
    prob_eps: float = 1e-8,
    cadr_min_samples: int = 30,
    show_cvxopt_progress: bool = False,
    include_weighted_t_test: bool = False,
    include_cadr_rescaled: bool = False,
    dr_bootstrap_reps: int = 1000,
    dr_bootstrap_seed: int = 0,
) -> BaselineIntervals:
    weights, pi_hist = compute_history_dependent_policy_weights(
        actions=actions,
        rewards=rewards,
        behavior_probs=behavior_probs,
        eval_algo_builder=eval_algo_builder,
        n_eval_prob_mc=n_eval_prob_mc,
        eval_seed=eval_seed,
        weight_mode=weight_mode,
        eps=prob_eps,
    )

    datagen = compress_datagen(weights, rewards)
    eps = 1e-10
    wmin = float(min(np.min(weights), 1.0 - eps))
    wmax = float(max(np.max(weights), 1.0 + eps))
    rmin = float(np.min(rewards))
    rmax = float(np.max(rewards))

    try:
        elfcb, _ = elfcb_confidence_interval_repo(
            datagen=datagen,
            wmin=wmin,
            wmax=wmax,
            alpha_mis=1.0 - conf_level,
            rmin=rmin,
            rmax=rmax,
            show_cvxopt_progress=show_cvxopt_progress,
        )
    except RuntimeError:
        elfcb = (float("nan"), float("nan"))
    ipw = ipw_wald_interval(weights, rewards, conf_level)
    weighted_t_test = None
    if include_weighted_t_test:
        weighted_t_test = weighted_t_test_interval(weights, rewards, conf_level)
    cadr = cadr_interval(rewards, weights, conf_level, min_samples=cadr_min_samples)
    cadr_rescaled = None
    if include_cadr_rescaled:
        cadr_rescaled = cadr_rescaled_interval(
            rewards=rewards,
            weights=weights,
            conf_level=conf_level,
            min_samples=cadr_min_samples,
        )
    dr = dr_interval(
        actions=actions,
        rewards=rewards,
        weights=weights,
        pi_hist=pi_hist,
        conf_level=conf_level,
        reward_model_prior_mean=reward_model_prior_mean,
        reward_model_prior_var=reward_model_prior_var,
        reward_model_obs_sigma=reward_model_obs_sigma,
        env_type=env_type,
        reward_model_prior_alpha=reward_model_prior_alpha,
        reward_model_prior_beta=reward_model_prior_beta,
        n_bootstrap=dr_bootstrap_reps,
        seed=dr_bootstrap_seed,
    )
    return BaselineIntervals(
        elfcb=elfcb,
        ipw=ipw,
        weighted_t_test=weighted_t_test,
        cadr=cadr,
        cadr_rescaled=cadr_rescaled,
        dr=dr,
    )

@njit(cache=False)
def _fit_arm_reward_model_numba(
    actions: np.ndarray,
    rewards: np.ndarray,
    obs_var: np.ndarray,
    prior_mean: float,
    prior_var: float,
    k: int,
) -> np.ndarray:
    counts = np.zeros(k, dtype=np.float64)
    reward_sums = np.zeros(k, dtype=np.float64)

    for i in range(actions.shape[0]):
        a = actions[i]
        counts[a] += 1.0
        reward_sums[a] += rewards[i]

    prior_prec = 1.0 / prior_var
    like_prec = counts / obs_var
    post_prec = prior_prec + like_prec
    post_mean = (prior_mean * prior_prec + reward_sums / obs_var) / post_prec
    return post_mean

# Normalizes scalar or vector scale inputs and enforces positivity.

def _as_positive_std_array(x, k: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        if arr <= 0.0:
            raise ValueError(f"{name} must be positive.")
        return np.full(k, float(arr), dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != k:
        raise ValueError(f"{name} must be scalar or length-{k}.")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    return arr


def validate_static_bandit_inputs(
    reward_means,
    behavior_policy,
    env_type: str = "normal",
    reward_stds=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reward_means = np.asarray(reward_means, dtype=np.float64)
    behavior_policy = np.asarray(behavior_policy, dtype=np.float64)

    if reward_means.ndim != 1:
        raise ValueError("reward_means must be 1D.")
    if behavior_policy.ndim != 1:
        raise ValueError("behavior_policy must be 1D.")
    if behavior_policy.shape[0] != reward_means.shape[0]:
        raise ValueError("behavior_policy must match reward_means.")
    if np.any(behavior_policy <= 0.0):
        raise ValueError("behavior_policy must be strictly positive.")
    if not np.isclose(behavior_policy.sum(), 1.0, atol=1e-10):
        raise ValueError("behavior_policy must sum to 1.")

    if env_type == "bernoulli":
        if np.any((reward_means < 0.0) | (reward_means > 1.0)):
            raise ValueError("For bernoulli env, mus must be probabilities in [0,1].")
        reward_stds_arr = np.sqrt(np.clip(reward_means * (1.0 - reward_means), 0.0, None))
        return reward_means, reward_stds_arr, behavior_policy
    if env_type != "normal":
        raise ValueError("env_type must be 'normal' or 'bernoulli'.")

    reward_stds = _as_positive_std_array(reward_stds, reward_means.shape[0], "reward_stds")
    return reward_means, reward_stds, behavior_policy

# Draws logged actions and rewards under a fixed behavior policy.

def simulate_offline_dataset(
    t_off: int,
    reward_means: np.ndarray,
    reward_stds: np.ndarray,
    behavior_policy: np.ndarray,
    rng: np.random.Generator,
    env_type: str = "normal",
) -> tuple[np.ndarray, np.ndarray]:
    k = reward_means.shape[0]
    actions = rng.choice(k, size=t_off, p=behavior_policy)
    if env_type == "bernoulli":
        rewards = rng.binomial(1, reward_means[actions]).astype(np.float64)
    else:
        rewards = rng.normal(
            loc=reward_means[actions],
            scale=reward_stds[actions],
            size=t_off,
        ).astype(np.float64)
    return actions.astype(np.int64), rewards

# Monte Carlo estimate of the target policy value via repeated online rollouts.

def estimate_policy_average_value_mc(
    env,
    algo_builder,
    t: int,
    n_rollouts: int,
    seed: int,
) -> dict[str, float]:
    result = bandit_exp_runner(
        env=env,
        algo_builder=algo_builder,
        T=t,
        n_reps=n_rollouts,
        base_exp_seed=seed,
        table_seed=seed + 7919,
        table_renew=True,
    )
    return {
        "theta_star": float(result["mean_avg_reward"]),
        "theta_star_se": float(result["se_avg_reward"]),
    }


# Returns evaluation-policy action probabilities, using Monte Carlo when needed.

def _policy_action_probs(algo, n_prob_mc: int) -> np.ndarray:
    if hasattr(algo, "select_action_with_probs"):
        try:
            _, probs = algo.select_action_with_probs(n_samples=n_prob_mc)
        except TypeError:
            _, probs = algo.select_action_with_probs()
        probs = np.asarray(probs, dtype=np.float64)
    else:
        action = int(algo.select_action())
        probs = np.zeros(algo.n_actions, dtype=np.float64)
        probs[action] = 1.0

    probs = np.maximum(probs, 1e-8)
    probs /= probs.sum()
    return probs


def compute_history_dependent_policy_weights(
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
    eval_algo_builder,
    n_eval_prob_mc: int,
    eval_seed: int,
    weight_mode: str = "one_step",
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    behavior_probs = np.asarray(behavior_probs, dtype=np.float64)

    if behavior_probs.ndim != 2 or behavior_probs.shape[0] != actions.shape[0]:
        raise ValueError("behavior_probs must have shape (T, K).")

    algo = eval_algo_builder(eval_seed)
    n = actions.shape[0]
    k = behavior_probs.shape[1]

    if weight_mode not in {"one_step", "cumulative"}:
        raise ValueError("weight_mode must be either 'one_step' or 'cumulative'.")

    weights = np.empty(n, dtype=np.float64)
    pi_hist = np.empty((n, k), dtype=np.float64)
    cumulative_ratio = 1.0

    for t in range(n):
        pi_t = _policy_action_probs(algo, n_eval_prob_mc)
        pi_t = (1.0 - k * eps) * pi_t + eps
        pi_t /= pi_t.sum()
        pi_hist[t] = pi_t

        a = actions[t]
        denom = max(float(behavior_probs[t, a]), eps)
        one_step_ratio = float(pi_t[a]) / denom
        if weight_mode == "one_step":
            weights[t] = one_step_ratio
        else:
            cumulative_ratio *= one_step_ratio
            weights[t] = cumulative_ratio

        algo.update(int(a), float(rewards[t]))

    return weights, pi_hist

# One-step uses the current ratio only; cumulative multiplies ratios over time.

def compute_history_dependent_ts_weights(
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_policy: np.ndarray,
    prior_mean: float,
    prior_var: float,
    obs_sigma: float | np.ndarray,
    n_ts_prob_mc: int,
    rng: np.random.Generator,
    env_type: str = "normal",
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    weight_mode: str = "one_step",
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    behavior_policy = np.asarray(behavior_policy, dtype=np.float64)

    n = actions.shape[0]
    k = behavior_policy.shape[0]
    if env_type == "bernoulli":
        alpha = np.full(k, prior_alpha, dtype=np.float64)
        beta = np.full(k, prior_beta, dtype=np.float64)
    else:
        obs_sigma = _as_positive_std_array(obs_sigma, k, "obs_sigma")
        obs_var = obs_sigma ** 2
        post_mean = np.full(k, prior_mean, dtype=np.float64)
        post_var = np.full(k, prior_var, dtype=np.float64)

    weights = np.empty(n, dtype=np.float64)
    pi_hist = np.empty((n, k), dtype=np.float64)
    cumulative_ratio = 1.0

    if weight_mode not in {"one_step", "cumulative"}:
        raise ValueError("weight_mode must be either 'one_step' or 'cumulative'.")

    for t in range(n):
        if env_type == "bernoulli":
            theta = rng.beta(alpha, beta, size=(n_ts_prob_mc, k))
        else:
            theta = rng.normal(
                loc=post_mean,
                scale=np.sqrt(post_var),
                size=(n_ts_prob_mc, k),
            )
        chosen = np.argmax(theta, axis=1)
        counts = np.bincount(chosen, minlength=k).astype(np.float64)

        pi_t = counts / n_ts_prob_mc
        pi_t = (1.0 - k * eps) * pi_t + eps
        pi_t = pi_t / pi_t.sum()

        pi_hist[t] = pi_t

        a = actions[t]
        one_step_ratio = pi_t[a] / behavior_policy[a]
        if weight_mode == "one_step":
            weights[t] = one_step_ratio
        else:
            cumulative_ratio *= one_step_ratio
            weights[t] = cumulative_ratio

        r = rewards[t]
        if env_type == "bernoulli":
            alpha[a] += r
            beta[a] += 1.0 - r
        else:
            v0 = post_var[a]
            m0 = post_mean[a]
            sig2 = obs_var[a]

            v1 = 1.0 / (1.0 / v0 + 1.0 / sig2)
            m1 = v1 * (m0 / v0 + r / sig2)

            post_var[a] = v1
            post_mean[a] = m1

    return weights, pi_hist


def _mean_and_se(xs: np.ndarray) -> tuple[float, float]:
    xs = np.asarray(xs, dtype=np.float64)
    mean_x, se = _mean_and_se_numba(xs)
    return float(mean_x), float(se)


def ipw_wald_interval(
    weights: np.ndarray,
    rewards: np.ndarray,
    conf_level: float,
) -> tuple[float, float]:
    xs = np.asarray(weights, dtype=np.float64) * np.asarray(rewards, dtype=np.float64)
    mean_x, se_x = _mean_and_se(xs)
    z = norm.ppf(0.5 + conf_level / 2.0)
    return mean_x - z * se_x, mean_x + z * se_x


def weighted_t_test_interval(
    weights: np.ndarray,
    rewards: np.ndarray,
    conf_level: float,
) -> tuple[float, float]:
    xs = np.asarray(weights, dtype=np.float64) * np.asarray(rewards, dtype=np.float64)
    mean_x, se_x = _mean_and_se(xs)
    if xs.size < 2:
        return mean_x, mean_x
    tcrit = t.ppf(0.5 + conf_level / 2.0, df=xs.size - 1)
    return mean_x - tcrit * se_x, mean_x + tcrit * se_x


def naive_t_test_interval(
    rewards: np.ndarray,
    conf_level: float,
) -> tuple[float, float]:
    xs = np.asarray(rewards, dtype=np.float64)
    mean_x, se_x = _mean_and_se(xs)
    if xs.size < 2:
        return mean_x, mean_x
    tcrit = t.ppf(0.5 + conf_level / 2.0, df=xs.size - 1)
    return mean_x - tcrit * se_x, mean_x + tcrit * se_x

# CADR rescales weighted rewards using a running variance estimate before inference.

def cadr_interval(
    rewards: np.ndarray,
    weights: np.ndarray,
    conf_level: float,
    min_samples: int = 30,
) -> tuple[float, float]:
    rewards = np.asarray(rewards, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    d0 = weights * rewards
    sigma_inv = _running_cadr_sigma_inv_numba(d0, min_samples=min_samples)
    d = sigma_inv * d0

    _, se_d = _mean_and_se(d)
    z = norm.ppf(0.5 + conf_level / 2.0)

    sigma_bar = float(np.mean(1.0 / sigma_inv))
    half_width = z * se_d * sigma_bar
    center = float(np.mean(d0))
    return center - half_width, center + half_width


def fit_arm_reward_model(
    actions: np.ndarray,
    rewards: np.ndarray,
    k: int,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    obs_sigma: float | np.ndarray = 1.0,
    env_type: str = "normal",
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if env_type == "bernoulli":
        counts = np.zeros(k, dtype=np.float64)
        successes = np.zeros(k, dtype=np.float64)
        for i in range(actions.shape[0]):
            a = actions[i]
            counts[a] += 1.0
            successes[a] += rewards[i]
        return (successes + prior_alpha) / (counts + prior_alpha + prior_beta)
    obs_sigma = _as_positive_std_array(obs_sigma, k, "obs_sigma")
    obs_var = obs_sigma ** 2

    return _fit_arm_reward_model_numba(actions, rewards, obs_var, prior_mean, prior_var, k)


def dr_scores(
    actions: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    pi_hist: np.ndarray,
    qhat: np.ndarray,
) -> np.ndarray:
    direct_part = pi_hist @ qhat
    residual_part = weights * (rewards - qhat[actions])
    return direct_part + residual_part


def dr_wald_interval(
    actions: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    pi_hist: np.ndarray,
    conf_level: float,
    reward_model_prior_mean: float = 0.0,
    reward_model_prior_var: float = 1.0,
    reward_model_obs_sigma: float | np.ndarray = 1.0,
    env_type: str = "normal",
    reward_model_prior_alpha: float = 1.0,
    reward_model_prior_beta: float = 1.0,
) -> tuple[float, float]:
    k = pi_hist.shape[1]
    qhat = fit_arm_reward_model(
        actions=actions,
        rewards=rewards,
        k=k,
        prior_mean=reward_model_prior_mean,
        prior_var=reward_model_prior_var,
        obs_sigma=reward_model_obs_sigma,
        env_type=env_type,
        prior_alpha=reward_model_prior_alpha,
        prior_beta=reward_model_prior_beta,
    )

    phi = dr_scores(
        actions=actions,
        rewards=rewards,
        weights=weights,
        pi_hist=pi_hist,
        qhat=qhat,
    )

    mean_phi, se_phi = _mean_and_se(phi)
    z = norm.ppf(0.5 + conf_level / 2.0)
    return mean_phi - z * se_phi, mean_phi + z * se_phi


def compress_datagen(weights, rewards):
    counts = {}
    for w, r in zip(weights, rewards):
        key = (float(w), float(r))
        counts[key] = counts.get(key, 0) + 1

    items = [(c, w, r) for (w, r), c in counts.items()]

    def datagen():
        for c, w, r in items:
            yield c, w, r

    return datagen


def elfcb_estimate_repo(datagen, wmin, wmax, rmin=0.0, rmax=1.0):
    assert wmin >= 0.0
    assert wmin < 1.0
    assert wmax > 1.0
    assert rmax >= rmin

    num = sum(c for c, _, _ in datagen())
    if num < 1:
        raise ValueError("Need at least one observation.")

    def sumofw(beta):
        return sum(
            (c * w) / ((w - 1.0) * beta + num)
            for c, w, _ in datagen()
            if c > 0
        )

    def graddualobjective(beta):
        return sum(
            c * (w - 1.0) / ((w - 1.0) * beta + num)
            for c, w, _ in datagen()
            if c > 0
        )

    betamax = min(
        ((num - c) / (1.0 - w) for c, w, _ in datagen() if w < 1.0 and c > 0),
        default=num / (1.0 - wmin),
    )
    betamax = min(betamax, num / (1.0 - wmin))

    betamin = max(
        ((num - c) / (1.0 - w) for c, w, _ in datagen() if w > 1.0 and c > 0),
        default=num / (1.0 - wmax),
    )
    betamin = max(betamin, num / (1.0 - wmax))

    gradmin = graddualobjective(betamin)
    gradmax = graddualobjective(betamax)

    if gradmin * gradmax < 0.0:
        betastar = brentq(graddualobjective, betamin, betamax)
    elif gradmin < 0.0:
        betastar = betamin
    else:
        betastar = betamax

    remw = max(0.0, 1.0 - sumofw(betastar))
    vhat = 0.0

    for c, w, r in datagen():
        if c > 0:
            vhat += w * r * c / ((w - 1.0) * betastar + num)

    vmin = vhat + remw * rmin
    vmax = vhat + remw * rmax
    vhat += remw * (rmin + rmax) / 2.0

    qfunc = lambda c, w, r: c / (num + betastar * (w - 1.0))
    return {
        "betastar": betastar,
        "vmin": vmin,
        "vmax": vmax,
        "num": num,
        "vhat": vhat,
        "qfunc": qfunc,
    }


def elfcb_confidence_interval_repo(
    datagen,
    wmin,
    wmax,
    alpha_mis=0.05,
    rmin=0.0,
    rmax=1.0,
    show_cvxopt_progress=False,
):
    qmle = elfcb_estimate_repo(datagen, wmin=wmin, wmax=wmax, rmin=rmin, rmax=rmax)
    num = qmle["num"]
    if num < 2:
        return (rmin, rmax), (None, None)

    betamle = qmle["betastar"]
    delta = 0.5 * f.isf(q=alpha_mis, dfn=1, dfd=num - 1)

    sumwsq = sum(c * w * w for c, w, _ in datagen())
    wscale = max(1.0, np.sqrt(sumwsq / num))
    rscale = max(1.0, abs(rmin), abs(rmax))

    tiny = 1e-5
    logtiny = math.log(tiny)

    def logstar(x):
        if x > tiny:
            return math.log(x)
        xt = x / tiny
        return -1.5 + logtiny + 2.0 * xt - 0.5 * xt * xt

    def jaclogstar(x):
        if x > tiny:
            return 1.0 / x
        return (2.0 - (x / tiny)) / tiny

    def hesslogstar(x):
        if x > tiny:
            return -1.0 / (x * x)
        return -1.0 / (tiny * tiny)

    def dualobjective(p, sign):
        gamma, beta = float(p[0]), float(p[1])
        logcost = -delta
        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                mledenom = num + betamle * (w - 1.0)
                logcost += c * (logstar(denom) - logstar(mledenom))
        assert n == num
        if n > 0:
            logcost /= n
        return (-n * math.exp(logcost) + gamma + beta / wscale) / rscale

    def jacdualobjective(p, sign):
        gamma, beta = float(p[0]), float(p[1])
        logcost = -delta
        jac = np.zeros(2, dtype=np.float64)
        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                mledenom = num + betamle * (w - 1.0)
                logcost += c * (logstar(denom) - logstar(mledenom))
                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wscale)
        assert n == num
        if n > 0:
            logcost /= n
            jac /= n
        jac *= -(n / rscale) * math.exp(logcost)
        jac[0] += 1.0 / rscale
        jac[1] += 1.0 / (wscale * rscale)
        return jac

    def hessdualobjective(p, sign):
        gamma, beta = float(p[0]), float(p[1])
        logcost = -delta
        jac = np.zeros(2, dtype=np.float64)
        hess = np.zeros((2, 2), dtype=np.float64)
        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                mledenom = num + betamle * (w - 1.0)
                logcost += c * (logstar(denom) - logstar(mledenom))
                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wscale)
                hesslogcost = c * hesslogstar(denom)
                hess[0, 0] += hesslogcost
                hess[0, 1] += hesslogcost * (w / wscale)
                hess[1, 1] += hesslogcost * (w / wscale) * (w / wscale)
        assert n == num
        if n > 0:
            logcost /= n
            jac /= n
            hess /= n
        hess[1, 0] = hess[0, 1]
        hess += np.outer(jac, jac)
        hess *= -(n / rscale) * math.exp(logcost)
        return hess

    conse = np.array(
        [[1.0, w / wscale] for w in (wmin, wmax) for r in (rmin, rmax)],
        dtype=np.float64,
    )

    retvals = []
    easybounds = [
        (qmle["vmin"] <= rmin + tiny, rmin),
        (qmle["vmax"] >= rmax - tiny, rmax),
    ]

    solvers.options["show_progress"] = show_cvxopt_progress

    for what in range(2):
        if easybounds[what][0]:
            retvals.append((easybounds[what][1], None))
            continue

        sign = 1 - 2 * what
        d = np.array(
            [-sign * w * r + tiny for w in (wmin, wmax) for r in (rmin, rmax)],
            dtype=np.float64,
        )
        minsr = min(sign * rmin, sign * rmax)
        gamma0 = num - qmle["betastar"] + 2.0 * tiny
        beta0 = wscale * (qmle["betastar"] - (1.0 + 1.0 / wscale) * minsr)
        x0 = np.array([gamma0, beta0], dtype=np.float64)

        def F(x=None, z=None):
            if x is None:
                return 0, matrix(x0)
            p = np.reshape(np.array(x), -1)
            fval = dualobjective(p, sign)
            jf = jacdualobjective(p, sign)
            df = matrix(jf, (1, 2))
            if z is None:
                return fval, df
            hf = z[0] * hessdualobjective(p, sign)
            h = matrix(hf, hf.shape)
            return fval, df, h

        soln = solvers.cp(F, G=-matrix(conse, conse.shape), h=-matrix(d))

        if soln["status"] != "optimal":
            raise RuntimeError(f"cvxopt.cp failed with status={soln['status']}")

        xstar = np.reshape(np.array(soln["x"]), -1)
        fstar = float(soln["primal objective"])
        gammastar = float(xstar[0])
        betastar = float(xstar[1]) / wscale
        kappastar = (-rscale * fstar + gammastar + betastar) / num

        qfunc = lambda c, w, r, kappa=kappastar, gamma=gammastar, beta=betastar, s=sign: (
            kappa * c / (gamma + (beta + s * r) * w)
        )

        vbound = -sign * rscale * fstar
        retvals.append(
            (
                vbound,
                {
                    "gammastar": gammastar,
                    "betastar": betastar,
                    "kappastar": kappastar,
                    "qfunc": qfunc,
                },
            )
        )

    return (retvals[0][0], retvals[1][0]), (retvals[0][1], retvals[1][1])


@dataclass
class BaselineIntervals:
    elfcb: tuple[float, float]
    ipw: tuple[float, float]
    weighted_t_test: tuple[float, float] | None
    cadr: tuple[float, float]
    dr: tuple[float, float]

# End-to-end baseline evaluation: weights, ELFCB, IPW, CADR, DR, and optional t-test.

def compute_all_intervals(
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
    eval_algo_builder,
    conf_level: float,
    n_eval_prob_mc: int,
    eval_seed: int,
    reward_model_prior_mean: float,
    reward_model_prior_var: float,
    reward_model_obs_sigma: float | np.ndarray,
    env_type: str = "normal",
    reward_model_prior_alpha: float = 1.0,
    reward_model_prior_beta: float = 1.0,
    weight_mode: str = "one_step",
    prob_eps: float = 1e-8,
    cadr_min_samples: int = 30,
    show_cvxopt_progress: bool = False,
    include_weighted_t_test: bool = False,
) -> BaselineIntervals:
    weights, pi_hist = compute_history_dependent_policy_weights(
        actions=actions,
        rewards=rewards,
        behavior_probs=behavior_probs,
        eval_algo_builder=eval_algo_builder,
        n_eval_prob_mc=n_eval_prob_mc,
        eval_seed=eval_seed,
        weight_mode=weight_mode,
        eps=prob_eps,
    )

    datagen = compress_datagen(weights, rewards)
    eps = 1e-10
    wmin = float(min(np.min(weights), 1.0 - eps))
    wmax = float(max(np.max(weights), 1.0 + eps))
    rmin = float(np.min(rewards))
    rmax = float(np.max(rewards))

    try:
        elfcb, _ = elfcb_confidence_interval_repo(
            datagen=datagen,
            wmin=wmin,
            wmax=wmax,
            alpha_mis=1.0 - conf_level,
            rmin=rmin,
            rmax=rmax,
            show_cvxopt_progress=show_cvxopt_progress,
        )
    except RuntimeError:
        elfcb = (float("nan"), float("nan"))
    ipw = ipw_wald_interval(weights, rewards, conf_level)
    weighted_t_test = None
    if include_weighted_t_test:
        weighted_t_test = weighted_t_test_interval(weights, rewards, conf_level)
    cadr = cadr_interval(rewards, weights, conf_level, min_samples=cadr_min_samples)
    dr = dr_wald_interval(
        actions=actions,
        rewards=rewards,
        weights=weights,
        pi_hist=pi_hist,
        conf_level=conf_level,
        reward_model_prior_mean=reward_model_prior_mean,
        reward_model_prior_var=reward_model_prior_var,
        reward_model_obs_sigma=reward_model_obs_sigma,
        env_type=env_type,
        reward_model_prior_alpha=reward_model_prior_alpha,
        reward_model_prior_beta=reward_model_prior_beta,
    )
    return BaselineIntervals(
        elfcb=elfcb,
        ipw=ipw,
        weighted_t_test=weighted_t_test,
        cadr=cadr,
        dr=dr,
    )
