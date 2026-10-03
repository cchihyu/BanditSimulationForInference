from __future__ import annotations

from dataclasses import dataclass
import math
import copy
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import f, norm, t


@dataclass
class ContextualBaselineIntervals:
    elfcb: tuple[float, float]
    ipw: tuple[float, float]
    weighted_t_test: tuple[float, float] | None
    cadr: tuple[float, float]
    dr: tuple[float, float]


class PerActionLinearRewardModel:
    """
    Lightweight contextual reward model for DR.

    Fits one unpenalized linear regression per action:
        q_hat(a, x) = beta_a^T [1, x].
    """

    def __init__(self):
        self.coefs_: np.ndarray | None = None

    def fit(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        n_actions: int,
    ) -> "PerActionLinearRewardModel":
        contexts = _as_2d_contexts(contexts)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float64)
        _validate_1d_logs(actions, rewards, contexts.shape[0])

        design = _with_intercept(contexts)
        n_features = design.shape[1]
        coefs = np.zeros((n_actions, n_features), dtype=np.float64)
        global_mean = float(rewards.mean()) if rewards.size else 0.0

        for action in range(n_actions):
            idx = actions == action
            if not np.any(idx):
                coefs[action, 0] = global_mean
                continue

            x_a = design[idx]
            y_a = rewards[idx]
            lhs = x_a.T @ x_a
            rhs = x_a.T @ y_a
            try:
                coefs[action] = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                coefs[action] = np.linalg.pinv(lhs) @ rhs

        self.coefs_ = coefs
        return self

    def predict_all(self, contexts: np.ndarray) -> np.ndarray:
        if self.coefs_ is None:
            raise RuntimeError("Reward model must be fit before prediction.")
        design = _with_intercept(_as_2d_contexts(contexts))
        return design @ self.coefs_.T

    def predict(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        q_all = self.predict_all(contexts)
        actions = np.asarray(actions, dtype=np.int64)
        return q_all[np.arange(actions.shape[0]), actions]


class PerActionLogisticRewardModel:
    """
    Per-action logistic regression reward model for contextual DR.

    Fits q_hat(a, x) = P(R=1 | A=a, X=x) with one logistic model per action.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-8):
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coefs_: np.ndarray | None = None

    def fit(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        n_actions: int,
    ) -> "PerActionLogisticRewardModel":
        contexts = _as_2d_contexts(contexts)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float64)
        _validate_1d_logs(actions, rewards, contexts.shape[0])
        if np.any((rewards < 0.0) | (rewards > 1.0)):
            raise ValueError("Logistic rewards must lie in [0, 1].")
        if not np.allclose(rewards, np.round(rewards), atol=1e-8):
            raise ValueError("Logistic rewards must be binary 0/1 values.")

        design = _with_intercept(contexts)
        n_features = design.shape[1]
        coefs = np.zeros((n_actions, n_features), dtype=np.float64)
        global_rate = _clip_prob(float(rewards.mean())) if rewards.size else 0.5

        for action in range(n_actions):
            idx = actions == action
            if not np.any(idx):
                coefs[action, 0] = _logit(global_rate)
                continue

            x_a = design[idx]
            y_a = rewards[idx]
            if np.all(y_a == y_a[0]):
                rate = _clip_prob((float(y_a.sum()) + 0.5) / (y_a.size + 1.0))
                coefs[action, 0] = _logit(rate)
                continue

            coefs[action] = self._fit_irls(x_a, y_a)

        self.coefs_ = coefs
        return self

    def predict_all(self, contexts: np.ndarray) -> np.ndarray:
        if self.coefs_ is None:
            raise RuntimeError("Reward model must be fit before prediction.")
        design = _with_intercept(_as_2d_contexts(contexts))
        return expit(np.clip(design @ self.coefs_.T, -35.0, 35.0))

    def predict(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        q_all = self.predict_all(contexts)
        actions = np.asarray(actions, dtype=np.int64)
        return q_all[np.arange(actions.shape[0]), actions]

    def _fit_irls(self, x_design: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        n_features = x_design.shape[1]
        beta = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.max_iter):
            eta = np.clip(x_design @ beta, -35.0, 35.0)
            probs = expit(eta)
            weights = np.maximum(probs * (1.0 - probs), 1e-10)
            gradient = x_design.T @ (rewards - probs)
            hessian = (x_design.T * weights) @ x_design
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(hessian) @ gradient
            beta_next = beta + step
            if np.linalg.norm(step) <= self.tol * (1.0 + np.linalg.norm(beta)):
                beta = beta_next
                break
            beta = beta_next
        return beta


def _mean_and_se(xs: np.ndarray) -> tuple[float, float]:
    xs = np.asarray(xs, dtype=np.float64)
    mean_x = float(xs.mean())
    if xs.size < 2:
        return mean_x, 0.0
    se_x = float(xs.std(ddof=1) / np.sqrt(xs.size))
    return mean_x, se_x


def _clip_prob(prob: float) -> float:
    return float(np.clip(prob, 1e-8, 1.0 - 1e-8))


def _logit(prob: float) -> float:
    prob = _clip_prob(prob)
    return float(np.log(prob / (1.0 - prob)))


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


def contextual_cadr_sigmas(
    contexts, actions, rewards, behavior_probs, target_probs,
    current_behavior_probs, min_samples=30, variance_floor=1e-12,
    warmup_sigma=1.0,
):
    """Estimate current IPW-score SDs from past observations.

    current_behavior_probs(t, past_contexts) must return (t, K) probabilities
    from the logging policy state BEFORE observation t, without updating it.
    This callback also controls probability Monte Carlo settings for TS.
    The target must be a fixed contextual policy. Stored propensities alone
    cannot reconstruct adaptive logging policies at historical contexts.
    """
    contexts, actions, rewards, behavior_probs = _validate_contextual_inputs(
        contexts, actions, rewards, behavior_probs)
    target_probs = np.asarray(target_probs, dtype=float)
    if target_probs.shape != behavior_probs.shape:
        raise ValueError("target_probs must have shape (n, K).")
    if min_samples < 1 or variance_floor <= 0 or warmup_sigma <= 0:
        raise ValueError("Warm-up length and variance scales must be positive.")
    n = len(rewards)
    sigmas = np.full(n, warmup_sigma, dtype=float)
    for t in range(min_samples, n):
        current = np.asarray(current_behavior_probs(t, contexts[:t].copy()), dtype=float)
        if (current.shape != (t, behavior_probs.shape[1])
                or np.any(~np.isfinite(current)) or np.any(current <= 0)
                or not np.allclose(current.sum(axis=1), 1.0)):
            raise ValueError("Current logging probabilities must be positive normalized (t, K) rows.")
        rows = np.arange(t)
        gt = current[rows, actions[:t]]
        gs = behavior_probs[rows, actions[:t]]
        score = target_probs[rows, actions[:t]] * rewards[:t] / gt
        ratio = gt / gs
        variance = np.mean(ratio * score**2) - np.mean(ratio * score)**2
        sigmas[t] = np.sqrt(max(float(variance), variance_floor))
    return sigmas


def contextual_cadr_sigmas_static(
    actions, rewards, behavior_probs, target_probs, min_samples=30,
    variance_floor=1e-12, warmup_sigma=1.0,
):
    """O(n) CADR conditional-scale estimate for a fixed logging policy."""
    actions=np.asarray(actions,dtype=int);rewards=np.asarray(rewards,dtype=float)
    behavior_probs=np.asarray(behavior_probs,dtype=float);target_probs=np.asarray(target_probs,dtype=float)
    if behavior_probs.shape!=target_probs.shape or behavior_probs.shape[0]!=len(rewards):
        raise ValueError('Probability histories must have matching (n, K) shapes.')
    chosen=np.arange(len(rewards)),actions
    scores=target_probs[chosen]*rewards/behavior_probs[chosen]
    sigmas=np.full(len(rewards),warmup_sigma,dtype=float);running_sum=0.;running_sumsq=0.
    for t,score in enumerate(scores):
        if t>=min_samples:
            variance=running_sumsq/t-(running_sum/t)**2
            sigmas[t]=np.sqrt(max(float(variance),variance_floor))
        running_sum+=score;running_sumsq+=score*score
    return sigmas


def cadr_interval(
    rewards: np.ndarray,
    weights: np.ndarray,
    conf_level: float,
    min_samples: int = 30,
    *,
    conditional_sigmas: np.ndarray | None = None,
) -> tuple[float, float]:
    """Stabilized IPW interval; requires past-measurable conditional SDs.

    min_samples is retained for API compatibility; warm-up is performed by
    contextual_cadr_sigmas, not by a running variance of historical scores.
    """
    if conditional_sigmas is None:
        raise ValueError("CADR requires conditional_sigmas or current logging-policy evaluations via the wrapper.")
    rewards = np.asarray(rewards, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sigmas = np.asarray(conditional_sigmas, dtype=float)
    if (rewards.ndim != 1 or not rewards.size or weights.shape != rewards.shape
            or sigmas.shape != rewards.shape or np.any(~np.isfinite(sigmas))
            or np.any(sigmas <= 0) or not 0 < conf_level < 1):
        raise ValueError("Invalid scores, confidence level, or conditional standard deviations.")
    inverse = 1.0 / sigmas
    center = float(np.sum(inverse * weights * rewards) / inverse.sum())
    gamma = 1.0 / inverse.mean()
    half_width = norm.ppf(0.5 + conf_level / 2.0) * gamma / np.sqrt(rewards.size)
    return center - half_width, center + half_width


def compress_datagen(weights: np.ndarray, rewards: np.ndarray):
    counts: dict[tuple[float, float], int] = {}
    for weight, reward in zip(weights, rewards):
        key = (float(weight), float(reward))
        counts[key] = counts.get(key, 0) + 1
    items = [(count, weight, reward) for (weight, reward), count in counts.items()]

    def datagen():
        yield from items

    return datagen


def _elfcb_estimate(
    datagen,
    wmin: float,
    wmax: float,
    rmin: float = 0.0,
    rmax: float = 1.0,
) -> dict[str, Any]:
    if wmin < 0.0 or wmin >= 1.0:
        raise ValueError("wmin must lie in [0, 1).")
    if wmax <= 1.0:
        raise ValueError("wmax must be greater than 1.")
    if rmax < rmin:
        raise ValueError("rmax must be at least rmin.")

    num = sum(count for count, _, _ in datagen())
    if num < 1:
        raise ValueError("Need at least one observation.")

    def sumofw(beta: float) -> float:
        return sum(
            (count * weight) / ((weight - 1.0) * beta + num)
            for count, weight, _ in datagen()
            if count > 0
        )

    def graddualobjective(beta: float) -> float:
        return sum(
            count * (weight - 1.0) / ((weight - 1.0) * beta + num)
            for count, weight, _ in datagen()
            if count > 0
        )

    betamax = min(
        ((num - count) / (1.0 - weight) for count, weight, _ in datagen() if weight < 1.0 and count > 0),
        default=num / (1.0 - wmin),
    )
    betamax = min(betamax, num / (1.0 - wmin))

    betamin = max(
        ((num - count) / (1.0 - weight) for count, weight, _ in datagen() if weight > 1.0 and count > 0),
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
    for count, weight, reward in datagen():
        if count > 0:
            vhat += weight * reward * count / ((weight - 1.0) * betastar + num)

    vmin = vhat + remw * rmin
    vmax = vhat + remw * rmax
    vhat += remw * (rmin + rmax) / 2.0

    return {
        "betastar": betastar,
        "vmin": vmin,
        "vmax": vmax,
        "num": num,
        "vhat": vhat,
    }


def _elfcb_confidence_interval(
    datagen,
    wmin: float,
    wmax: float,
    alpha_mis: float,
    rmin: float,
    rmax: float,
    show_cvxopt_progress: bool,
) -> tuple[float, float]:
    try:
        from cvxopt import matrix, solvers
    except ImportError as exc:
        raise RuntimeError("ELFCB requires cvxopt.") from exc

    qmle = _elfcb_estimate(datagen, wmin=wmin, wmax=wmax, rmin=rmin, rmax=rmax)
    num = qmle["num"]
    if num < 2:
        return rmin, rmax

    betamle = qmle["betastar"]
    delta = 0.5 * f.isf(q=alpha_mis, dfn=1, dfd=num - 1)

    sumwsq = sum(count * weight * weight for count, weight, _ in datagen())
    wscale = max(1.0, np.sqrt(sumwsq / num))
    rscale = max(1.0, abs(rmin), abs(rmax))

    tiny = 1e-5
    logtiny = math.log(tiny)

    def logstar(x: float) -> float:
        if x > tiny:
            return math.log(x)
        xt = x / tiny
        return -1.5 + logtiny + 2.0 * xt - 0.5 * xt * xt

    def jaclogstar(x: float) -> float:
        if x > tiny:
            return 1.0 / x
        return (2.0 - (x / tiny)) / tiny

    def hesslogstar(x: float) -> float:
        if x > tiny:
            return -1.0 / (x * x)
        return -1.0 / (tiny * tiny)

    def dualobjective(p: np.ndarray, sign: int) -> float:
        gamma, beta = float(p[0]), float(p[1])
        logcost = -delta
        n = 0
        for count, weight, reward in datagen():
            if count > 0:
                n += count
                denom = gamma + (beta + sign * wscale * reward) * (weight / wscale)
                mledenom = num + betamle * (weight - 1.0)
                logcost += count * (logstar(denom) - logstar(mledenom))
        if n != num:
            raise RuntimeError("ELFCB datagen returned inconsistent counts.")
        if n > 0:
            logcost /= n
        return (-n * math.exp(logcost) + gamma + beta / wscale) / rscale

    def jacdualobjective(p: np.ndarray, sign: int) -> np.ndarray:
        gamma, beta = float(p[0]), float(p[1])
        logcost = -delta
        jac = np.zeros(2, dtype=np.float64)
        n = 0
        for count, weight, reward in datagen():
            if count > 0:
                n += count
                denom = gamma + (beta + sign * wscale * reward) * (weight / wscale)
                mledenom = num + betamle * (weight - 1.0)
                logcost += count * (logstar(denom) - logstar(mledenom))
                jaclogcost = count * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (weight / wscale)
        if n != num:
            raise RuntimeError("ELFCB datagen returned inconsistent counts.")
        if n > 0:
            logcost /= n
            jac /= n
        jac *= -(n / rscale) * math.exp(logcost)
        jac[0] += 1.0 / rscale
        jac[1] += 1.0 / (wscale * rscale)
        return jac

    def hessdualobjective(p: np.ndarray, sign: int) -> np.ndarray:
        gamma, beta = float(p[0]), float(p[1])
        logcost = -delta
        jac = np.zeros(2, dtype=np.float64)
        hess = np.zeros((2, 2), dtype=np.float64)
        n = 0
        for count, weight, reward in datagen():
            if count > 0:
                n += count
                denom = gamma + (beta + sign * wscale * reward) * (weight / wscale)
                mledenom = num + betamle * (weight - 1.0)
                logcost += count * (logstar(denom) - logstar(mledenom))
                jaclogcost = count * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (weight / wscale)
                hesslogcost = count * hesslogstar(denom)
                hess[0, 0] += hesslogcost
                hess[0, 1] += hesslogcost * (weight / wscale)
                hess[1, 1] += hesslogcost * (weight / wscale) * (weight / wscale)
        if n != num:
            raise RuntimeError("ELFCB datagen returned inconsistent counts.")
        if n > 0:
            logcost /= n
            jac /= n
            hess /= n
        hess[1, 0] = hess[0, 1]
        hess += np.outer(jac, jac)
        hess *= -(n / rscale) * math.exp(logcost)
        return hess

    constraints = np.array(
        [[1.0, weight / wscale] for weight in (wmin, wmax) for _ in (rmin, rmax)],
        dtype=np.float64,
    )
    easybounds = [
        (qmle["vmin"] <= rmin + tiny, rmin),
        (qmle["vmax"] >= rmax - tiny, rmax),
    ]

    solvers.options["show_progress"] = show_cvxopt_progress
    retvals = []
    for what in range(2):
        if easybounds[what][0]:
            retvals.append(easybounds[what][1])
            continue

        sign = 1 - 2 * what
        d = np.array(
            [-sign * weight * reward + tiny for weight in (wmin, wmax) for reward in (rmin, rmax)],
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
            return fval, df, matrix(hf, hf.shape)

        soln = solvers.cp(F, G=-matrix(constraints, constraints.shape), h=-matrix(d))
        if soln["status"] != "optimal":
            raise RuntimeError(f"cvxopt.cp failed with status={soln['status']}")
        retvals.append(-(sign * rscale * float(soln["primal objective"])))

    return retvals[0], retvals[1]


def _as_2d_contexts(contexts: np.ndarray) -> np.ndarray:
    contexts = np.asarray(contexts, dtype=np.float64)
    if contexts.ndim == 1:
        contexts = contexts.reshape(-1, 1)
    if contexts.ndim != 2:
        raise ValueError("contexts must be a 1D or 2D numeric array.")
    return contexts


def _with_intercept(contexts: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(contexts.shape[0], dtype=np.float64), contexts])


def _validate_1d_logs(actions: np.ndarray, rewards: np.ndarray, n: int) -> None:
    if actions.ndim != 1:
        raise ValueError("actions must be 1D.")
    if rewards.ndim != 1:
        raise ValueError("rewards must be 1D.")
    if actions.shape[0] != n or rewards.shape[0] != n:
        raise ValueError("contexts, actions, and rewards must have the same length.")


def _validate_contextual_inputs(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    contexts = _as_2d_contexts(contexts)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    behavior_probs = np.asarray(behavior_probs, dtype=np.float64)
    _validate_1d_logs(actions, rewards, contexts.shape[0])

    if behavior_probs.ndim != 2:
        raise ValueError("behavior_probs must have shape (T, K).")
    if behavior_probs.shape[0] != contexts.shape[0]:
        raise ValueError("behavior_probs must have one row per logged observation.")
    if np.any(actions < 0) or np.any(actions >= behavior_probs.shape[1]):
        raise ValueError("actions must be valid columns of behavior_probs.")
    if np.any(behavior_probs <= 0.0):
        raise ValueError("behavior_probs must be strictly positive for overlap.")
    row_sums = behavior_probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("Each row of behavior_probs must sum to 1.")

    return contexts, actions, rewards, behavior_probs


def _normalize_probs(probs: np.ndarray, n_actions: int, eps: float) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or probs.shape[0] != n_actions:
        raise ValueError(f"Policy probabilities must be a length-{n_actions} vector.")
    if np.any(~np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("Policy probabilities must be finite and non-negative.")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("Policy probabilities must have positive mass.")
    probs = probs / total
    probs = (1.0 - n_actions * eps) * probs + eps
    return probs / probs.sum()


def _history(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
    pi_hist: np.ndarray,
    t: int,
) -> dict[str, np.ndarray]:
    return {
        "contexts": contexts[:t],
        "actions": actions[:t],
        "rewards": rewards[:t],
        "behavior_probs": behavior_probs[:t],
        "target_probs": pi_hist[:t],
    }


def _call_policy_action_probs(
    eval_policy: Any,
    context: np.ndarray,
    history: dict[str, np.ndarray],
    n_actions: int,
) -> np.ndarray:
    if hasattr(eval_policy, "action_probs"):
        method = eval_policy.action_probs
        try:
            return method(context, history=history, n_actions=n_actions)
        except TypeError:
            try:
                return method(context, history)
            except TypeError:
                return method(context)

    if callable(eval_policy):
        try:
            return eval_policy(context, history=history, n_actions=n_actions)
        except TypeError:
            try:
                return eval_policy(context, history)
            except TypeError:
                return eval_policy(context)

    raise TypeError(
        "eval_policy must be callable or expose action_probs(context, history=...)."
    )


def _maybe_update_policy(
    eval_policy: Any,
    context: np.ndarray,
    action: int,
    reward: float,
) -> None:
    if not hasattr(eval_policy, "update"):
        return
    try:
        eval_policy.update(context, action, reward)
    except TypeError:
        eval_policy.update(action, reward)


def compute_contextual_policy_weights(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
    eval_policy: Any,
    weight_mode: str = "one_step",
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute contextual importance weights.

    `eval_policy` may be a callable or an object with
    `action_probs(context, history=..., n_actions=...)`. If it also has
    `update`, the logged action/reward is replayed after each probability call.
    """
    contexts, actions, rewards, behavior_probs = _validate_contextual_inputs(
        contexts, actions, rewards, behavior_probs
    )
    if weight_mode not in {"one_step", "cumulative"}:
        raise ValueError("weight_mode must be either 'one_step' or 'cumulative'.")

    n, n_actions = behavior_probs.shape
    weights = np.empty(n, dtype=np.float64)
    pi_hist = np.empty((n, n_actions), dtype=np.float64)
    cumulative_ratio = 1.0

    for t in range(n):
        hist = _history(contexts, actions, rewards, behavior_probs, pi_hist, t)
        pi_t = _call_policy_action_probs(eval_policy, contexts[t], hist, n_actions)
        pi_t = _normalize_probs(pi_t, n_actions=n_actions, eps=eps)
        pi_hist[t] = pi_t

        action = int(actions[t])
        ratio = float(pi_t[action]) / max(float(behavior_probs[t, action]), eps)
        if weight_mode == "one_step":
            weights[t] = ratio
        else:
            cumulative_ratio *= ratio
            weights[t] = cumulative_ratio

        _maybe_update_policy(eval_policy, contexts[t], action, float(rewards[t]))

    return weights, pi_hist


def _fit_reward_model(
    reward_model: Any,
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_actions: int,
) -> Any:
    if reward_model is None:
        reward_model = PerActionLinearRewardModel()
    if hasattr(reward_model, "fit"):
        try:
            return reward_model.fit(contexts, actions, rewards, n_actions=n_actions)
        except TypeError:
            return reward_model.fit(contexts, actions, rewards)
    return reward_model


def _predict_all_rewards(
    reward_model: Any,
    contexts: np.ndarray,
    n_actions: int,
) -> np.ndarray:
    if hasattr(reward_model, "predict_all"):
        q_all = reward_model.predict_all(contexts)
    elif callable(reward_model):
        q_all = reward_model(contexts)
    else:
        raise TypeError(
            "reward_model must be callable or expose predict_all(contexts)."
        )
    q_all = np.asarray(q_all, dtype=np.float64)
    if q_all.shape != (contexts.shape[0], n_actions):
        raise ValueError(f"Reward predictions must have shape {(contexts.shape[0], n_actions)}.")
    return q_all


def contextual_dr_scores(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    pi_hist: np.ndarray,
    reward_model: Any = None,
) -> np.ndarray:
    contexts = _as_2d_contexts(contexts)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    pi_hist = np.asarray(pi_hist, dtype=np.float64)
    _validate_1d_logs(actions, rewards, contexts.shape[0])

    if pi_hist.ndim != 2 or pi_hist.shape[0] != contexts.shape[0]:
        raise ValueError("pi_hist must have shape (T, K).")
    if weights.ndim != 1 or weights.shape[0] != contexts.shape[0]:
        raise ValueError("weights must have length T.")

    n_actions = pi_hist.shape[1]
    fitted_model = _fit_reward_model(
        reward_model, contexts, actions, rewards, n_actions=n_actions
    )
    q_all = _predict_all_rewards(fitted_model, contexts, n_actions=n_actions)
    q_logged = q_all[np.arange(contexts.shape[0]), actions]
    direct_part = np.sum(pi_hist * q_all, axis=1)
    residual_part = weights * (rewards - q_logged)
    return direct_part + residual_part


def contextual_dr_wald_interval(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    pi_hist: np.ndarray,
    conf_level: float,
    reward_model: Any = None,
) -> tuple[float, float]:
    phi = contextual_dr_scores(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        weights=weights,
        pi_hist=pi_hist,
        reward_model=reward_model,
    )
    mean_phi, se_phi = _mean_and_se(phi)
    z = norm.ppf(0.5 + conf_level / 2.0)
    return mean_phi - z * se_phi, mean_phi + z * se_phi


def contextual_dr_bootstrap_interval(
    contexts, actions, rewards, weights, pi_hist, conf_level,
    reward_model=None, bootstrap_reps=1000, bootstrap_seed=2026,
):
    """Normal interval using bootstrap SD (denominator B), per the appendix.

    Resample complete rows including stored target probabilities and weights.
    Refit a fresh copied reward model each time; never replay shuffled histories.
    Failed fits raise rather than silently discarding bootstrap replicates.
    """
    if bootstrap_reps < 2 or not 0 < conf_level < 1:
        raise ValueError("Need at least two bootstrap replicates and 0 < conf_level < 1.")
    arrays = [np.asarray(x) for x in (contexts, actions, rewards, weights, pi_hist)]
    n = len(arrays[2])
    if n < 2 or any(len(x) != n for x in arrays):
        raise ValueError("Need at least two aligned observations.")
    def estimate(data):
        model = copy.deepcopy(reward_model)
        if model is not None and not hasattr(model, "fit"):
            raise ValueError("Bootstrap reward models must expose fit for refitting.")
        value = float(np.mean(contextual_dr_scores(*data, reward_model=model)))
        if not np.isfinite(value):
            raise ValueError("Nonfinite bootstrap DR estimate.")
        return value
    center = estimate(arrays)
    rng = np.random.default_rng(bootstrap_seed)
    estimates = np.empty(bootstrap_reps)
    for b in range(bootstrap_reps):
        idx = rng.integers(n, size=n)
        try:
            estimates[b] = estimate([x[idx] for x in arrays])
        except Exception as exc:
            raise RuntimeError(f"DR bootstrap replicate {b} failed; no interval returned.") from exc
    half_width = norm.ppf(0.5 + conf_level / 2.0) * estimates.std(ddof=0)
    return center - half_width, center + half_width


def compute_all_contextual_intervals(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
    eval_policy: Any,
    conf_level: float,
    reward_model: Any = None,
    weight_mode: str = "one_step",
    prob_eps: float = 1e-8,
    cadr_min_samples: int = 30,
    show_cvxopt_progress: bool = False,
    include_weighted_t_test: bool = False,
    include_elfcb: bool = True,
    dr_ci_method: str = "bootstrap",
    dr_bootstrap_reps: int = 1000,
    dr_bootstrap_seed: int = 2026,
    cadr_conditional_sigmas: np.ndarray | None = None,
    cadr_current_behavior_probs: Any = None,
    cadr_variance_floor: float = 1e-12,
    cadr_warmup_sigma: float = 1.0,
    cadr_target_is_fixed: bool = True,
    allow_adaptive_cadr: bool = False,
    cadr_behavior_is_static: bool = False,
    include_cadr: bool = True,
) -> ContextualBaselineIntervals:
    contexts, actions, rewards, behavior_probs = _validate_contextual_inputs(
        contexts, actions, rewards, behavior_probs
    )
    weights, pi_hist = compute_contextual_policy_weights(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        behavior_probs=behavior_probs,
        eval_policy=eval_policy,
        weight_mode=weight_mode,
        eps=prob_eps,
    )

    datagen = compress_datagen(weights, rewards)
    eps = 1e-10
    wmin = float(min(np.min(weights), 1.0 - eps))
    wmax = float(max(np.max(weights), 1.0 + eps))
    rmin = float(np.min(rewards))
    rmax = float(np.max(rewards))

    if include_elfcb:
        try:
            elfcb = _elfcb_confidence_interval(
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
    else:
        elfcb = (float("nan"), float("nan"))

    ipw = ipw_wald_interval(weights, rewards, conf_level)
    weighted_t_test = (
        weighted_t_test_interval(weights, rewards, conf_level)
        if include_weighted_t_test
        else None
    )
    if dr_ci_method not in {"bootstrap", "wald"}:
        raise ValueError("dr_ci_method must be bootstrap or wald.")
    if weight_mode != "one_step":
        raise ValueError("DR/CADR wrapper supports one_step only; cumulative weights require sequential inference.")
    cadr = (float("nan"), float("nan"))
    if include_cadr:
        if not cadr_target_is_fixed and not allow_adaptive_cadr:
            raise ValueError("Set allow_adaptive_cadr=True to compute adaptive-target CADR.")
        if cadr_conditional_sigmas is None:
            if cadr_behavior_is_static:
                cadr_conditional_sigmas=contextual_cadr_sigmas_static(
                    actions,rewards,behavior_probs,pi_hist,cadr_min_samples,
                    cadr_variance_floor,cadr_warmup_sigma)
            elif cadr_current_behavior_probs is None:
                raise ValueError("Supply cadr_current_behavior_probs, cadr_conditional_sigmas, or disable CADR.")
            else:
                cadr_conditional_sigmas = contextual_cadr_sigmas(
                    contexts, actions, rewards, behavior_probs, pi_hist,
                    cadr_current_behavior_probs, cadr_min_samples,
                    cadr_variance_floor, cadr_warmup_sigma)
        cadr = cadr_interval(rewards, weights, conf_level,
                             conditional_sigmas=cadr_conditional_sigmas)
    dr_function = (contextual_dr_bootstrap_interval if dr_ci_method == "bootstrap"
                   else contextual_dr_wald_interval)
    dr_options = ({"bootstrap_reps": dr_bootstrap_reps, "bootstrap_seed": dr_bootstrap_seed}
                  if dr_ci_method == "bootstrap" else {})
    dr = dr_function(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        weights=weights,
        pi_hist=pi_hist,
        conf_level=conf_level,
        reward_model=reward_model,
        **dr_options,
    )
    return ContextualBaselineIntervals(
        elfcb=elfcb,
        ipw=ipw,
        weighted_t_test=weighted_t_test,
        cadr=cadr,
        dr=dr,
    )
