from __future__ import annotations

import numpy as np
from scipy.special import expit


class ContextualEpsilonGreedyPolicy:
    """
    Contextual epsilon-greedy policy for linear Gaussian or logistic rewards.

    The policy fits one unpenalized model per action from its own online
    history.
    """

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        epsilon: float = 0.1,
        reward_type: str = "linear_gaussian",
        include_intercept: bool = True,
        explore_untried: bool = True,
        max_iter: int = 50,
        tol: float = 1e-8,
        seed: int | None = None,
    ):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if context_dim < 0:
            raise ValueError("context_dim must be nonnegative.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1].")
        if reward_type not in {"linear_gaussian", "logistic_bernoulli"}:
            raise ValueError("reward_type must be 'linear_gaussian' or 'logistic_bernoulli'.")
        self.n_actions = int(n_actions)
        self.context_dim = int(context_dim)
        self.epsilon = float(epsilon)
        self.reward_type = reward_type
        self.include_intercept = bool(include_intercept)
        self.explore_untried = bool(explore_untried)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rng = np.random.default_rng(seed)
        self._contexts: list[list[np.ndarray]] = [[] for _ in range(self.n_actions)]
        self._rewards: list[list[float]] = [[] for _ in range(self.n_actions)]
        self._coefs = np.zeros((self.n_actions, self._feature_dim), dtype=np.float64)
        self._dirty = np.ones(self.n_actions, dtype=bool)

    @property
    def _feature_dim(self) -> int:
        return self.context_dim + int(self.include_intercept)

    @property
    def counts(self) -> np.ndarray:
        return np.array([len(xs) for xs in self._contexts], dtype=int)

    def action_probs(self, context: np.ndarray, history: dict[str, np.ndarray] | None = None) -> np.ndarray:
        counts = self.counts
        if self.explore_untried and np.any(counts == 0):
            probs = np.zeros(self.n_actions, dtype=np.float64)
            probs[counts == 0] = 1.0 / np.sum(counts == 0)
            return probs

        means = self.predict_means(context)
        best = int(np.argmax(means))
        probs = np.full(self.n_actions, self.epsilon / self.n_actions, dtype=np.float64)
        probs[best] += 1.0 - self.epsilon
        return probs

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError("action is out of range.")
        self._contexts[action].append(np.asarray(context, dtype=np.float64).reshape(-1))
        self._rewards[action].append(float(reward))
        self._dirty[action] = True

    def predict_means(self, context: np.ndarray) -> np.ndarray:
        for action in np.where(self._dirty)[0]:
            action = int(action)
            if not self._contexts[action]:
                self._coefs[action] = 0.0
            else:
                x_action = np.vstack(
                    [
                        _policy_features(c, self.context_dim, self.include_intercept)
                        for c in self._contexts[action]
                    ]
                )
                y_action = np.asarray(self._rewards[action], dtype=np.float64)
                if self.reward_type == "linear_gaussian":
                    self._coefs[action] = _fit_weighted_linear(x_action, y_action)
                else:
                    self._coefs[action] = _fit_weighted_logistic_irls(
                        x_action,
                        y_action,
                        max_iter=self.max_iter,
                        tol=self.tol,
                    )
            self._dirty[action] = False

        x = _policy_features(context, self.context_dim, self.include_intercept)
        values = self._coefs @ x
        if self.reward_type == "logistic_bernoulli":
            return expit(np.clip(values, -35.0, 35.0))
        return values


class ContextualTSPolicy:
    """
    Contextual Thompson sampling policy for linear Gaussian or logistic rewards.

    For reward_type="linear_gaussian", this uses Bayesian linear regression per
    action. For reward_type="logistic_bernoulli", this uses a per-action
    Laplace approximation to logistic regression.
    """

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        reward_type: str = "linear_gaussian",
        obs_sigma: float = 1.0,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        prior_precision: float | None = None,
        include_intercept: bool = True,
        n_prob_mc: int = 5000,
        pi_clip: float = 1e-8,
        posterior_scale: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-8,
        seed: int | None = None,
    ):
        if n_actions <= 0 or context_dim < 0 or (context_dim == 0 and not include_intercept):
            raise ValueError("Need positive actions and at least one feature including the intercept.")
        if reward_type not in {"linear_gaussian", "logistic_bernoulli"}:
            raise ValueError("reward_type must be 'linear_gaussian' or 'logistic_bernoulli'.")
        if obs_sigma <= 0.0:
            raise ValueError("obs_sigma must be positive.")
        if prior_var <= 0.0:
            raise ValueError("prior_var must be positive.")
        if prior_precision is not None and prior_precision < 0.0:
            raise ValueError("prior_precision must be non-negative.")
        if posterior_scale <= 0.0:
            raise ValueError("posterior_scale must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0.0:
            raise ValueError("tol must be positive.")
        self.n_actions = int(n_actions)
        self.context_dim = int(context_dim)
        self.reward_type = reward_type
        self.obs_sigma = float(obs_sigma)
        self.prior_mean = float(prior_mean)
        self.prior_var = float(prior_var)
        self.prior_precision = float(1.0 / prior_var if prior_precision is None else prior_precision)
        self.include_intercept = bool(include_intercept)
        self.n_prob_mc = int(n_prob_mc)
        self.pi_clip = float(pi_clip)
        self.posterior_scale = float(posterior_scale)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rng = np.random.default_rng(seed)
        dim = self.context_dim + int(self.include_intercept)

        if self.reward_type == "linear_gaussian":
            prior_precision_matrix = np.eye(dim, dtype=np.float64) / self.prior_var
            prior_info = np.full(dim, self.prior_mean, dtype=np.float64) / self.prior_var
            self._precision = np.repeat(prior_precision_matrix[None, :, :], self.n_actions, axis=0)
            self._info = np.repeat(prior_info[None, :], self.n_actions, axis=0)
        else:
            self._contexts: list[list[np.ndarray]] = [[] for _ in range(self.n_actions)]
            self._rewards: list[list[float]] = [[] for _ in range(self.n_actions)]
            self._coefs = np.zeros((self.n_actions, dim), dtype=np.float64)
            self._covs = np.repeat(np.eye(dim, dtype=np.float64)[None, :, :], self.n_actions, axis=0)
            if self.prior_precision > 0.0:
                self._covs /= self.prior_precision
            else:
                self._covs *= 1e6
            self._dirty = np.ones(self.n_actions, dtype=bool)

    def action_probs(self, context: np.ndarray, history: dict[str, np.ndarray] | None = None) -> np.ndarray:
        x = _policy_features(context, self.context_dim, self.include_intercept)
        values = np.zeros((self.n_prob_mc, self.n_actions), dtype=np.float64)

        if self.reward_type == "linear_gaussian":
            for action in range(self.n_actions):
                cov = _inv_or_pinv(self._precision[action])
                mean = cov @ self._info[action]
                theta = self.rng.multivariate_normal(mean, cov, size=self.n_prob_mc)
                values[:, action] = theta @ x
        else:
            self._fit_logistic_dirty_models()
            for action in range(self.n_actions):
                theta = self.rng.multivariate_normal(
                    self._coefs[action],
                    self.posterior_scale**2 * self._covs[action],
                    size=self.n_prob_mc,
                )
                values[:, action] = expit(np.clip(theta @ x, -35.0, 35.0))

        winners = np.argmax(values, axis=1)
        return _counts_to_probs(winners, self.n_actions, self.pi_clip)

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError("action is out of range.")
        if self.reward_type == "linear_gaussian":
            x = _policy_features(context, self.context_dim, self.include_intercept)
            self._precision[action] += np.outer(x, x) / (self.obs_sigma**2)
            self._info[action] += x * float(reward) / (self.obs_sigma**2)
        else:
            self._contexts[action].append(np.asarray(context, dtype=np.float64).reshape(-1))
            self._rewards[action].append(float(reward))
            self._dirty[action] = True

    def posterior_means(self) -> np.ndarray:
        if self.reward_type == "linear_gaussian":
            means = np.zeros((self.n_actions, self.context_dim + int(self.include_intercept)), dtype=np.float64)
            for action in range(self.n_actions):
                means[action] = _solve_or_pinv(self._precision[action], self._info[action])
            return means
        self._fit_logistic_dirty_models()
        return self._coefs.copy()

    def _fit_logistic_dirty_models(self) -> None:
        for action in np.where(self._dirty)[0]:
            action = int(action)
            dim = self._coefs.shape[1]
            if not self._contexts[action]:
                self._coefs[action] = 0.0
                precision = self.prior_precision * np.eye(dim, dtype=np.float64)
                if self.include_intercept:
                    precision[0, 0] = 0.0
                self._covs[action] = _inv_or_pinv(precision + 1e-8 * np.eye(dim))
            else:
                x_action = np.vstack(
                    [
                        _policy_features(c, self.context_dim, self.include_intercept)
                        for c in self._contexts[action]
                    ]
                )
                y_action = np.asarray(self._rewards[action], dtype=np.float64)
                beta = _fit_weighted_logistic_irls(
                    x_action,
                    y_action,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
                probs = expit(np.clip(x_action @ beta, -35.0, 35.0))
                weights = np.maximum(probs * (1.0 - probs), 1e-10)
                precision = (x_action.T * weights) @ x_action + self.prior_precision * np.eye(
                    dim, dtype=np.float64
                )
                if self.include_intercept:
                    precision[0, 0] -= self.prior_precision
                self._coefs[action] = beta
                self._covs[action] = _inv_or_pinv(precision + 1e-8 * np.eye(dim))
            self._dirty[action] = False


def _policy_features(
    context: np.ndarray,
    context_dim: int,
    include_intercept: bool,
) -> np.ndarray:
    context = np.asarray(context, dtype=np.float64).reshape(-1)
    if context.shape[0] != context_dim:
        raise ValueError(f"context must have length {context_dim}.")
    if include_intercept:
        return np.concatenate([[1.0], context])
    return context


def _fit_weighted_linear(
    x_design: np.ndarray,
    rewards: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    x_design = np.asarray(x_design, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if sample_weights is None:
        sample_weights = np.ones(x_design.shape[0], dtype=np.float64)
    else:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
    lhs = (x_design.T * sample_weights) @ x_design
    rhs = x_design.T @ (sample_weights * rewards)
    return _solve_or_pinv(lhs, rhs)


def _fit_weighted_logistic_irls(
    x_design: np.ndarray,
    rewards: np.ndarray,
    max_iter: int,
    tol: float,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    x_design = np.asarray(x_design, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if np.any((rewards < 0.0) | (rewards > 1.0)):
        raise ValueError("Logistic rewards must lie in [0, 1].")
    if not np.allclose(rewards, np.round(rewards), atol=1e-8):
        raise ValueError("Logistic rewards must be binary 0/1 values.")
    if sample_weights is None:
        sample_weights = np.ones(x_design.shape[0], dtype=np.float64)
    else:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
    beta = np.zeros(x_design.shape[1], dtype=np.float64)

    for _ in range(max_iter):
        eta = np.clip(x_design @ beta, -35.0, 35.0)
        probs = expit(eta)
        weights = np.maximum(probs * (1.0 - probs), 1e-10)
        gradient = x_design.T @ (sample_weights * (rewards - probs))
        hessian = (x_design.T * (sample_weights * weights)) @ x_design
        step = _solve_or_pinv(hessian, gradient)
        beta_next = beta + step
        if np.linalg.norm(step) <= tol * (1.0 + np.linalg.norm(beta)):
            return beta_next
        beta = beta_next
    return beta


def _counts_to_probs(winners: np.ndarray, n_actions: int, pi_clip: float) -> np.ndarray:
    probs = np.bincount(np.asarray(winners, dtype=np.int64), minlength=n_actions).astype(np.float64)
    probs /= max(probs.sum(), 1.0)
    if pi_clip > 0.0:
        probs = np.maximum(probs, pi_clip)
    return probs / probs.sum()


def _solve_or_pinv(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def _inv_or_pinv(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)
