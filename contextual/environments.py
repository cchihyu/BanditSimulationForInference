from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import expit


class ContextualLinearGaussianRewardModel:
    """
    Correctly specified contextual normal reward model with linear mean.

    R | X=x, A=a ~ Normal(phi(x, a)^T beta, sigma^2).

    The default feature map is one intercept plus the raw context for each
    action, block-encoded into a vector of length K * (d + 1).
    """

    def __init__(
        self,
        n_actions: int,
        context_dim: int | None = None,
        sigma: float | None = None,
        adaptive_behavior: bool = False,
        feature_map: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
    ):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if sigma is not None and sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        self.n_actions = int(n_actions)
        self.context_dim = context_dim
        self.sigma = None if sigma is None else float(sigma)
        self.adaptive_behavior = bool(adaptive_behavior)
        self.feature_map = feature_map
        self.lambda_hat_: np.ndarray | None = None
        self.Sigma_: np.ndarray | None = None

    def _base_features(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(context, dtype=np.float64).reshape(-1)
        if self.context_dim is None:
            self.context_dim = context.shape[0]
        if context.shape[0] != self.context_dim:
            raise ValueError(f"context must have length {self.context_dim}.")
        return np.concatenate([[1.0], context])

    def features(self, context: np.ndarray, action: int) -> np.ndarray:
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError("action is out of range.")
        if self.feature_map is not None:
            feat = np.asarray(
                self.feature_map(np.asarray(context, dtype=np.float64), action, self.n_actions),
                dtype=np.float64,
            )
            if feat.ndim != 1:
                raise ValueError("feature_map must return a 1D feature vector.")
            return feat

        base = self._base_features(context)
        feat = np.zeros(self.n_actions * base.shape[0], dtype=np.float64)
        start = action * base.shape[0]
        feat[start : start + base.shape[0]] = base
        return feat

    def design_matrix(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        contexts = _as_2d_contexts(contexts)
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape[0] != contexts.shape[0]:
            raise ValueError("contexts and actions must have the same length.")
        rows = [self.features(contexts[i], int(actions[i])) for i in range(contexts.shape[0])]
        return np.vstack(rows)

    def fit(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_probs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        contexts = _as_2d_contexts(contexts)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float64)
        _validate_logs(contexts, actions, rewards)

        x_design = self.design_matrix(contexts, actions)
        n_obs, n_params = x_design.shape
        estimating_weights = _estimating_weights(
            behavior_probs,
            actions,
            adaptive_behavior=self.adaptive_behavior,
            n_obs=n_obs,
        )
        lhs = (x_design.T * estimating_weights) @ x_design
        rhs = x_design.T @ (estimating_weights * rewards)
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(lhs) @ rhs

        residuals = rewards - x_design @ beta
        if self.sigma is None:
            if self.adaptive_behavior:
                sigma2 = float(
                    max(
                        np.sum(estimating_weights * residuals**2)
                        / max(np.sum(estimating_weights), 1.0),
                        1e-12,
                    )
                )
            else:
                rank = int(np.linalg.matrix_rank(x_design))
                df_resid = max(n_obs - rank, 1)
                sigma2 = float(max((residuals @ residuals) / df_resid, 1e-12))
        else:
            sigma2 = self.sigma**2

        if self.adaptive_behavior:
            score_rows = (residuals[:, None] / sigma2) * x_design
            h_hat = ((x_design.T * estimating_weights) @ x_design) / (n_obs * sigma2)
            v_hat = (score_rows.T * (estimating_weights**2)) @ score_rows / n_obs
            try:
                h_inv = np.linalg.inv(h_hat)
            except np.linalg.LinAlgError:
                h_inv = np.linalg.pinv(h_hat)
            Sigma = h_inv @ v_hat @ h_inv
        else:
            gram = (x_design.T @ x_design) / n_obs
            try:
                gram_inv = np.linalg.inv(gram)
            except np.linalg.LinAlgError:
                gram_inv = np.linalg.pinv(gram)
            Sigma = sigma2 * gram_inv

        self.lambda_hat_ = beta
        self.Sigma_ = Sigma
        if self.sigma is None:
            self.sigma = float(np.sqrt(sigma2))
        return beta, Sigma

    def mean(self, context: np.ndarray, action: int, params: np.ndarray | None = None) -> float:
        params = self._params(params)
        return float(self.features(context, action) @ params)

    def sample(
        self,
        context: np.ndarray,
        action: int,
        rng: np.random.Generator,
        params: np.ndarray | None = None,
    ) -> float:
        if self.sigma is None:
            raise RuntimeError("sigma is unknown; fit the model or provide sigma first.")
        return float(rng.normal(self.mean(context, action, params=params), self.sigma))

    def score(
        self,
        context: np.ndarray,
        action: int,
        reward: float,
        params: np.ndarray | None = None,
    ) -> np.ndarray:
        params = self._params(params)
        if self.sigma is None:
            raise RuntimeError("sigma is unknown; fit the model or provide sigma first.")
        feat = self.features(context, action)
        residual = float(reward) - float(feat @ params)
        return feat * residual / (self.sigma**2)

    def _params(self, params: np.ndarray | None) -> np.ndarray:
        if params is None:
            if self.lambda_hat_ is None:
                raise RuntimeError("Model parameters are unavailable; fit the model first.")
            return self.lambda_hat_
        return np.asarray(params, dtype=np.float64)


class ContextualLogisticBernoulliRewardModel:
    """
    Correctly specified contextual logistic reward model.

    R | X=x, A=a ~ Bernoulli(sigmoid(phi(x, a)^T lambda)).

    By default, features are block-encoded by action and include an intercept:
    phi(x, a) has length K * (d + 1).
    """

    def __init__(
        self,
        n_actions: int,
        context_dim: int | None = None,
        include_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-8,
        adaptive_behavior: bool = False,
        feature_map: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
    ):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0.0:
            raise ValueError("tol must be positive.")
        self.n_actions = int(n_actions)
        self.context_dim = context_dim
        self.include_intercept = bool(include_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.adaptive_behavior = bool(adaptive_behavior)
        self.feature_map = feature_map
        self.lambda_hat_: np.ndarray | None = None
        self.Sigma_: np.ndarray | None = None

    def _base_features(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(context, dtype=np.float64).reshape(-1)
        if self.context_dim is None:
            self.context_dim = context.shape[0]
        if context.shape[0] != self.context_dim:
            raise ValueError(f"context must have length {self.context_dim}.")
        if self.include_intercept:
            return np.concatenate([[1.0], context])
        return context

    def features(self, context: np.ndarray, action: int) -> np.ndarray:
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError("action is out of range.")
        if self.feature_map is not None:
            feat = np.asarray(
                self.feature_map(np.asarray(context, dtype=np.float64), action, self.n_actions),
                dtype=np.float64,
            )
            if feat.ndim != 1:
                raise ValueError("feature_map must return a 1D feature vector.")
            return feat

        base = self._base_features(context)
        feat = np.zeros(self.n_actions * base.shape[0], dtype=np.float64)
        start = action * base.shape[0]
        feat[start : start + base.shape[0]] = base
        return feat

    def design_matrix(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        contexts = _as_2d_contexts(contexts)
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape[0] != contexts.shape[0]:
            raise ValueError("contexts and actions must have the same length.")
        rows = [self.features(contexts[i], int(actions[i])) for i in range(contexts.shape[0])]
        return np.vstack(rows)

    def fit(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_probs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        contexts = _as_2d_contexts(contexts)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float64)
        _validate_logs(contexts, actions, rewards)
        if np.any((rewards < 0.0) | (rewards > 1.0)):
            raise ValueError("Logistic Bernoulli rewards must lie in [0, 1].")
        if not np.allclose(rewards, np.round(rewards), atol=1e-8):
            raise ValueError("Logistic Bernoulli rewards must be binary 0/1 values.")

        x_design = self.design_matrix(contexts, actions)
        estimating_weights = _estimating_weights(
            behavior_probs,
            actions,
            adaptive_behavior=self.adaptive_behavior,
            n_obs=contexts.shape[0],
        )
        beta = self._fit_irls(x_design, rewards, sample_weights=estimating_weights)
        probs = expit(np.clip(x_design @ beta, -35.0, 35.0))

        score_rows = (rewards - probs)[:, None] * x_design
        h_weights = estimating_weights * probs * (1.0 - probs)
        h_hat = (x_design.T * h_weights) @ x_design / contexts.shape[0]
        v_hat = (score_rows.T * (estimating_weights**2)) @ score_rows / contexts.shape[0]
        try:
            h_inv = np.linalg.inv(h_hat)
        except np.linalg.LinAlgError:
            h_inv = np.linalg.pinv(h_hat)
        Sigma = h_inv @ v_hat @ h_inv

        self.lambda_hat_ = beta
        self.Sigma_ = Sigma
        return beta, Sigma

    def mean(self, context: np.ndarray, action: int, params: np.ndarray | None = None) -> float:
        params = self._params(params)
        return float(expit(np.clip(self.features(context, action) @ params, -35.0, 35.0)))

    def predict_all(self, contexts: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        contexts = _as_2d_contexts(contexts)
        out = np.zeros((contexts.shape[0], self.n_actions), dtype=np.float64)
        for action in range(self.n_actions):
            out[:, action] = [self.mean(context, action, params=params) for context in contexts]
        return out

    def sample(
        self,
        context: np.ndarray,
        action: int,
        rng: np.random.Generator,
        params: np.ndarray | None = None,
    ) -> float:
        return float(rng.binomial(1, self.mean(context, action, params=params)))

    def score(
        self,
        context: np.ndarray,
        action: int,
        reward: float,
        params: np.ndarray | None = None,
    ) -> np.ndarray:
        params = self._params(params)
        prob = self.mean(context, action, params=params)
        return self.features(context, action) * (float(reward) - prob)

    def _fit_irls(
        self,
        x_design: np.ndarray,
        rewards: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        n_params = x_design.shape[1]
        beta = np.zeros(n_params, dtype=np.float64)
        if sample_weights is None:
            sample_weights = np.ones(x_design.shape[0], dtype=np.float64)
        else:
            sample_weights = np.asarray(sample_weights, dtype=np.float64)

        for _ in range(self.max_iter):
            eta = np.clip(x_design @ beta, -35.0, 35.0)
            probs = expit(eta)
            weights = np.maximum(probs * (1.0 - probs), 1e-10)
            gradient = x_design.T @ (sample_weights * (rewards - probs))
            hessian = (x_design.T * (sample_weights * weights)) @ x_design
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

    def _params(self, params: np.ndarray | None) -> np.ndarray:
        if params is None:
            if self.lambda_hat_ is None:
                raise RuntimeError("Model parameters are unavailable; fit the model first.")
            return self.lambda_hat_
        return np.asarray(params, dtype=np.float64)


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


def _estimating_weights(
    behavior_probs: np.ndarray | None,
    actions: np.ndarray,
    adaptive_behavior: bool,
    n_obs: int,
) -> np.ndarray:
    """
    Weights for the adaptive-pi0 estimating equation.

    For adaptive logging, the sandwich construction compares the adaptive
    behavior policy against a fixed reference behavior policy. The verification
    scripts use a static uniform reference policy, so the selected-action weight
    is

        W_t = pi_ref(A_t) / pi0(A_t | X_t, H_{t-1})
            = (1 / K) / pi0(A_t | X_t, H_{t-1}).

    Under static uniform logging this gives W_t = 1, so the adaptive sandwich
    collapses to the ordinary unweighted sandwich sanity check.
    """
    if not adaptive_behavior:
        return np.ones(n_obs, dtype=np.float64)
    if behavior_probs is None:
        raise ValueError("behavior_probs are required when adaptive_behavior=True.")
    behavior_probs = np.asarray(behavior_probs, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    if behavior_probs.ndim != 2 or behavior_probs.shape[0] != n_obs:
        raise ValueError("behavior_probs must have shape (n_obs, n_actions).")
    if actions.shape[0] != n_obs:
        raise ValueError("actions must have length n_obs.")
    if np.any(actions < 0) or np.any(actions >= behavior_probs.shape[1]):
        raise ValueError("actions must be valid columns of behavior_probs.")
    logged_probs = behavior_probs[np.arange(n_obs), actions]
    if np.any(~np.isfinite(logged_probs)) or np.any(logged_probs <= 0.0):
        raise ValueError("logged behavior probabilities must be positive and finite.")
    row_sums = behavior_probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("Each behavior probability row must sum to 1.")
    reference_logged_probs = np.full(n_obs, 1.0 / behavior_probs.shape[1], dtype=np.float64)
    return reference_logged_probs / logged_probs
