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


class ContextualSubGaussianWorkingModel:
    """Gaussian surrogate with frozen dispersion by default.

    Uses the existing feature maps and mean estimators. Empirical additive
    dispersion optionally uses joint estimating-equation sandwich covariance.
    Adaptive covariance retains the repository's inverse-propensity convention;
    this implementation alone does not establish an adaptive CLT.
    """
    def __init__(self, n_actions, context_dim, kind, variance_method='empirical',
                 scales=1., support_widths=None, adaptive_behavior=False,
                 propagate_variance_uncertainty=False, variance_floor=1e-8,
                 proxy_alpha=.49, proxy_grid_size=4001, feature_map=None):
        from .dispersion import arm_values
        if kind not in {'scaled_bernoulli','uniform','gaussian_mixture','beta'}:
            raise ValueError('Unsupported environment')
        if variance_method not in {'empirical','variance_proxy','hoeffding'}:
            raise ValueError('Unknown variance method')
        if propagate_variance_uncertainty and variance_method == 'variance_proxy':
            raise ValueError('Uncertainty propagation for estimated proxies is not implemented')
        if variance_method == 'hoeffding' and kind == 'gaussian_mixture':
            raise ValueError('Hoeffding requires known bounded support')
        if variance_floor <= 0:
            raise ValueError('variance_floor must be positive')
        self.n_actions, self.context_dim = n_actions, context_dim
        self.kind, self.variance_method = kind, variance_method
        self.scales = arm_values(scales,n_actions,'scales')
        self.widths = None if support_widths is None else arm_values(support_widths,n_actions,'support widths')
        if variance_method == 'hoeffding' and self.widths is None:
            raise ValueError('Known support widths required')
        self.adaptive_behavior = adaptive_behavior
        self.propagate = propagate_variance_uncertainty and variance_method == 'empirical'
        self.floor, self.proxy_alpha, self.proxy_grid_size = variance_floor, proxy_alpha, proxy_grid_size
        cls = ContextualLogisticBernoulliRewardModel if kind == 'scaled_bernoulli' else ContextualLinearGaussianRewardModel
        self.mean_model = cls(n_actions,context_dim,adaptive_behavior=adaptive_behavior,feature_map=feature_map)

    def fit(self, contexts, actions, rewards, behavior_probs=None):
        from .dispersion import residual_proxy
        x = np.asarray(contexts,dtype=float); a = np.asarray(actions,dtype=int); y = np.asarray(rewards,dtype=float)
        _validate_logs(x,a,y)
        if np.any(a < 0) or np.any(a >= self.n_actions) or not np.isfinite(y).all() or not np.isfinite(x).all():
            raise ValueError('Invalid observations')
        design = self.mean_model.design_matrix(x,a)
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError('Offline feature design is rank deficient; collect more observations')
        n,p = design.shape; self.p = p
        weights = _estimating_weights(behavior_probs,a,self.adaptive_behavior,n)
        response = y/self.scales[a] if self.kind == 'scaled_bernoulli' else y
        beta,_ = self.mean_model.fit(x,a,response,behavior_probs)
        if not np.isfinite(beta).all():
            raise ValueError('Nonfinite mean fit')
        self.frozen_beta = beta.copy()
        if self.kind == 'scaled_bernoulli':
            prob = expit(design@beta)
            if np.any(prob < 1e-7) or np.any(prob > 1-1e-7):
                raise ValueError('Logistic fit near separation; more offline data required')
            residual = response-prob
            jac = (design.T*(weights*prob*(1-prob)))@design/n
            scores = weights[:,None]*design*residual[:,None]
            inv = np.linalg.inv(jac)
            covariance = inv@(scores.T@scores/n)@inv.T
            self.variances = None
            params = beta
        else:
            residual = y-design@beta
            self.variances = np.empty(self.n_actions)
            for arm in range(self.n_actions):
                mask = a == arm
                if mask.sum() < 2:
                    raise ValueError('Need >=2 observations per arm')
                v = np.average(residual[mask]**2,weights=weights[mask])
                if self.variance_method == 'hoeffding': v = self.widths[arm]**2/4
                if self.variance_method == 'variance_proxy':
                    v = residual_proxy(residual[mask],self.proxy_alpha,self.proxy_grid_size,self.floor)
                self.variances[arm] = max(v,self.floor)
            size = p+self.n_actions if self.propagate else p
            scores = np.zeros((n,size)); jac = np.zeros((size,size))
            scores[:,:p] = weights[:,None]*design*residual[:,None]
            jac[:p,:p] = (design.T*weights)@design/n
            if self.propagate:
                for arm in range(self.n_actions):
                    wa = weights*(a == arm)
                    scores[:,p+arm] = wa*(residual**2-self.variances[arm])
                    jac[p+arm,:p] = np.mean(2*wa[:,None]*residual[:,None]*design,axis=0)
                    jac[p+arm,p+arm] = wa.mean()
            inv = np.linalg.inv(jac)
            covariance = inv@(scores.T@scores/n)@inv.T
            params = np.r_[beta,self.variances] if self.propagate else beta
        self.lambda_hat_ = params
        self.Sigma_ = (covariance+covariance.T)/2
        if not np.isfinite(self.lambda_hat_).all() or not np.isfinite(self.Sigma_).all():
            raise ValueError('Nonfinite fit/covariance; check data, overlap and numerical conditioning')
        return self.lambda_hat_, self.Sigma_

    def mean(self,x,a,params=None):
        theta = self.lambda_hat_ if params is None else np.asarray(params)
        eta = self.mean_model.features(x,a)@theta[:self.p]
        return float(self.scales[a]*expit(eta) if self.kind == 'scaled_bernoulli' else eta)

    def variance(self,x,a,params=None):
        from .dispersion import bernoulli_proxy
        theta = self.lambda_hat_ if params is None else np.asarray(params)
        if self.kind == 'scaled_bernoulli':
            beta = theta[:self.p] if self.propagate else self.frozen_beta
            prob = expit(self.mean_model.features(x,a)@beta)
            if self.variance_method == 'hoeffding': v = self.widths[a]**2/4
            elif self.variance_method == 'empirical': v = self.scales[a]**2*prob*(1-prob)
            else: v = self.scales[a]**2*float(bernoulli_proxy(prob))
        else:
            v = theta[self.p+a] if self.propagate else self.variances[a]
        return float(max(v,self.floor))

    def sample(self,x,a,rng,params=None):
        return float(rng.normal(self.mean(x,a,params),np.sqrt(self.variance(x,a,params))))

    def score(self,x,a,reward,params=None):
        theta = self.lambda_hat_ if params is None else np.asarray(params)
        feat = self.mean_model.features(x,a)
        mu = self.mean(x,a,theta); v = self.variance(x,a,theta); e = reward-mu
        result = np.zeros(len(theta))
        dm = feat
        if self.kind == 'scaled_bernoulli':
            prob = mu/self.scales[a]; dm = self.scales[a]*prob*(1-prob)*feat
        result[:self.p] = dm*e/v
        variance_score = (e*e-v)/(2*v*v)
        if self.propagate:
            if self.kind == 'scaled_bernoulli':
                raw = self.scales[a]**2*prob*(1-prob)
                if raw > self.floor:
                    result[:self.p] += variance_score*self.scales[a]**2*prob*(1-prob)*(1-2*prob)*feat
            elif theta[self.p+a] > self.floor:
                result[self.p+a] = variance_score
        return result
