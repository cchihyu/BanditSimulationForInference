import numpy as np
from scipy.stats import norm

from environments import *


def bandit_exp_runner(
    env,
    algo_builder,
    T,
    n_reps=1,
    base_exp_seed=1013,
    table_seed=2026,
    table_renew=False,
    adaptive=False,
):
    all_actions = np.zeros((n_reps, T), dtype=int)
    all_rewards = np.zeros((n_reps, T), dtype=float)
    all_probs = np.zeros((n_reps, T, env.n_actions), dtype=float) if adaptive else None

    response_table = env.sample_table(T=T, seed=table_seed)

    for rep in range(n_reps):
        if table_renew:
            response_table = env.sample_table(T=T, seed=table_seed + rep)

        algo = algo_builder(base_exp_seed + rep)

        for t in range(T):
            if adaptive:
                action, probs = algo.select_action_with_probs()
                all_probs[rep, t] = probs
            else:
                action = algo.select_action()

            reward = response_table["table"][action, t]
            algo.update(action, reward)

            all_actions[rep, t] = action
            all_rewards[rep, t] = reward

    total_rewards = all_rewards.sum(axis=1)
    avg_rewards = total_rewards / T

    return {
        "table_seed": response_table["seed"],
        "exp_seed": base_exp_seed,
        "env": env,
        "n_rep": n_reps,
        "T": T,
        "all_actions": all_actions,
        "all_rewards": all_rewards,
        "all_probs": all_probs,  # NEW: (n_reps, T, K) or None
        "mean_total_reward": total_rewards.mean(),
        "se_total_reward": total_rewards.std(ddof=1) / np.sqrt(n_reps)
        if n_reps > 1
        else 0.0,
        "std_total_reward": total_rewards.std(ddof=1) if n_reps > 1 else 0.0,
        "mean_avg_reward": avg_rewards.mean(),
        "se_avg_reward": avg_rewards.std(ddof=1) / np.sqrt(n_reps)
        if n_reps > 1
        else 0.0,
        "std_avg_reward": avg_rewards.std(ddof=1) if n_reps > 1 else 0.0,
    }


def compute_arm_estimates_adaptive(
    # return the matrix V=M^{-1}Sigma M^{-1}
    actions, rewards, probs, n_actions, estimate_sigma=True, sigma_env=None, env = "Gaussian"):

    if not estimate_sigma and sigma_env is None:
        raise ValueError("sigma_env must be provided when estimate_sigma=False")

    T_off = len(actions)
    pi_t = probs[np.arange(T_off), actions]
    # compute the weights ====================================================
    W = 1.0 / np.sqrt(pi_t) # (T_offline,)
    
    # compute the mean and variance ==========================================
    mu_hat = np.zeros(n_actions, dtype=float)
    sigma_hat = np.zeros(n_actions, dtype=float)

    for a in range(n_actions):
        idx = actions == a
        if not idx.any():
            continue

        w_a = W[idx] # weight when pulling arm a
        r_a = rewards[idx] # reward when pulling arm a

        # weighted mean
        mu_hat[a] = (w_a * r_a).sum() / w_a.sum()

        res_a = r_a - mu_hat[a] # residual

        # sigma
        if estimate_sigma:
            sigma2_a = max((w_a * res_a**2).sum() / w_a.sum(), 1e-8)
            sigma_hat[a] = np.sqrt(sigma2_a)
        else:
            sigma2_a = sigma_env[a] ** 2
            sigma_hat[a] = sigma_env[a]
        

    # clip mu_hat for Bernoulli to avoid division by zero in scores/hessian
    if env == "Bernoulli":
        mu_hat = np.clip(mu_hat, 1e-6, 1.0 - 1e-6)

    # compute m dot (the gradient) =============================================
    if env == "Gaussian":
        s2_used = sigma_hat[actions]**2 if estimate_sigma else sigma_env[actions]**2
        m_dot_mu_t = (rewards - mu_hat[actions]) / (s2_used)  # (T_off,)
        if estimate_sigma:
            m_dot_sig2_t = (rewards - mu_hat[actions])**2 / 2 / s2_used**2 - 1 / 2 / s2_used  # (T_off,)
    elif env == "Bernoulli":
        m_dot_mu_t = (rewards - mu_hat[actions]) / (1 - mu_hat[actions]) / mu_hat[actions]  # (T_off,)

    # compute m dot dot (the hessian) =============================================
    if env == "Gaussian":
        if estimate_sigma == True:
            res = rewards - mu_hat[actions]   # (T_off,)
            s2_hat = sigma_hat[actions]**2
            m_ddot = np.zeros((T_off, 2, 2))
            m_ddot[:, 0, 0] = -1 / s2_hat
            m_ddot[:, 0, 1] = -res / s2_hat**2
            m_ddot[:, 1, 0] = -res / s2_hat**2
            m_ddot[:, 1, 1] = -res**2 / s2_hat**3 + 1 / (2 * s2_hat**2)
        elif estimate_sigma == False:
            s2 = sigma_env[actions]**2
            m_ddot = np.zeros((T_off, 1, 1))
            m_ddot[:, 0, 0] = -1 / s2

    if env == "Bernoulli":
        m_ddot = np.zeros((T_off, 1, 1))
        m_ddot[:, 0, 0] = (
            -rewards / mu_hat[actions] ** 2
            - (1 - rewards) / (1 - mu_hat[actions]) ** 2
        )

    # compute M_ddot and Sigma =============================================
    block_size = 2 if (env == "Gaussian" and estimate_sigma) else 1
    M_ddot_blocks = np.zeros((n_actions, block_size, block_size))
    Sigma_blocks = np.zeros((n_actions, block_size, block_size))

    for a in range(n_actions):
        idx = actions == a
        if not idx.any():
            M_ddot_blocks[a] = np.full((block_size, block_size), np.nan)
            Sigma_blocks[a] = np.full((block_size, block_size), np.nan)
            continue

        w_a = W[idx]

        # M_ddot block for arm a
        M_ddot_blocks[a] = (m_ddot[idx] * w_a[:, None, None]).sum(axis=0) / T_off

        # stack gradient components into (n_a, block_size)
        if env == "Gaussian" and estimate_sigma:
            m_dot_a = np.column_stack([m_dot_mu_t[idx], m_dot_sig2_t[idx]])  # (n_a, 2)
        else:
            m_dot_a = m_dot_mu_t[idx].reshape(-1, 1)  # (n_a, 1)

        # Sigma block for arm a: sum_t W_t^2 * m_dot_t m_dot_t^T / T_off
        weighted_m_dot_a = w_a[:, None] * m_dot_a  # (n_a, block_size)
        Sigma_blocks[a] = weighted_m_dot_a.T @ weighted_m_dot_a / T_off

    # V = M^{-1} Sigma M^{-1}, block-diagonal with one block per arm
    lam_dim = n_actions * block_size
    V = np.zeros((lam_dim, lam_dim))
    for a in range(n_actions):
        s = a * block_size        # start row/col of arm a's block in V
        e = s + block_size        # end row/col (exclusive)
        M_a = M_ddot_blocks[a]  # Fisher info (positive definite); M_ddot is negative
        M_a_inv = np.linalg.inv(M_a)
        V[s:e, s:e] = M_a_inv @ Sigma_blocks[a] @ M_a_inv

    return mu_hat, sigma_hat, V


# given a result from the runner, compute the mean and std of the reward for each arm
def compute_arm_mean_std(all_actions, all_rewards, n_actions):
    n_reps, _ = all_actions.shape

    arm_means = np.full((n_reps, n_actions), np.nan, dtype=float)
    arm_std = np.full((n_reps, n_actions), np.nan, dtype=float)

    # loop through all repeated experiments to find out the reward's mean and variance
    for rep in range(n_reps):
        for a in range(n_actions):
            mask = all_actions[rep] == a
            rewards_a = all_rewards[rep][mask]

            if len(rewards_a) > 0:
                arm_means[rep, a] = rewards_a.mean()
            if len(rewards_a) >= 2:
                arm_std[rep, a] = rewards_a.std(ddof=1)
            elif len(rewards_a) == 1:
                arm_std[rep, a] = 0.0

    return {
        "arm_means": arm_means,
        "arm_std": arm_std,
    }


# Base class: collect the common information
class BaseInference:
    def __init__(
        self, true_env, algo_builder1, algo_builder2, T, algo_seed=2026, table_seed=1013
    ):
        self.true_env = true_env
        self.algo_builder1 = algo_builder1
        self.algo_builder2 = algo_builder2
        self.T = T
        self.algo_seed = algo_seed
        self.table_seed = table_seed


class AdaptiveNormalBSI(BaseInference):
    def __init__(
        self,
        true_env,
        algo_builder1,
        algo_builder2,
        T,
        algo_seed=2026,
        table_seed=1013,
        estimate_sigma=False,
    ):
        super().__init__(
            true_env, algo_builder1, algo_builder2, T, algo_seed, table_seed
        )
        self.estimate_sigma = estimate_sigma

    def compute_se(
        self, all_actions, all_rewards, hat_mu, hat_sigma2, V
    ): # compute sqrt of gvg
        
        all_actions = np.asarray(all_actions, dtype=int)
        all_rewards = np.asarray(all_rewards, dtype=float)
        n_reps, T = all_actions.shape
        n_actions = len(hat_mu)
        
        # compute the gradient
        grad_hat_mu = np.zeros(n_actions, dtype=float)

        if self.estimate_sigma:
            grad_hat_sigma2 = np.zeros(n_actions, dtype=float)

        for rep in range(n_reps):
            actions = all_actions[rep]  # (T,)
            rewards = all_rewards[rep]  # (T,)

            G = np.cumsum(rewards[::-1])[::-1]
            G = np.concatenate([G, [0]])  # This defines G1 through G(T+1)
            residuals = rewards - hat_mu[actions]  # (R_t - mu_a), shape (T,)
            s2 = hat_sigma2[actions]  # sigma_a^2 for each t, shape (T,)

            # ── mu gradient ──────────────────────────────────────────────
            scores_mu = residuals / (s2)  # (T,)
            contrib_mu = (1.0 + scores_mu * G[1:]) / T  # (T,)
            np.add.at(grad_hat_mu, actions, contrib_mu)

            # ── sigma^2 gradient (only when estimating sigma) ────────────
            # score is -1/(2*sigma^2) + (R-mu)^2/(2*sigma^4)
            if self.estimate_sigma:
                scores_sigma2 = -0.5 / s2 + residuals**2 / (
                    2.0 * s2**2
                )  # FIXED: -0.5/s2
                contrib_sigma2 = (scores_sigma2 * G[1:]) / T
                np.add.at(grad_hat_sigma2, actions, contrib_sigma2)

        grad_hat_mu /= n_reps
        if self.estimate_sigma:
            grad_hat_sigma2 /= n_reps
        grad = grad_hat_mu
        if self.estimate_sigma:
            grad = np.stack([grad_hat_mu, grad_hat_sigma2], axis=1).ravel()

        return float(np.sqrt(grad @ V @ grad)), grad

        
    
    def run(self, offline_data, alphas, n_reps):
        from scipy.stats import chi2

        T_off = offline_data["T"]
        actions = offline_data["all_actions"][0]  # single rep
        rewards = offline_data["all_rewards"][0]
        probs = offline_data["all_probs"][0]
        n_actions = self.true_env.n_actions

        # estimate arm parameters from offline data
        sigma_env = None if self.estimate_sigma else self.true_env.sigmas
        hat_mu, hat_sigma, V = compute_arm_estimates_adaptive(
            actions,
            rewards,
            probs,
            n_actions,
            estimate_sigma=self.estimate_sigma,
            sigma_env=sigma_env,
        )

        hat_sigma2 = hat_sigma**2
        lambda_hat = (
            np.stack([hat_mu, hat_sigma2], axis=1).ravel()
            if self.estimate_sigma
            else hat_mu
        )

        if np.any(np.isnan(hat_mu)):
            raise ValueError(
                "Some arm mean is NaN; every arm must be observed offline at least once."
            )
        if np.any(np.isinf(V)):
            raise ValueError(
                "Some arm was never selected offline; V contains inf."
            )

        # simulate pi1 on imagined environment
        imagined_env = NormalRewardEnv(mus=hat_mu, sigma=hat_sigma)
        self.result = bandit_exp_runner(
            env=imagined_env,
            algo_builder=self.algo_builder2,
            T=self.T,
            n_reps=n_reps,
            base_exp_seed=self.algo_seed,
            table_seed=self.table_seed,
            table_renew=True,
        )

        # SE and bias correction
        sqrt_gVg, gradient = self.compute_se(
            all_actions=self.result["all_actions"],
            all_rewards=self.result["all_rewards"],
            hat_mu=hat_mu,
            hat_sigma2=hat_sigma2,
            V=V,
        )

        center = float(self.result["mean_avg_reward"])

        # chi2 CI: hw = sqrt(chi2_q / T_off) * sqrt_gVg
        df = 2 * n_actions if self.estimate_sigma else n_actions
        center_se = float(self.result["se_avg_reward"])
        var_delta = sqrt_gVg**2 / T_off
        se_delta = float(np.sqrt(var_delta))
        ci_widths = {
            alpha: float(np.sqrt(chi2.ppf(1 - alpha, df=df) / T_off) * sqrt_gVg)
            for alpha in alphas
        }
        ci_widths_adjusted = {
            alpha: float(norm.ppf(1 - alpha / 2) * se_delta)
            for alpha in alphas
        }

        return {
            "center": center,
            "center_se": center_se,
            "ci_width_proj": ci_widths,
            "ci_width": ci_widths_adjusted,
            "se": sqrt_gVg,
            "pi1_img": self.result,
            "estimate_sigma": self.estimate_sigma,
            "lambda_hat": lambda_hat,
            "gradient": gradient,
            "V": V,
        }

class AdaptiveBernoulliBSI(BaseInference):
    def __init__(
        self,
        true_env,
        algo_builder1,
        algo_builder2,
        T,
        algo_seed=2026,
        table_seed=1013,
        eps=1e-6,
    ):
        super().__init__(
            true_env, algo_builder1, algo_builder2, T, algo_seed, table_seed
        )
        self.eps = eps

    def _clip_mu(self, mu):
        return np.clip(np.asarray(mu, dtype=float), self.eps, 1.0 - self.eps)

    def compute_se(self, all_actions, all_rewards, hat_mu, V):

        all_actions = np.asarray(all_actions, dtype=int)
        all_rewards = np.asarray(all_rewards, dtype=float)

        n_reps, T = all_actions.shape
        n_actions = self.true_env.n_actions
        hat_mu = self._clip_mu(hat_mu) # for numerical stability

        # compute the gradient of f ==========================================
        grad_hat_mu = np.zeros(n_actions, dtype=float) #store the gradient of f

        for rep in range(n_reps):
            actions = all_actions[rep]
            rewards = all_rewards[rep]

            G = np.cumsum(rewards[::-1])[::-1]
            G = np.concatenate([G, [0]])  # This defines G2 through G(T+1)

            mu_a = hat_mu[actions] # (T,)

            # Bernoulli score
            scores_mu = (rewards - mu_a) / (mu_a * (1.0 - mu_a))

            contrib_mu = (1+scores_mu * G[1:]) / T
            np.add.at(grad_hat_mu, actions, contrib_mu)

        grad_hat_mu /= n_reps # compute te expectation term

        return float(np.sqrt(grad_hat_mu @ V @ grad_hat_mu)), grad_hat_mu


    def run(self, offline_data, alphas, n_reps):
        from scipy.stats import chi2

        T_off = offline_data["T"]
        actions = offline_data["all_actions"][0]
        rewards = offline_data["all_rewards"][0]
        probs   = offline_data["all_probs"][0]
        n_actions = self.true_env.n_actions

        hat_mu, _, V = compute_arm_estimates_adaptive(
            actions, rewards, probs, n_actions, env="Bernoulli"
        )
        hat_mu = self._clip_mu(hat_mu)
        lambda_hat = hat_mu

        if np.any(np.isnan(hat_mu)):
            raise ValueError(
                "Some arm mean is NaN; every arm must be observed offline at least once."
            )
        if np.any(np.isinf(V)):
            raise ValueError(
                "Some arm was never selected offline; V contains inf."
            )

        imagined_env = BernoulliRewardEnv(mus=hat_mu)

        self.result = bandit_exp_runner(
            env=imagined_env,
            algo_builder=self.algo_builder2,
            T=self.T,
            n_reps=n_reps,
            base_exp_seed=self.algo_seed,
            table_seed=self.table_seed,
            table_renew=True,
        )

        sqrt_gVg, gradient = self.compute_se(
            all_actions=self.result["all_actions"],
            all_rewards=self.result["all_rewards"],
            hat_mu=hat_mu,
            V=V,
        )

        center = float(self.result["mean_avg_reward"])

        df = n_actions
        center_se = float(self.result["se_avg_reward"])
        var_delta = sqrt_gVg**2 / T_off
        se_delta = float(np.sqrt(var_delta))
        ci_widths = {
            alpha: float(np.sqrt(chi2.ppf(1 - alpha, df=df) / T_off) * sqrt_gVg)
            for alpha in alphas
        }
        ci_widths_adjusted = {
            alpha: float(norm.ppf(1 - alpha / 2) * se_delta)
            for alpha in alphas
        }

        return {
            "center": center,
            "center_se": center_se,
            "ci_width_proj": ci_widths,
            "ci_width": ci_widths_adjusted,
            "se": sqrt_gVg,
            "pi1_img": self.result,
            "lambda_hat": lambda_hat,
            "gradient": gradient,
            "V": V,
        }

class NormalBSI(BaseInference):
    def __init__(
        self,
        true_env,
        algo_builder1,
        algo_builder2,
        T,
        algo_seed=2026,
        table_seed=1013,
        estimate_sigma=False,
    ):
        super().__init__(
            true_env, algo_builder1, algo_builder2, T, algo_seed, table_seed
        )
        self.estimate_sigma = estimate_sigma

    def _compute_Sigma(self, hat_sigma2, offline_data):
        n_actions = self.true_env.n_actions
        offline_actions = offline_data["all_actions"].flatten()
        N_a = np.bincount(offline_actions, minlength=n_actions).astype(float)
        T = offline_data["T"]
        if self.estimate_sigma:
            Sigma_diag = np.stack([hat_sigma2, 2.0 * hat_sigma2**2], axis=1).ravel() * T / N_a.repeat(2)
        else:
            Sigma_diag = hat_sigma2 * T / N_a
        return np.diag(Sigma_diag)

    def compute_se(self, all_actions, all_rewards, hat_mu, hat_sigma2, offline_data):
        all_actions = np.asarray(all_actions, dtype=int)
        all_rewards = np.asarray(all_rewards, dtype=float)
        n_reps, T = all_actions.shape
        n_actions = self.true_env.n_actions

        # ── Step 1: gradient of f w.r.t. lambda = (mu, sigma^2) ──────────
        grad_hat_mu = np.zeros(n_actions, dtype=float)
        if self.estimate_sigma:
            grad_hat_sigma2 = np.zeros(n_actions, dtype=float)

        for rep in range(n_reps):
            actions = all_actions[rep]
            rewards = all_rewards[rep]
            G = np.cumsum(rewards[::-1])[::-1]
            G = np.concatenate([G, [0]])  # G[t] = sum_{s=t}^T R_s
            residuals = rewards - hat_mu[actions]
            s2 = hat_sigma2[actions]

            scores_mu = residuals / s2
            np.add.at(grad_hat_mu, actions, (1.0 + scores_mu * G[1:]) / T)

            if self.estimate_sigma:
                scores_sigma2 = -0.5 / s2 + residuals**2 / (2.0 * s2**2)
                np.add.at(grad_hat_sigma2, actions, (scores_sigma2 * G[1:]) / T)

        grad_hat_mu /= n_reps
        if self.estimate_sigma:
            grad_hat_sigma2 /= n_reps

        # ── Step 2: SE = sqrt( g^T Sigma g ) ─────────────────────────────
        if self.estimate_sigma:
            grad = np.stack([grad_hat_mu, grad_hat_sigma2], axis=1).ravel()
        else:
            grad = grad_hat_mu

        Sigma = self._compute_Sigma(hat_sigma2, offline_data)
        return np.sqrt(grad @ Sigma @ grad / offline_data['T']), grad
 

    def run(self, offline_data, alphas, n_reps):

        # ── estimate arm means (and optionally variances) from offline data ──
        pi0_summary = compute_arm_mean_std(
            all_actions=offline_data["all_actions"],
            all_rewards=offline_data["all_rewards"],
            n_actions=self.true_env.n_actions,
        )

        hat_mu = np.asarray(pi0_summary["arm_means"][0], dtype=float)

        if self.estimate_sigma:
            hat_std = np.asarray(pi0_summary["arm_std"][0], dtype=float)
            hat_sigma2 = np.where(
                np.isnan(hat_std) | (hat_std == 0.0),
                self.true_env.sigmas**2,
                hat_std**2,
            )
        else:
            hat_sigma2 = self.true_env.sigmas**2
        lambda_hat = (
            np.stack([hat_mu, hat_sigma2], axis=1).ravel()
            if self.estimate_sigma
            else hat_mu
        )

        # ── imagined environment uses hat_sigma2 ─────────────────────────
        imagined_env = NormalRewardEnv(mus=hat_mu, sigma=np.sqrt(hat_sigma2))

        # ── simulate pi1 on imagined environment ─────────────────────────
        self.result = bandit_exp_runner(
            env=imagined_env,
            algo_builder=self.algo_builder2,
            T=self.T,
            n_reps=n_reps,
            base_exp_seed=self.algo_seed,
            table_seed=self.table_seed,
            table_renew=True,
        )

        se, gradient = self.compute_se(
            all_actions=self.result["all_actions"],
            all_rewards=self.result["all_rewards"],
            hat_mu=hat_mu,
            hat_sigma2=hat_sigma2,
            offline_data=offline_data,
        )

        center_se = float(self.result["se_avg_reward"])
        ci_widths = {alpha: norm.ppf(1 - alpha / 2) * se for alpha in alphas}

        Sigma = self._compute_Sigma(hat_sigma2, offline_data)

        return {
            "center": np.asarray(self.result["mean_avg_reward"], dtype=float),
            "center_se": center_se,
            "ci_width_proj": ci_widths,
            "se": se,
            "pi0_summary": pi0_summary,
            "pi1_img": self.result,
            "estimate_sigma": self.estimate_sigma,
            "lambda_hat": lambda_hat,
            "gradient": gradient,
            "Sigma": Sigma,
        }

class BernoulliBSI(BaseInference):
    def __init__(
        self,
        true_env,
        algo_builder1,
        algo_builder2,
        T,
        algo_seed=2026,
        table_seed=1013,
        eps=1e-6,
    ):
        super().__init__(
            true_env, algo_builder1, algo_builder2, T, algo_seed, table_seed
        )
        self.eps = eps

    def _clip_mu(self, mu):
        return np.clip(np.asarray(mu, dtype=float), self.eps, 1.0 - self.eps)

    def compute_se(self, all_actions, all_rewards, hat_mu, offline_data):

        all_actions = np.asarray(all_actions, dtype=int)
        all_rewards = np.asarray(all_rewards, dtype=float)

        n_reps, T = all_actions.shape
        n_actions = self.true_env.n_actions
        hat_mu = self._clip_mu(hat_mu) # for numerical stability

        # compute the gradient of f ==========================================
        grad_hat_mu = np.zeros(n_actions, dtype=float) #store the gradient of f

        for rep in range(n_reps):
            actions = all_actions[rep]
            rewards = all_rewards[rep]

            G = np.cumsum(rewards[::-1])[::-1]
            G = np.concatenate([G, [0]])  # This defines G2 through G(T+1)

            mu_a = hat_mu[actions] # (T,)

            # Bernoulli score
            scores_mu = (rewards - mu_a) / (mu_a * (1.0 - mu_a))

            contrib_mu = (1+scores_mu * G[1:]) / T
            np.add.at(grad_hat_mu, actions, contrib_mu)
            #import ipdb; ipdb.set_trace()

        grad_hat_mu /= n_reps # compute te expectation term
        
        # compute sigma_pi0 ==========================================
        offline_actions = np.asarray(offline_data["all_actions"], dtype=int).flatten()
        N_a = np.bincount(offline_actions, minlength=n_actions).astype(float) # (|A|,)
        
        if np.any(N_a == 0):
            raise ValueError("Some arm has N_a = 0 in offline data.")

        Sigma_diag = (offline_data["T"]/N_a) * hat_mu * (1.0 - hat_mu)
        var_delta = np.sum(Sigma_diag * grad_hat_mu**2)
        
        # compute se from g*Sigma*g==========================================
        se = float(np.sqrt(var_delta) / np.sqrt(offline_data["T"]))
        Sigma = np.diag(Sigma_diag)
        return se, grad_hat_mu, Sigma

    def run(self, offline_data, alphas, n_reps):
        pi0_summary = compute_arm_mean_std(
            all_actions=offline_data["all_actions"],
            all_rewards=offline_data["all_rewards"],
            n_actions=self.true_env.n_actions,
        )

        hat_mu = np.asarray(pi0_summary["arm_means"][0], dtype=float)

        if np.any(np.isnan(hat_mu)):
            raise ValueError(
                "Some arm mean is NaN; every arm must be observed offline at least once."
            )

        hat_mu = self._clip_mu(hat_mu)
        lambda_hat = hat_mu

        imagined_env = BernoulliRewardEnv(mus=hat_mu)

        self.result = bandit_exp_runner(
            env=imagined_env,
            algo_builder=self.algo_builder2,
            T=self.T,
            n_reps=n_reps,
            base_exp_seed=self.algo_seed,
            table_seed=self.table_seed,
            table_renew=True,
        )

        se, gradient, Sigma = self.compute_se(
            all_actions=self.result["all_actions"],
            all_rewards=self.result["all_rewards"],
            hat_mu=hat_mu,
            offline_data=offline_data,
        )

        center_se = float(self.result["se_avg_reward"])
        ci_widths = {alpha: norm.ppf(1 - alpha / 2) * se for alpha in alphas}

        center = float(self.result["mean_avg_reward"])

        return {
            "center": center,
            "center_se": center_se,
            "ci_width_proj": ci_widths,
            "se": se,
            "pi0_summary": pi0_summary,
            "pi1_img": self.result,
            "lambda_hat": lambda_hat,
            "gradient": gradient,
            "Sigma": Sigma,
        }
