import numpy as np
from scipy.stats import norm

from algorithms import (
    BatchExploreThenGreedy,
    EpsilonGreedy,
    ExploreThenCommit,
    TSBernoulli,
    TSNormal,
    UCB,
    UniformSampling,
)
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


def _sample_categorical_rows(probs, rng):
    draws = rng.random(probs.shape[0])
    cdf = np.cumsum(probs, axis=1)
    return (draws[:, None] > cdf[:, :-1]).sum(axis=1).astype(int)


def _sample_vectorized_rewards(env, actions, rng):
    if isinstance(env, NormalRewardEnv):
        return rng.normal(env.mus[actions], env.sigmas[actions]).astype(float)
    if isinstance(env, BernoulliRewardEnv):
        return rng.binomial(1, env.mus[actions]).astype(float)
    if isinstance(env, BetaRewardEnv):
        return rng.beta(env.alpha_params[actions], env.beta_params[actions]).astype(float)
    raise TypeError(f"Unsupported environment for vectorized MAB runner: {type(env)!r}")


def _summarize_rollouts(env, all_actions, all_rewards, all_probs, base_exp_seed, T):
    total_rewards = all_rewards.sum(axis=1)
    avg_rewards = total_rewards / T
    n_reps = all_rewards.shape[0]
    return {
        "table_seed": None,
        "exp_seed": base_exp_seed,
        "env": env,
        "n_rep": n_reps,
        "T": T,
        "all_actions": all_actions,
        "all_rewards": all_rewards,
        "all_probs": all_probs,
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


def bandit_exp_runner_vectorized(
    env,
    algo_builder,
    T,
    n_reps=1,
    base_exp_seed=1013,
    table_seed=2026,
    table_renew=False,
    adaptive=False,
):
    """
    Vectorized MAB rollout simulator over independent inner Monte Carlo reps.

    This is distributionally equivalent to ``bandit_exp_runner`` for the
    standard policies below, but it intentionally uses a single vectorized RNG
    stream rather than one ``default_rng(seed + rep)`` object per replication.
    Therefore individual simulated trajectories are not bit-for-bit identical to
    the legacy runner, while the bandit state updates and reward law are the
    same.
    """
    if adaptive or not table_renew:
        return bandit_exp_runner(
            env=env,
            algo_builder=algo_builder,
            T=T,
            n_reps=n_reps,
            base_exp_seed=base_exp_seed,
            table_seed=table_seed,
            table_renew=table_renew,
            adaptive=adaptive,
        )

    probe = algo_builder(base_exp_seed)
    supported = (
        UniformSampling,
        EpsilonGreedy,
        ExploreThenCommit,
        BatchExploreThenGreedy,
        UCB,
        TSNormal,
        TSBernoulli,
    )
    if not isinstance(probe, supported):
        return bandit_exp_runner(
            env=env,
            algo_builder=algo_builder,
            T=T,
            n_reps=n_reps,
            base_exp_seed=base_exp_seed,
            table_seed=table_seed,
            table_renew=table_renew,
            adaptive=adaptive,
        )

    n_actions = env.n_actions
    algo_rng = np.random.default_rng(base_exp_seed)
    reward_rng = np.random.default_rng(table_seed)
    all_actions = np.zeros((n_reps, T), dtype=int)
    all_rewards = np.zeros((n_reps, T), dtype=float)
    counts = np.zeros((n_reps, n_actions), dtype=int)
    reward_sums = np.zeros((n_reps, n_actions), dtype=float)
    rows = np.arange(n_reps)

    if isinstance(probe, TSNormal):
        post_mean = np.tile(probe.post_mean, (n_reps, 1)).astype(float)
        post_var = np.tile(probe.post_var, (n_reps, 1)).astype(float)
        obs_var = float(probe.obs_var)
    elif isinstance(probe, TSBernoulli):
        alpha = np.tile(probe.alpha, (n_reps, 1)).astype(float)
        beta = np.tile(probe.beta, (n_reps, 1)).astype(float)
    elif isinstance(probe, ExploreThenCommit):
        committed = np.full(n_reps, -1, dtype=int)
    elif isinstance(probe, BatchExploreThenGreedy):
        batch_size = int(probe.batch_size)
        current_batch = np.zeros((n_reps, batch_size), dtype=int)
        batch_pos = np.full(n_reps, batch_size, dtype=int)

    for t in range(T):
        if isinstance(probe, UniformSampling):
            actions = algo_rng.integers(n_actions, size=n_reps, dtype=int)

        elif isinstance(probe, EpsilonGreedy):
            if t < n_actions:
                actions = np.full(n_reps, t, dtype=int)
            else:
                means = np.divide(
                    reward_sums,
                    counts,
                    out=np.zeros_like(reward_sums, dtype=float),
                    where=counts > 0,
                )
                greedy = np.argmax(means, axis=1)
                probs = np.full((n_reps, n_actions), probe.epsilon / n_actions)
                probs[rows, greedy] += 1.0 - probe.epsilon
                actions = _sample_categorical_rows(probs, algo_rng)

        elif isinstance(probe, ExploreThenCommit):
            actions = committed.copy()
            active = committed < 0
            if np.any(active):
                needs_explore = counts[active] < probe.m
                explore_any = needs_explore.any(axis=1)
                active_rows = rows[active]
                if np.any(explore_any):
                    explore_rows = active_rows[explore_any]
                    actions[explore_rows] = np.argmax(needs_explore[explore_any], axis=1)
                if np.any(~explore_any):
                    commit_rows = active_rows[~explore_any]
                    means = np.divide(
                        reward_sums[commit_rows],
                        counts[commit_rows],
                        out=np.zeros((commit_rows.size, n_actions), dtype=float),
                        where=counts[commit_rows] > 0,
                    )
                    best = np.argmax(means, axis=1)
                    committed[commit_rows] = best
                    actions[commit_rows] = best

        elif isinstance(probe, BatchExploreThenGreedy):
            needs_explore = counts < probe.m
            explore_any = needs_explore.any(axis=1)
            actions = np.empty(n_reps, dtype=int)
            if np.any(explore_any):
                actions[explore_any] = np.argmax(needs_explore[explore_any], axis=1)
            greedy_rows = rows[~explore_any]
            if greedy_rows.size:
                rebuild = greedy_rows[batch_pos[greedy_rows] >= batch_size]
                if rebuild.size:
                    means = np.divide(
                        reward_sums[rebuild],
                        counts[rebuild],
                        out=np.zeros((rebuild.size, n_actions), dtype=float),
                        where=counts[rebuild] > 0,
                    )
                    current_batch[rebuild] = np.argsort(means, axis=1)[:, -batch_size:][:, ::-1]
                    batch_pos[rebuild] = 0
                pos = batch_pos[greedy_rows]
                actions[greedy_rows] = current_batch[greedy_rows, pos]
                batch_pos[greedy_rows] += 1

        elif isinstance(probe, UCB):
            untried = counts == 0
            explore_any = untried.any(axis=1)
            actions = np.empty(n_reps, dtype=int)
            if np.any(explore_any):
                actions[explore_any] = np.argmax(untried[explore_any], axis=1)
            exploit_rows = rows[~explore_any]
            if exploit_rows.size:
                means = reward_sums[exploit_rows] / counts[exploit_rows]
                bonus = probe.c * np.sqrt(np.log(t + 1) / counts[exploit_rows])
                actions[exploit_rows] = np.argmax(means + bonus, axis=1)

        elif isinstance(probe, TSNormal):
            samples = algo_rng.normal(post_mean, np.sqrt(post_var))
            actions = np.argmax(samples, axis=1)

        elif isinstance(probe, TSBernoulli):
            samples = algo_rng.beta(alpha, beta)
            actions = np.argmax(samples, axis=1)

        rewards = _sample_vectorized_rewards(env, actions, reward_rng)
        all_actions[:, t] = actions
        all_rewards[:, t] = rewards

        counts[rows, actions] += 1
        reward_sums[rows, actions] += rewards

        if isinstance(probe, TSNormal):
            v0 = post_var[rows, actions]
            m0 = post_mean[rows, actions]
            v1 = 1.0 / (1.0 / v0 + 1.0 / obs_var)
            m1 = v1 * (m0 / v0 + rewards / obs_var)
            post_var[rows, actions] = v1
            post_mean[rows, actions] = m1
        elif isinstance(probe, TSBernoulli):
            alpha[rows, actions] += rewards
            beta[rows, actions] += 1.0 - rewards

    return _summarize_rollouts(env, all_actions, all_rewards, None, base_exp_seed, T)


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


def normal_rollout_gradients(
    all_actions,
    all_rewards,
    hat_mu,
    hat_sigma2,
    estimate_sigma=False,
    average=True,
):
    all_actions = np.asarray(all_actions, dtype=int)
    all_rewards = np.asarray(all_rewards, dtype=float)
    hat_mu = np.asarray(hat_mu, dtype=float)
    hat_sigma2 = np.asarray(hat_sigma2, dtype=float)
    n_reps, T = all_actions.shape
    n_actions = hat_mu.shape[0]

    future_excluding_current = (
        np.cumsum(all_rewards[:, ::-1], axis=1)[:, ::-1] - all_rewards
    )
    residuals = all_rewards - hat_mu[all_actions]
    s2 = hat_sigma2[all_actions]
    scores_mu = residuals / s2
    contrib_mu = (1.0 + scores_mu * future_excluding_current) / T

    grad_mu = np.zeros((n_reps, n_actions), dtype=float)
    for action in range(n_actions):
        grad_mu[:, action] = np.sum(
            np.where(all_actions == action, contrib_mu, 0.0),
            axis=1,
        )

    if not estimate_sigma:
        return grad_mu.mean(axis=0) if average else grad_mu

    scores_sigma2 = -0.5 / s2 + residuals**2 / (2.0 * s2**2)
    contrib_sigma2 = (scores_sigma2 * future_excluding_current) / T
    grad_sigma2 = np.zeros((n_reps, n_actions), dtype=float)
    for action in range(n_actions):
        grad_sigma2[:, action] = np.sum(
            np.where(all_actions == action, contrib_sigma2, 0.0),
            axis=1,
        )

    grads = np.empty((n_reps, 2 * n_actions), dtype=float)
    grads[:, 0::2] = grad_mu
    grads[:, 1::2] = grad_sigma2
    return grads.mean(axis=0) if average else grads


def bernoulli_rollout_gradients(all_actions, all_rewards, hat_mu, average=True):
    all_actions = np.asarray(all_actions, dtype=int)
    all_rewards = np.asarray(all_rewards, dtype=float)
    hat_mu = np.clip(np.asarray(hat_mu, dtype=float), 1e-6, 1.0 - 1e-6)
    n_reps, T = all_actions.shape
    n_actions = hat_mu.shape[0]

    future_excluding_current = (
        np.cumsum(all_rewards[:, ::-1], axis=1)[:, ::-1] - all_rewards
    )
    mu_a = hat_mu[all_actions]
    scores = (all_rewards - mu_a) / (mu_a * (1.0 - mu_a))
    contrib = (1.0 + scores * future_excluding_current) / T

    grads = np.zeros((n_reps, n_actions), dtype=float)
    for action in range(n_actions):
        grads[:, action] = np.sum(
            np.where(all_actions == action, contrib, 0.0),
            axis=1,
        )
    return grads.mean(axis=0) if average else grads


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
        grad = normal_rollout_gradients(
            all_actions,
            all_rewards,
            hat_mu,
            hat_sigma2,
            estimate_sigma=self.estimate_sigma,
            average=True,
        )
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
        self.result = bandit_exp_runner_vectorized(
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
        grad_hat_mu = bernoulli_rollout_gradients(
            all_actions,
            all_rewards,
            self._clip_mu(hat_mu),
            average=True,
        )
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

        self.result = bandit_exp_runner_vectorized(
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
        grad = normal_rollout_gradients(
            all_actions,
            all_rewards,
            hat_mu,
            hat_sigma2,
            estimate_sigma=self.estimate_sigma,
            average=True,
        )
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
        self.result = bandit_exp_runner_vectorized(
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
        n_actions = self.true_env.n_actions
        hat_mu = self._clip_mu(hat_mu) # for numerical stability
        grad_hat_mu = bernoulli_rollout_gradients(
            all_actions,
            all_rewards,
            hat_mu,
            average=True,
        )
        
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

        self.result = bandit_exp_runner_vectorized(
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
