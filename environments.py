import numpy as np
from dataclasses import dataclass
# a env holder, a normal and binary reward model

class RewardEnv:
    """
    Base class for reward environments.
    """
    def sample(self, action):
        raise NotImplementedError

    def mean_reward(self, action):
        raise NotImplementedError

    def best_action(self):
        means = [self.mean_reward(a) for a in range(self.n_actions)]
        return int(np.argmax(means))


@dataclass
class BernoulliRewardEnv(RewardEnv):
    """
    Bernoulli reward environment:
        R(a) ~ Bernoulli(p_a)

    Each pull returns 0 or 1.
    """
    mus: np.ndarray
 
    def __post_init__(self): 
        self.mus = np.asarray(self.mus, dtype=float)
        assert self.mus.ndim == 1, "probs must be a 1D array"
        assert np.all((0 <= self.mus) & (self.mus <= 1)), "probs must be in [0,1]"
        self.n_actions = len(self.mus)

    def mean_reward(self, action):
        return float(self.mus[action])

    def sample_table(self, T, seed = None): 
        rng = np.random.default_rng(seed if seed is not None else self.seed)

        rewards = rng.binomial(
            n=1,
            p=self.mus[:, None],   # shape (n_actions, 1)
            size=(self.n_actions, T)
        ).astype(float)

        return {"seed": seed,"table": rewards,}    
        

@dataclass
class NormalRewardEnv(RewardEnv):
    """
    Normal reward environment:
        R(a) ~ Normal(mu_a, sigma_a^2)

    If sigma is scalar, same noise level is used for all actions.
    If sigma is an array, each action has its own std.
    """
    mus: np.ndarray
    sigma: np.ndarray
 
    def __post_init__(self): # check whether the input is valid or not
        self.mus = np.asarray(self.mus, dtype=float)
        assert self.mus.ndim == 1, "mus must be a 1D array"

        if np.isscalar(self.sigma):
            assert self.sigma > 0, "sigma must be positive"
            self.sigmas = np.full_like(self.mus, float(self.sigma), dtype=float)
        else:
            self.sigmas = np.asarray(self.sigma, dtype=float)
            assert self.sigmas.shape == self.mus.shape, "sigma must match mus shape"
            assert np.all(self.sigmas > 0), "all sigma values must be positive"

        self.n_actions = len(self.mus)
    
    def mean_reward(self, action): # return the mean reward given an action
        return float(self.mus[action])

    def sample_table(self, T, seed = None): # sample a response table
        assert T > 0, "T must be positive"
        rng = np.random.default_rng(seed if seed is not None else self.seed)

        rewards = rng.normal(
            loc=self.mus[:, None],       # shape (n_actions, 1)
            scale=self.sigmas[:, None],  # shape (n_actions, 1)
            size=(self.n_actions, T)
        )
        return {"seed": seed, "table": rewards}


@dataclass
class BetaRewardEnv(RewardEnv):
    """
    Beta reward environment: R(a) ~ Beta(alpha_a, beta_a).
    Exposes sigmas = [0.5] * K as a sub-Gaussian upper bound,
    suitable for Normal-approximation (beta-normal) inference.
    """
    alpha_params: np.ndarray
    beta_params: np.ndarray

    def __post_init__(self):
        self.alpha_params = np.asarray(self.alpha_params, dtype=float)
        self.beta_params = np.asarray(self.beta_params, dtype=float)
        assert self.alpha_params.ndim == 1 and self.beta_params.ndim == 1
        assert self.alpha_params.shape == self.beta_params.shape
        assert np.all(self.alpha_params > 0) and np.all(self.beta_params > 0)
        self.n_actions = len(self.alpha_params)
        self.mus = self.alpha_params / (self.alpha_params + self.beta_params)
        self.sigmas = np.full(self.n_actions, 0.5)

    def mean_reward(self, action):
        return float(self.mus[action])

    def sample_table(self, T, seed=None):
        rng = np.random.default_rng(seed)
        rewards = rng.beta(
            self.alpha_params[:, None],
            self.beta_params[:, None],
            size=(self.n_actions, T),
        )
        return {"seed": seed, "table": rewards}
