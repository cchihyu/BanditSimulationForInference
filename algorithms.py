import numpy as np


def clip_probs(vec, pi_clip):
    clipped = np.maximum(vec, pi_clip)
    excess = clipped.sum() - 1.0
    if excess <= 0:
        return vec  # Already satisfied or minor rounding

    room_above_floor = clipped - pi_clip
    total_room = room_above_floor.sum()

    if total_room > 0:
        clipped -= excess * (room_above_floor / total_room)
    return clipped


class BanditAlgo:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.counts = np.zeros(n_actions, dtype=int)  # counter for each action
        self.reward_sums = np.zeros(n_actions, dtype=float)  # collecting rewards
        self.t = 0

    def select_action(self):
        raise NotImplementedError

    def update(self, action, reward):  # update the counter
        self.counts[action] += 1
        self.reward_sums[action] += reward
        self.t += 1

    def estimated_means(self) -> np.ndarray:
        means = np.zeros(self.n_actions, dtype=float)
        mask = self.counts > 0
        means[mask] = self.reward_sums[mask] / self.counts[mask]
        return means


class EpsilonGreedy(BanditAlgo):
    def __init__(self, n_actions, epsilon=0.1, seed=None):
        super().__init__(n_actions)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

    def _probs(self):
        if self.t < self.n_actions:
            # initialisation: pull arm self.t deterministically
            p = np.zeros(self.n_actions)
            p[self.t] = 1.0
        else:
            p = np.full(self.n_actions, self.epsilon / self.n_actions)
            p[int(np.argmax(self.estimated_means()))] += 1.0 - self.epsilon
        return p

    def select_action(self):
        p = self._probs()
        return int(self.rng.choice(self.n_actions, p=p))

    def select_action_with_probs(self):
        p = self._probs()
        a = int(self.rng.choice(self.n_actions, p=p))
        return a, p


# impelment select_action: given the history and return an specific action
class UniformSampling(BanditAlgo):
    def __init__(self, n_actions, seed=None):
        super().__init__(n_actions)
        self.rng = np.random.default_rng(seed)

    def select_action(self):
        return int(self.rng.integers(self.n_actions))

    def select_action_with_probs(self):
        p = np.full(self.n_actions, 1.0 / self.n_actions, dtype=float)
        a = int(self.rng.choice(self.n_actions, p=p))
        return a, p


class ExploreThenCommit(BanditAlgo):
    def __init__(self, n_actions, m, seed=None):
        super().__init__(n_actions)
        self.m = m  # pulling each arm m times
        self.committed_action = None
        self.rng = np.random.default_rng(seed)

    def select_action(self):
        if self.committed_action is not None:
            return self.committed_action

        # exploration phase
        for a in range(self.n_actions):
            if self.counts[a] < self.m:
                return a

        # If already explored, stick to the optimal arm for every call
        means = self.estimated_means()
        self.committed_action = int(np.argmax(means))
        return self.committed_action

    def select_action_with_probs(self):
        action = self.select_action()
        p = np.zeros(self.n_actions, dtype=float)
        p[action] = 1.0
        return action, p


# batch greedy: explore a bit, then select the top k arms to perform in a batch
# suppose the batch size = 3, and the current best three arms are [1,4,2]. The next three call would return 1, 4, and 2.
class BatchExploreThenGreedy(BanditAlgo):
    def __init__(self, n_actions, m, batch_size, seed=None):
        super().__init__(n_actions)
        self.rng = np.random.default_rng(seed)
        self.m = m
        self.batch_size = batch_size

        self.current_batch = []
        self.batch_pos = 0

    def _build_greedy_batch(self):
        means = self.estimated_means()

        # pick top-k arms by empirical mean
        top_arms = np.argsort(means)[-self.batch_size :][::-1]

        self.current_batch = list(map(int, top_arms))
        self.batch_pos = 0

    def select_action(self):
        # exploration phase: make sure each arm is sampled m times
        for a in range(self.n_actions):
            if self.counts[a] < self.m:
                return a

        # greedy batch phase
        if self.batch_pos >= len(self.current_batch):
            self._build_greedy_batch()

        action = self.current_batch[self.batch_pos]
        self.batch_pos += 1
        return action

    def select_action_with_probs(self):
        action = self.select_action()
        p = np.zeros(self.n_actions, dtype=float)
        p[action] = 1.0
        return action, p


class UCB(BanditAlgo):
    def __init__(self, n_actions, c=np.sqrt(2), seed=None):
        self.rng = np.random.default_rng(seed)
        super().__init__(n_actions)
        self.c = c

    def select_action(self):
        # pull each arm once before applying UCB
        for a in range(self.n_actions):
            if self.counts[a] == 0:
                return a

        means = self.estimated_means()
        bonus = self.c * np.sqrt(np.log(self.t + 1) / self.counts)
        ucb_values = means + bonus
        return int(np.argmax(ucb_values))

    def select_action_with_probs(self):
        action = self.select_action()
        p = np.zeros(self.n_actions, dtype=float)
        p[action] = 1.0
        return action, p


class TSBernoulli(BanditAlgo):
    def __init__(
        self,
        n_actions,
        seed=None,
        alpha0=1.0,
        beta0=1.0,
        pi_clip=0.01,
    ):
        super().__init__(n_actions)
        # alpha and beta are the prior of the beta distribution, which should be adjusted later.
        self.rng = np.random.default_rng(seed)
        self.alpha = np.full(n_actions, alpha0, dtype=float)
        self.beta = np.full(n_actions, beta0, dtype=float)
        self.pi_clip = pi_clip

    def select_action(self):
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def select_action_with_probs(self, n_samples=5000, clip=0.01):
        samples = self.rng.beta(self.alpha, self.beta, size=(n_samples, self.n_actions))
        winners = np.argmax(samples, axis=1)
        p = np.bincount(winners, minlength=self.n_actions).astype(float) / n_samples
        p = clip_probs(p, self.pi_clip)
        # p = np.maximum(p, 1e-8)
        # p /= p.sum()
        a = int(np.argmax(self.rng.beta(self.alpha, self.beta)))
        return a, p

    def update(self, action: int, reward: float) -> None:
        super().update(action, reward)

        # reward should be 0 or 1
        self.alpha[action] += reward
        self.beta[action] += 1.0 - reward


class TSNormal(BanditAlgo):
    def __init__(
        self,
        n_actions,
        seed=None,
        prior_mean=0.0,
        prior_var=1.0,
        obs_sigma=1.0,
        pi_clip=0.01,
    ):
        super().__init__(n_actions)
        self.rng = np.random.default_rng(seed)

        # the variance below are the prior of the normal distribution, which should be adjusted later.
        self.obs_var = float(obs_sigma) ** 2
        self.post_mean = np.full(n_actions, prior_mean, dtype=float)
        self.post_var = np.full(n_actions, prior_var, dtype=float)
        self.pi_clip = pi_clip

    def select_action_with_probs(self, n_samples=5000):
        # estimate pi_t via Monte Carlo over current posterior
        samples = self.rng.normal(
            self.post_mean, np.sqrt(self.post_var), size=(n_samples, self.n_actions)
        )
        winners = np.argmax(samples, axis=1)
        p = np.bincount(winners, minlength=self.n_actions).astype(float) / n_samples
        p = clip_probs(p, self.pi_clip)
        # p = np.maximum(p, 1e-8)
        # p /= p.sum()

        # action drawn independently from the prob estimate to avoid correlation
        a = int(np.argmax(self.rng.normal(self.post_mean, np.sqrt(self.post_var))))
        return a, p

    def select_action(self) -> int:
        samples = self.rng.normal(self.post_mean, np.sqrt(self.post_var))
        return int(np.argmax(samples))

    def update(self, action: int, reward: float) -> None:
        super().update(action, reward)

        v0 = self.post_var[action]
        m0 = self.post_mean[action]
        sig2 = self.obs_var

        v1 = 1.0 / (1.0 / v0 + 1.0 / sig2)
        m1 = v1 * (m0 / v0 + reward / sig2)

        self.post_var[action] = v1
        self.post_mean[action] = m1
