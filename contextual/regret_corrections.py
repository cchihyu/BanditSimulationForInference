"""Empirical regret corrections: no finite-library worst-case guarantee."""
import numpy as np
from scipy.stats import norm


def standardized_mixture(rng, components):
    """Center and certify a proxy <=1 using component bounds plus Hoeffding.

    For a Gaussian mixture, max component variance + range(means)^2/4
    is a valid proxy bound. This avoids finite-grid proxy certification.
    """
    w = rng.dirichlet(np.ones(components))
    m = rng.normal(size=components); m -= w@m
    s = rng.uniform(.1,1.,components)
    bound = max(s*s)+np.ptp(m)**2/4
    return w,m/np.sqrt(bound),s/np.sqrt(bound)


class MixtureCandidate:
    def __init__(self, working_model, mixtures):
        self.model, self.mixtures = working_model, mixtures
        self.n_actions = working_model.n_actions

    def mean(self,x,a,params=None):
        return self.model.mean(x,a)

    def sample(self,x,a,rng,params=None):
        w,m,s = self.mixtures[a]; j = rng.choice(len(w),p=w)
        return self.mean(x,a)+np.sqrt(self.model.variance(x,a))*rng.normal(m[j],s[j])


def regret_rollouts(environment, policy_builder, context_sampler, horizon, reps,
                    benchmark='unrestricted', epsilon=0., seed=2026):
    if benchmark not in {'unrestricted','epsilon_floor'} or not 0 <= epsilon <= 1:
        raise ValueError('Invalid benchmark')
    if horizon < 1 or reps < 2:
        raise ValueError('Positive horizon and >=2 rollouts required')
    values = np.zeros(reps); regrets = np.zeros(reps)
    for rep in range(reps):
        seeds = np.random.SeedSequence([seed,rep]).spawn(4)
        xr,ar,rr = [np.random.default_rng(s) for s in seeds[:3]]
        policy = policy_builder(int(seeds[3].generate_state(1)[0]))
        xs = context_sampler(xr,horizon)
        aa = np.zeros(horizon,dtype=int); yy = np.zeros(horizon)
        for t,x in enumerate(xs):
            history = dict(contexts=xs[:t],actions=aa[:t],rewards=yy[:t])
            probs = np.asarray(policy.action_probs(x,history=history),dtype=float)
            if np.any(probs < 0) or not np.isfinite(probs).all() or not np.isclose(probs.sum(),1.):
                raise ValueError('Invalid policy probabilities')
            k = len(probs)
            if benchmark == 'epsilon_floor' and np.min(probs) < epsilon/k-1e-10:
                raise ValueError('Evaluation policy violates benchmark probability floor')
            means = np.array([environment.mean(x,a) for a in range(k)])
            oracle = means.max()
            if benchmark == 'epsilon_floor': oracle = (1-epsilon)*oracle+epsilon*means.mean()
            expected = probs@means
            regrets[rep] += oracle-expected
            values[rep] += expected/horizon
            aa[t] = ar.choice(k,p=probs); yy[t] = environment.sample(x,aa[t],rr)
            if hasattr(policy,'update'): policy.update(x,int(aa[t]),float(yy[t]))
    return dict(value=float(values.mean()),value_se=float(values.std(ddof=1)/np.sqrt(reps)),
                regret=float(regrets.mean()),regret_se=float(regrets.std(ddof=1)/np.sqrt(reps)))


def search_mixture_regret(model, policy_builder, context_sampler, horizon,
                          candidate_count=20, components=3, screen_rollouts=50,
                          refine_rollouts=200, n_refine=5, benchmark='unrestricted',
                          epsilon=0., seed=2026, mc_error_probability=.05):
    if candidate_count < 1 or components < 1 or n_refine < 1 or not 0 < mc_error_probability < 1:
        raise ValueError('Invalid mixture search settings')
    rng = np.random.default_rng(seed)
    candidates = [model]  # the exact fitted Gaussian working environment
    for _ in range(candidate_count-1):
        candidates.append(MixtureCandidate(model,[standardized_mixture(rng,components) for _ in range(model.n_actions)]))
    screened = [regret_rollouts(e,policy_builder,context_sampler,horizon,screen_rollouts,
                               benchmark,epsilon,seed+10000) for e in candidates]
    order = np.argsort([r['regret'] for r in screened])[-min(n_refine,candidate_count):]
    refined = []
    z = norm.ppf(1-mc_error_probability/len(order))
    for j in order:
        r = regret_rollouts(candidates[j],policy_builder,context_sampler,horizon,refine_rollouts,
                            benchmark,epsilon,seed+20000)
        refined.append(dict(candidate=int(j),**r,mc_adjusted_regret=max(0.,r['regret']+z*r['regret_se'])))
    chosen = max(refined,key=lambda r:r['mc_adjusted_regret'])
    return dict(B_T=chosen['mc_adjusted_regret'],method='gaussian_mixture_search',
                guarantee='empirical finite-library search; normal MC adjustment is approximate',
                proxy_constraint='component-bound certified candidates within fitted proxy budget',
                candidate_count=candidate_count,screened=screened,refined=refined,
                raw_max_regret=max(0.,max(r['regret'] for r in refined)),benchmark=benchmark)


def minimax_type_regret(horizon, n_actions, parameter_dimension, proxy_upper,
                       constant=1., formula='linear_dimension', include_log=True):
    """Configurable sensitivity formulas, NOT policy-specific proven bounds."""
    if min(horizon,n_actions,parameter_dimension,proxy_upper,constant) <= 0:
        raise ValueError('Bound inputs must be positive')
    if formula == 'linear_dimension': factor = parameter_dimension
    elif formula == 'mab_rate': factor = np.sqrt(n_actions)
    else: raise ValueError('Unknown minimax-inspired formula')
    log_factor = np.log(max(2,horizon)) if include_log else 1.
    bound = constant*np.sqrt(proxy_upper)*factor*np.sqrt(horizon*log_factor)
    return dict(B_T=float(bound),method='minimax_bound',formula=formula,
                guarantee='minimax-inspired sensitivity formula; not a certified bound for this policy',
                benchmark='unrestricted',proxy_upper=float(proxy_upper))


def expand_interval(interval, B_T, horizon):
    if B_T < 0 or horizon <= 0: raise ValueError('Invalid correction')
    return [float(interval[0]-B_T/horizon),float(interval[1]+B_T/horizon)]
