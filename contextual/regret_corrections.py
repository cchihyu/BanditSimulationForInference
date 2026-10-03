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
                    benchmark='unrestricted', epsilon=0., seed=2026, *,
                    backend='python', block_size=128, progress=False, trajectory_offset=0,
                    return_samples=False):
    if benchmark not in {'unrestricted','epsilon_floor'} or not 0 <= epsilon <= 1:
        raise ValueError('Invalid benchmark')
    if horizon < 1 or reps < (1 if return_samples else 2) or trajectory_offset < 0:
        raise ValueError('Positive horizon and >=2 rollouts required')
    from .accelerated import simulate
    fast=simulate(environment,policy_builder,context_sampler,horizon,reps,seed,
                  benchmark_epsilon=epsilon if benchmark=='epsilon_floor' else 0.,
                  backend=backend,block_size=block_size,progress=progress,
                  description='Policy value / regret',trajectory_offset=trajectory_offset)
    if fast is not None:
        values=fast['expected']; regrets=fast['regrets']
        if return_samples: return dict(values=values,regrets=regrets,backend='numba')
        return dict(value=float(values.mean()),value_se=float(values.std(ddof=1)/np.sqrt(reps)),
                    regret=float(regrets.mean()),regret_se=float(regrets.std(ddof=1)/np.sqrt(reps)),backend='numba')
    values = np.zeros(reps); regrets = np.zeros(reps)
    repetitions=range(reps)
    if progress:
        from tqdm.auto import tqdm
        repetitions=tqdm(repetitions,desc='Python policy rollouts',leave=False)
    for rep in repetitions:
        seeds = np.random.SeedSequence([seed,rep+trajectory_offset]).spawn(4)
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
    if return_samples: return dict(values=values,regrets=regrets,backend='python')
    return dict(value=float(values.mean()),value_se=float(values.std(ddof=1)/np.sqrt(reps)),
                regret=float(regrets.mean()),regret_se=float(regrets.std(ddof=1)/np.sqrt(reps)))


def regret_rollouts_horizons(environment, policy_builder, context_sampler, horizons, reps,
                             benchmark='unrestricted', epsilon=0., seed=2026, *,
                             backend='python', block_size=128, progress=False,
                             trajectory_offset=0, return_samples=False):
    checkpoints=sorted(set(int(h) for h in horizons))
    if not checkpoints or checkpoints[0]<1 or reps<1:
        raise ValueError('Positive horizons and rollouts required')
    if benchmark not in {'unrestricted','epsilon_floor'} or not 0<=epsilon<=1:
        raise ValueError('Invalid benchmark')
    from .accelerated import simulate_horizons
    fast=simulate_horizons(environment,policy_builder,context_sampler,checkpoints,reps,seed,
                           benchmark_epsilon=epsilon if benchmark=='epsilon_floor' else 0.,
                           backend=backend,block_size=block_size,progress=progress,
                           description='Multi-horizon policy value / regret',trajectory_offset=trajectory_offset)
    if fast is not None:
        values,regrets=fast['expected'],fast['regrets']; used_backend='numba'
    else:
        values=np.zeros((reps,len(checkpoints)));regrets=np.zeros_like(values)
        checkpoint_index={h:i for i,h in enumerate(checkpoints)}
        for local_rep in range(reps):
            rep=local_rep+trajectory_offset
            seeds=np.random.SeedSequence([seed,rep]).spawn(4)
            xr,ar,rr=[np.random.default_rng(s) for s in seeds[:3]]
            policy=policy_builder(int(seeds[3].generate_state(1)[0]))
            xs=context_sampler(xr,checkpoints[-1]);aa=np.zeros(checkpoints[-1],dtype=int);yy=np.zeros(checkpoints[-1])
            value_sum=0.;regret_sum=0.
            for t,x in enumerate(xs):
                history=dict(contexts=xs[:t],actions=aa[:t],rewards=yy[:t])
                probs=np.asarray(policy.action_probs(x,history=history),dtype=float);k=len(probs)
                if benchmark=='epsilon_floor' and np.min(probs)<epsilon/k-1e-10:
                    raise ValueError('Evaluation policy violates benchmark probability floor')
                means=np.array([environment.mean(x,a) for a in range(k)])
                oracle=(1-epsilon)*means.max()+epsilon*means.mean() if benchmark=='epsilon_floor' else means.max()
                expected=float(probs@means);value_sum+=expected;regret_sum+=oracle-expected
                aa[t]=ar.choice(k,p=probs);yy[t]=environment.sample(x,aa[t],rr)
                if hasattr(policy,'update'):policy.update(x,int(aa[t]),float(yy[t]))
                if t+1 in checkpoint_index:
                    j=checkpoint_index[t+1];values[local_rep,j]=value_sum/(t+1);regrets[local_rep,j]=regret_sum
        used_backend='python'
    if return_samples:return dict(horizons=checkpoints,values=values,regrets=regrets,backend=used_backend)
    return {T:dict(value=float(values[:,j].mean()),value_se=float(values[:,j].std(ddof=1)/np.sqrt(reps)),
                   regret=float(regrets[:,j].mean()),regret_se=float(regrets[:,j].std(ddof=1)/np.sqrt(reps)),
                   backend=used_backend) for j,T in enumerate(checkpoints)}


def search_mixture_regret(model, policy_builder, context_sampler, horizon,
                          candidate_count=20, components=3, screen_rollouts=50,
                          refine_rollouts=200, n_refine=5, benchmark='unrestricted',
                          epsilon=0., seed=2026, mc_error_probability=.05, *,
                          backend='python', block_size=128, progress=False):
    if candidate_count < 1 or components < 1 or n_refine < 1 or not 0 < mc_error_probability < 1:
        raise ValueError('Invalid mixture search settings')
    rng = np.random.default_rng(seed)
    candidates = [model]  # the exact fitted Gaussian working environment
    for _ in range(candidate_count-1):
        candidates.append(MixtureCandidate(model,[standardized_mixture(rng,components) for _ in range(model.n_actions)]))
    screened = [regret_rollouts(e,policy_builder,context_sampler,horizon,screen_rollouts,
                               benchmark,epsilon,seed+10000,backend=backend,block_size=block_size,progress=progress) for e in candidates]
    order = np.argsort([r['regret'] for r in screened])[-min(n_refine,candidate_count):]
    refined = []
    z = norm.ppf(1-mc_error_probability/len(order))
    for j in order:
        r = regret_rollouts(candidates[j],policy_builder,context_sampler,horizon,refine_rollouts,
                            benchmark,epsilon,seed+20000,backend=backend,block_size=block_size,progress=progress)
        refined.append(dict(candidate=int(j),**r,mc_adjusted_regret=max(0.,r['regret']+z*r['regret_se'])))
    chosen = max(refined,key=lambda r:r['mc_adjusted_regret'])
    return dict(B_T=chosen['mc_adjusted_regret'],method='gaussian_mixture_search',
                guarantee='empirical finite-library search; normal MC adjustment is approximate',
                proxy_constraint='component-bound certified candidates within fitted proxy budget',
                candidate_count=candidate_count,screened=screened,refined=refined,
                raw_max_regret=max(0.,max(r['regret'] for r in refined)),benchmark=benchmark)


def search_mixture_regret_horizons(model, policy_builder, context_sampler, horizons,
                                   candidate_count=20, components=3, screen_rollouts=50,
                                   refine_rollouts=200, n_refine=5, benchmark='unrestricted',
                                   epsilon=0., seed=2026, mc_error_probability=.05, *,
                                   backend='python', block_size=128, progress=False):
    """Share each candidate trajectory across all requested horizons."""
    checkpoints=sorted(set(int(h) for h in horizons))
    if candidate_count<1 or components<1 or n_refine<1 or not 0<mc_error_probability<1:
        raise ValueError('Invalid mixture search settings')
    rng=np.random.default_rng(seed);candidates=[model]
    for _ in range(candidate_count-1):
        candidates.append(MixtureCandidate(model,[standardized_mixture(rng,components) for _ in range(model.n_actions)]))
    screened_samples=[regret_rollouts_horizons(e,policy_builder,context_sampler,checkpoints,screen_rollouts,
                      benchmark,epsilon,seed+10000,backend=backend,block_size=block_size,
                      progress=progress,return_samples=True) for e in candidates]
    screen_means=np.array([[s['regrets'][:,j].mean() for j in range(len(checkpoints))] for s in screened_samples])
    selected={T:np.argsort(screen_means[:,j])[-min(n_refine,candidate_count):] for j,T in enumerate(checkpoints)}
    union=sorted(set(int(i) for indices in selected.values() for i in indices))
    refined_samples={i:regret_rollouts_horizons(candidates[i],policy_builder,context_sampler,checkpoints,refine_rollouts,
                    benchmark,epsilon,seed+20000,backend=backend,block_size=block_size,
                    progress=progress,return_samples=True) for i in union}
    output={}
    for j,T in enumerate(checkpoints):
        order=selected[T];z=norm.ppf(1-mc_error_probability/len(order));refined=[]
        for i in order:
            sample=refined_samples[int(i)]['regrets'][:,j]
            mean=float(sample.mean());se=float(sample.std(ddof=1)/np.sqrt(refine_rollouts))
            refined.append(dict(candidate=int(i),regret=mean,regret_se=se,
                                mc_adjusted_regret=max(0.,mean+z*se),backend=refined_samples[int(i)]['backend']))
        chosen=max(refined,key=lambda r:r['mc_adjusted_regret'])
        output[T]=dict(B_T=chosen['mc_adjusted_regret'],method='gaussian_mixture_search',
            guarantee='empirical finite-library search; normal MC adjustment is approximate',
            proxy_constraint='component-bound certified candidates within fitted proxy budget',
            candidate_count=candidate_count,
            screened=[dict(regret=float(screen_means[i,j]),backend=screened_samples[i]['backend']) for i in range(candidate_count)],
            refined=refined,raw_max_regret=max(0.,max(r['regret'] for r in refined)),benchmark=benchmark)
    return output


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
