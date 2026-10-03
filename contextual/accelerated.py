"""Numba rollouts for default-feature sub-Gaussian SVI and linear policies.

No fastmath or nested threading: offline datasets are the process parallel unit.
Each trajectory has its own seed, independent of block size and worker schedule.
Python custom models/policies retain the reference implementation.
"""
from __future__ import annotations
import numpy as np
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        return lambda function: function


@njit(cache=True)
def sigmoid(z):
    if z >= 0: return 1.0/(1.0+np.exp(-z))
    e=np.exp(z); return e/(1.0+e)


@njit(cache=True)
def bernoulli_proxy_scalar(p):
    if p <= 0. or p >= 1.: return 0.
    delta=1.-2.*p
    if abs(delta)<1e-6: return .25-delta*delta/12.
    return delta/(2.*(np.log1p(-p)-np.log(p)))


@njit(cache=True)
def posterior_update(mean, covariance, x, reward, observation_variance):
    cx=covariance@x
    denominator=observation_variance+x@cx
    updated_mean=mean+cx*((reward-x@mean)/denominator)
    updated_cov=covariance-np.outer(cx,cx)/denominator
    return updated_mean, (updated_cov+updated_cov.T)*.5


@njit(cache=True)
def linear_ts_probabilities(means, covariances, x, n_mc, clip):
    k=len(means); predictions=means@x; sd=np.empty(k)
    for a in range(k): sd[a]=np.sqrt(max(0.,x@covariances[a]@x))
    counts=np.zeros(k)
    for draw in range(n_mc):
        winner=0; best=-np.inf
        for a in range(k):
            value=predictions[a]+sd[a]*np.random.normal()
            if value>best: best=value; winner=a
        counts[winner]+=1.
    probs=counts/n_mc
    for a in range(k): probs[a]=max(probs[a],clip)
    return probs/probs.sum()


@njit(cache=True)
def choose(probs):
    u=np.random.random(); total=0.
    for a in range(len(probs)):
        total+=probs[a]
        if u<total:return a
    return len(probs)-1


@njit(cache=True)
def rollout_block(contexts, seeds, beta, variance_beta, variances, scales, widths,
                  mix_weights, mix_means, mix_sds, env_code, logistic,
                  variance_code, floor, joint, policy_code, epsilon, prior_mean,
                  prior_var, obs_var, n_mc, pi_clip, explore_untried,
                  checkpoints, benchmark_epsilon, probability_floor, gradient_size):
    batch,horizon,d=contexts.shape; k,p=beta.shape
    n_checkpoints=len(checkpoints)
    observed=np.zeros((batch,n_checkpoints)); expected=np.zeros((batch,n_checkpoints)); regrets=np.zeros((batch,n_checkpoints))
    gradients=np.zeros((batch,gradient_size))
    for rep in range(batch):
        np.random.seed(seeds[rep])
        policy_means=np.full((k,p),prior_mean if policy_code==2 else 0.)
        cov=np.zeros((k,p,p)); gram=np.zeros((k,p,p)); info=np.zeros((k,p)); counts=np.zeros(k)
        for a in range(k): cov[a]=np.eye(p)*prior_var
        cumulative_score=np.zeros(gradient_size)
        observed_sum=0.; expected_sum=0.; regret_sum=0.; checkpoint_index=0
        x=np.ones(p)
        for t in range(horizon):
            x[1:]=contexts[rep,t]
            probs=np.full(k,1./k)
            if policy_code==2:
                probs=linear_ts_probabilities(policy_means,cov,x,n_mc,pi_clip)
            elif policy_code==1:
                if explore_untried and np.min(counts)==0:
                    probs=(counts==0).astype(np.float64);probs/=probs.sum()
                else:
                    best=np.argmax(policy_means@x)
                    probs[:]=epsilon/k;probs[best]+=1.-epsilon
            # Match SVI's normalization when requested; regret path uses raw policy probabilities.
            if probability_floor>0:
                for a in range(k): probs[a]=max(probs[a],probability_floor)
                probs/=probs.sum()
            if np.min(probs)<benchmark_epsilon/k-1e-10:
                raise ValueError('Evaluation policy violates benchmark probability floor')
            means=beta@x
            if logistic:
                for a in range(k):means[a]=scales[a]*sigmoid(means[a])
            policy_value=probs@means
            expected_sum+=policy_value
            oracle=(1.-benchmark_epsilon)*np.max(means)+benchmark_epsilon*np.mean(means)
            regret_sum+=oracle-policy_value
            a=choose(probs);mu=means[a]
            v=variances[a]
            if logistic:
                prob=sigmoid(variance_beta[a]@x)
                if variance_code==0:v=scales[a]**2*prob*(1.-prob)
                elif variance_code==1:v=scales[a]**2*bernoulli_proxy_scalar(prob)
                else:v=widths[a]**2/4.
            v=max(v,floor)
            if env_code==1:
                reward=scales[a]*(np.random.random()<mu/scales[a])
            elif env_code==2:
                reward=mu+widths[a]*(2.*np.random.random()-1.)
            elif env_code==3 or env_code==4:
                component=choose(mix_weights[a])
                noise=mix_means[a,component]+mix_sds[a,component]*np.random.normal()
                reward=mu+(np.sqrt(v)*noise if env_code==4 else noise)
            else:reward=mu+np.sqrt(v)*np.random.normal()
            observed_sum+=reward
            if gradient_size>0:
                residual=reward-mu
                dm=1.
                if logistic:
                    prob=mu/scales[a];dm=scales[a]*prob*(1.-prob)
                for j in range(p): cumulative_score[a*p+j]+=x[j]*dm*residual/v
                if joint:
                    variance_score=(residual*residual-v)/(2.*v*v)
                    if logistic:
                        raw=scales[a]**2*prob*(1.-prob)
                        if raw>floor:
                            for j in range(p):cumulative_score[a*p+j]+=variance_score*scales[a]**2*prob*(1.-prob)*(1.-2.*prob)*x[j]
                    elif variances[a]>floor:cumulative_score[k*p+a]+=variance_score
                gradients[rep]+=cumulative_score*(reward/horizon)
            if policy_code==2:
                policy_means[a],cov[a]=posterior_update(policy_means[a],cov[a],x,reward,obs_var)
            elif policy_code==1:
                gram[a]+=np.outer(x,x);info[a]+=x*reward;counts[a]+=1.
                try:policy_means[a]=np.linalg.solve(gram[a],info[a])
                except Exception:policy_means[a]=np.linalg.pinv(gram[a])@info[a]
            if checkpoint_index<n_checkpoints and t+1==checkpoints[checkpoint_index]:
                observed[rep,checkpoint_index]=observed_sum/(t+1)
                expected[rep,checkpoint_index]=expected_sum/(t+1)
                regrets[rep,checkpoint_index]=regret_sum
                checkpoint_index+=1
    return observed,expected,regrets,gradients


def pack(environment, policy, params=None):
    """Return None for models whose arbitrary Python semantics cannot be compiled."""
    from .algorithms import ContextualTSPolicy, ContextualEpsilonGreedyPolicy
    from .simulation import UniformContextualPolicy
    from .environments import ContextualSubGaussianWorkingModel
    from .subgaussian_environments import SubGaussianEnvironment
    from .regret_corrections import MixtureCandidate
    if type(policy) is UniformContextualPolicy:
        policy_code=0
    elif type(policy) in (ContextualTSPolicy,ContextualEpsilonGreedyPolicy):
        if policy.reward_type!='linear_gaussian' or not policy.include_intercept:return None
        policy_code=2 if type(policy) is ContextualTSPolicy else 1
    else:return None
    candidate=type(environment) is MixtureCandidate
    model=environment.model if candidate else environment
    working=type(model) is ContextualSubGaussianWorkingModel
    true=type(model) is SubGaussianEnvironment
    if not working and not true:return None
    if working and model.mean_model.feature_map is not None:return None
    k=model.n_actions;p=model.context_dim+1
    if policy.n_actions!=k:return None
    if hasattr(policy,'context_dim') and policy.context_dim!=model.context_dim:return None
    if working:
        theta=model.lambda_hat_ if params is None else np.asarray(params)
        if model.p!=k*p:return None
        beta=theta[:k*p].reshape(k,p)
        variance_beta=beta if model.propagate else model.frozen_beta.reshape(k,p)
        variances=(theta[k*p:] if model.propagate else model.variances) if model.kind!='scaled_bernoulli' else np.ones(k)
        widths=model.widths if model.widths is not None else np.ones(k)
        var_code={'empirical':0,'variance_proxy':1,'hoeffding':2}[model.variance_method]
        floor=model.floor;joint=model.propagate
        code=4 if candidate else 0
    else:
        beta=model.beta;variance_beta=beta;variances=np.ones(k);widths=model.half_widths
        code={'scaled_bernoulli':1,'uniform':2,'gaussian_mixture':3}[model.kind]
        var_code=0;floor=1e-8;joint=False
    mixes=environment.mixtures if candidate else model.mixtures if true else [(np.ones(1),np.zeros(1),np.ones(1)) for _ in range(k)]
    components=max(len(w) for w,m,s in mixes)
    mw=np.zeros((k,components));mm=mw.copy();ms=np.ones_like(mw)
    for a,(w,m,s) in enumerate(mixes):mw[a,:len(w)]=w;mm[a,:len(w)]=m;ms[a,:len(w)]=s
    numeric=[beta,variance_beta,variances,model.scales,widths,mw,mm,ms]
    numeric=[np.ascontiguousarray(x,dtype=np.float64) for x in numeric]
    n_mc=int(getattr(policy,'n_prob_mc',1))
    if n_mc<1:raise ValueError('ts_prob_mc must be positive')
    return (*numeric,code,model.kind=='scaled_bernoulli',var_code,float(floor),bool(joint),
            policy_code,float(getattr(policy,'epsilon',0.)),float(getattr(policy,'prior_mean',0.)),
            float(getattr(policy,'prior_var',1.)),float(getattr(policy,'obs_sigma',1.))**2,
            n_mc,float(getattr(policy,'pi_clip',0.)),bool(getattr(policy,'explore_untried',False)))


def simulate(environment, policy_builder, sampler, horizon, reps, seed,
             params=None, gradient=False, benchmark_epsilon=0., backend='auto',
             block_size=128, progress=False, description='Rollouts', context_seed=None, trajectory_offset=0):
    """Streaming rollouts: retain per-trajectory summaries, not full histories."""
    if backend not in {'auto','python','numba'}:raise ValueError('Unknown backend')
    if backend=='python':return None
    if not NUMBA_AVAILABLE:
        if backend=='numba':raise ImportError('Install numba to use --backend numba')
        return None
    packed=pack(environment,policy_builder(seed),params)
    if packed is None:
        if backend=='numba':raise ValueError('Numba backend does not support this custom model/policy')
        return None
    if horizon<1 or reps<1 or block_size<1 or trajectory_offset<0:raise ValueError('Invalid rollout sizes')
    size=len(params if params is not None else environment.lambda_hat_) if gradient else 0
    observed=np.empty(reps);expected=np.empty(reps);regrets=np.empty(reps);grads=np.empty((reps,size))
    starts=range(0,reps,block_size)
    if progress:
        from tqdm.auto import tqdm
        starts=tqdm(starts,total=(reps+block_size-1)//block_size,desc=description,unit='block',leave=False)
    for start in starts:
        end=min(reps,start+block_size)
        contexts=[];seeds=[]
        for rep in range(start+trajectory_offset,end+trajectory_offset):
            streams=np.random.SeedSequence([int(seed),rep]).spawn(2)
            xr=np.random.default_rng(streams[0] if context_seed is None else np.random.SeedSequence([int(context_seed),rep]))
            contexts.append(sampler(xr,horizon))
            seeds.append(streams[1].generate_state(1)[0])
        result=rollout_block(np.ascontiguousarray(contexts,dtype=float),np.array(seeds,dtype=np.uint32),
                            *packed,np.array([horizon],dtype=np.int64),float(benchmark_epsilon),
                            1e-12 if gradient else 0.,size)
        observed[start:end],expected[start:end],regrets[start:end],grads[start:end]=result[0][:,0],result[1][:,0],result[2][:,0],result[3]
    return dict(observed=observed,expected=expected,regrets=regrets,trajectory_gradients=grads,backend='numba')


def simulate_horizons(environment, policy_builder, sampler, horizons, reps, seed,
                      benchmark_epsilon=0., backend='auto', block_size=128,
                      progress=False, description='Rollouts', trajectory_offset=0):
    """Simulate to max(horizons) once and return every requested checkpoint."""
    checkpoints=np.array(sorted(set(int(h) for h in horizons)),dtype=np.int64)
    if checkpoints.size==0 or checkpoints[0]<1:raise ValueError('Horizons must be positive')
    if backend not in {'auto','python','numba'}:raise ValueError('Unknown backend')
    if backend=='python':return None
    if not NUMBA_AVAILABLE:
        if backend=='numba':raise ImportError('Install numba to use --backend numba')
        return None
    packed=pack(environment,policy_builder(seed))
    if packed is None:
        if backend=='numba':raise ValueError('Numba backend does not support this custom model/policy')
        return None
    if reps<1 or block_size<1 or trajectory_offset<0:raise ValueError('Invalid rollout sizes')
    observed=np.empty((reps,len(checkpoints)));expected=np.empty_like(observed);regrets=np.empty_like(observed)
    starts=range(0,reps,block_size)
    if progress:
        from tqdm.auto import tqdm
        starts=tqdm(starts,total=(reps+block_size-1)//block_size,desc=description,unit='block',leave=False)
    for start in starts:
        end=min(reps,start+block_size);contexts=[];seeds=[]
        for rep in range(start+trajectory_offset,end+trajectory_offset):
            streams=np.random.SeedSequence([int(seed),rep]).spawn(2)
            contexts.append(sampler(np.random.default_rng(streams[0]),int(checkpoints[-1])))
            seeds.append(streams[1].generate_state(1)[0])
        result=rollout_block(np.ascontiguousarray(contexts,dtype=float),np.array(seeds,dtype=np.uint32),
                             *packed,checkpoints,float(benchmark_epsilon),0.,0)
        observed[start:end],expected[start:end],regrets[start:end]=result[:3]
    return dict(horizons=checkpoints,observed=observed,expected=expected,regrets=regrets,backend='numba')
