"""Run contextual SVI sub-Gaussian coverage studies (MAB: --context_dim 0)."""
from __future__ import annotations
import argparse
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import json
from pathlib import Path
import numpy as np
from .algorithms import ContextualEpsilonGreedyPolicy, ContextualTSPolicy
from .simulation import UniformContextualPolicy, context_sampler, collect_subgaussian_data
from .subgaussian_environments import SubGaussianEnvironment
from .environments import ContextualSubGaussianWorkingModel
from .contextual_bsi import ContextualParametricSVI, contextual_bandit_exp_runner, simulate_svi_summary
from .select_inner_reps import per_trajectory_gradients, estimate_m_from_pilot
from .regret_corrections import search_mixture_regret, minimax_type_regret, regret_rollouts, expand_interval
from .run_contextual import to_serializable
from .baselines import compute_all_contextual_intervals, PerActionLinearRewardModel, PerActionLogisticRewardModel


class ScaledBernoulliBaselineModel:
    def __init__(self, scales): self.scales = np.asarray(scales)
    def fit(self, contexts, actions, rewards, n_actions):
        self.model = PerActionLogisticRewardModel().fit(contexts,actions,rewards/self.scales[actions],n_actions)
        return self
    def predict_all(self, contexts): return self.model.predict_all(contexts)*self.scales


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env',choices=['scaled_bernoulli','gaussian_mixture','uniform'],default='uniform')
    p.add_argument('--n_actions',type=int,default=3)
    p.add_argument('--context_dim',type=int,default=2)
    p.add_argument('--context_var',type=float,default=1.)
    p.add_argument('--beta',type=float,nargs='+',help='Action-major conditional-mean coefficients; logits for Bernoulli')
    p.add_argument('--scales',type=float,nargs='+',default=[1.])
    p.add_argument('--half_widths',type=float,nargs='+',default=[1.])
    p.add_argument('--mixtures_json',help='List of per-arm weights, means, sigmas; offsets are centered automatically')
    p.add_argument('--variance_methods',nargs='+',choices=['hoeffding','variance_proxy','empirical'],default=['empirical'])
    p.add_argument('--propagate_variance_uncertainty',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--variance_floor',type=float,default=1e-8)
    p.add_argument('--proxy_alpha',type=float,default=.49)
    p.add_argument('--proxy_grid_size',type=int,default=4001)
    p.add_argument('--pi0',choices=['uniform','contextual_epsilon','contextual_ts'],default='uniform')
    p.add_argument('--pi1',choices=['uniform','contextual_epsilon','contextual_ts'],default='contextual_epsilon')
    p.add_argument('--epsilon0',type=float,default=.1)
    p.add_argument('--epsilon1',type=float,default=.1)
    p.add_argument('--obs_sigma',type=float,default=1.)
    p.add_argument('--ts_prob_mc',type=int,default=500)
    p.add_argument('--T_values',type=int,nargs='+',default=[50])
    p.add_argument('--T_offline_values',type=int,nargs='+',default=[200])
    p.add_argument('--offline_reps',type=int,default=10)
    p.add_argument('--inner_reps',type=int,default=500)
    p.add_argument('--truth_reps',type=int,default=2000)
    p.add_argument('--alpha',type=float,default=.1)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--select_M',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--M_pilot',type=int,default=100)
    p.add_argument('--M_bootstrap_reps',type=int,default=100)
    p.add_argument('--Mmax',type=int,default=10000)
    p.add_argument('--M_tau',type=float,default=.05)
    p.add_argument('--M_rel_eps',type=float,default=.05)
    p.add_argument('--regret_methods',nargs='+',choices=['gaussian_mixture_search','minimax_bound'],default=['gaussian_mixture_search'])
    p.add_argument('--regret_benchmark',choices=['unrestricted','epsilon_floor'],default='unrestricted')
    p.add_argument('--benchmark_epsilon',type=float,default=.1)
    p.add_argument('--candidate_count',type=int,default=20)
    p.add_argument('--mixture_components',type=int,default=3)
    p.add_argument('--screen_rollouts',type=int,default=50)
    p.add_argument('--refine_rollouts',type=int,default=200)
    p.add_argument('--n_refine',type=int,default=5)
    p.add_argument('--mc_error_probability',type=float,default=.05)
    p.add_argument('--bound_formula',choices=['linear_dimension','mab_rate'],default='linear_dimension')
    p.add_argument('--bound_constant',type=float,default=1.)
    p.add_argument('--bound_log',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--include_baselines',action='store_true')
    p.add_argument('--include_elfcb',action='store_true')
    p.add_argument('--dr_ci_method',choices=['bootstrap','wald'],default='bootstrap')
    p.add_argument('--dr_bootstrap_reps',type=int,default=1000)
    p.add_argument('--cadr_min_samples',type=int,default=30)
    p.add_argument('--cadr_variance_floor',type=float,default=1e-8)
    p.add_argument('--cadr_warmup_sigma',type=float,default=1.)
    p.add_argument('--save_path',type=Path,default=Path('results/contextual_svi.json'))
    p.add_argument('--n_jobs', type=int, default=1, help='Offline dataset workers; -1 uses available CPUs')
    p.add_argument('--backend', choices=['auto','numba','python'], default='auto')
    p.add_argument('--rollout_block_size', type=int, default=128)
    p.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--on_error', choices=['raise','continue'], default='raise',
                   help='Completed replications are checkpointed either way')
    return p


def policy_builder(args,name,epsilon):
    # Linear working learners accept real-valued rewards in both true and
    # Gaussian-surrogate environments. Binary-only learners are not silently reused.
    def build(seed):
        if name == 'uniform': return UniformContextualPolicy(args.n_actions)
        if name == 'contextual_epsilon':
            return ContextualEpsilonGreedyPolicy(args.n_actions,args.context_dim,
                epsilon=epsilon,reward_type='linear_gaussian',explore_untried=False,seed=seed)
        return ContextualTSPolicy(args.n_actions,args.context_dim,
            reward_type='linear_gaussian',obs_sigma=args.obs_sigma,n_prob_mc=args.ts_prob_mc,seed=seed)
    return build



def _replication_task(task):
    args,beta,n,rep,truth=task
    records=[]; baseline_records=[]
    env = SubGaussianEnvironment(args.env,beta,args.scales,args.half_widths,
                                 None if args.mixtures_json is None else json.loads(args.mixtures_json))
    sampler = context_sampler(args.context_dim,args.context_var)
    behavior = policy_builder(args,args.pi0,args.epsilon0)
    target = policy_builder(args,args.pi1,args.epsilon1)
    seed = int(np.random.SeedSequence([args.seed,n,rep]).generate_state(1)[0])
    offline,callback = collect_subgaussian_data(env,behavior,sampler,n,seed,
        retain_policy_states=args.include_baselines and args.pi1 == 'uniform')
    if args.include_baselines:
        model = ScaledBernoulliBaselineModel(env.scales) if args.env == 'scaled_bernoulli' else PerActionLinearRewardModel()
        base = compute_all_contextual_intervals(
            offline['contexts'],offline['actions'],offline['rewards'],offline['behavior_probs'],
            target(seed+1),1-args.alpha,reward_model=model,include_elfcb=args.include_elfcb,
            dr_ci_method=args.dr_ci_method,dr_bootstrap_reps=args.dr_bootstrap_reps,
            dr_bootstrap_seed=seed+2,include_cadr=args.pi1 == 'uniform',
            cadr_current_behavior_probs=callback,cadr_min_samples=args.cadr_min_samples,
            cadr_variance_floor=args.cadr_variance_floor,cadr_warmup_sigma=args.cadr_warmup_sigma)
        baseline_records.append(dict(n=n,rep=rep,intervals=vars(base),
            target_interpretation='fixed contextual value' if args.pi1 == 'uniform' else 'one-step logged-history comparison; not fresh adaptive deployment',
            cadr_status='included' if args.pi1 == 'uniform' else 'omitted: adaptive evaluation target unsupported'))
    for method in args.variance_methods:
        model = ContextualSubGaussianWorkingModel(args.n_actions,args.context_dim,args.env,method,
            scales=env.scales,support_widths=None if args.env == 'gaussian_mixture' else env.support_widths(),
            adaptive_behavior=args.pi0 != 'uniform',propagate_variance_uncertainty=args.propagate_variance_uncertainty,
            variance_floor=args.variance_floor,proxy_alpha=args.proxy_alpha,proxy_grid_size=args.proxy_grid_size)
        for T in args.T_values:
            svi = ContextualParametricSVI(model,target,sampler,T,algo_seed=seed+100,context_seed=seed+200)
            M=args.inner_reps; m_info=None
            if args.select_M:
                lam,cov=model.fit(offline['contexts'],offline['actions'],offline['rewards'],offline['behavior_probs'])
                pilot=simulate_svi_summary(model,target,sampler,T,args.M_pilot,lam,seed+300,seed+400,backend=args.backend,block_size=args.rollout_block_size)
                grads=per_trajectory_gradients(model,pilot,lam)
                m_info=estimate_m_from_pilot(grads,cov,B=args.M_bootstrap_reps,tau=args.M_tau,rel_eps=args.M_rel_eps,seed=seed+500)
                M=max(2,min(int(m_info['m_star']),args.Mmax))
                m_info.update(selected=M,capped=int(m_info['m_star'])>args.Mmax)
            result=svi.run(offline,[args.alpha],M,backend=args.backend,rollout_block_size=args.rollout_block_size,progress=args.progress and args.n_jobs==1)
            wald=[result.center-result.ci_width[args.alpha],result.center+result.ci_width[args.alpha]]
            proj=[result.center-result.proj_ci_width[args.alpha],result.center+result.proj_ci_width[args.alpha]]
            primary=wald if args.pi0 == 'uniform' else proj
            for correction in args.regret_methods:
                if correction == 'gaussian_mixture_search':
                    reg=search_mixture_regret(model,target,sampler,T,args.candidate_count,args.mixture_components,
                        args.screen_rollouts,args.refine_rollouts,args.n_refine,args.regret_benchmark,
                        args.benchmark_epsilon,seed+600,args.mc_error_probability,backend=args.backend,block_size=args.rollout_block_size,progress=args.progress and args.n_jobs==1)
                else:
                    proxy=max(env.scales**2/4) if args.env == 'scaled_bernoulli' else max(model.variances)
                    reg=minimax_type_regret(T,args.n_actions,model.p,proxy,args.bound_constant,args.bound_formula,args.bound_log)
                corrected=expand_interval(primary,reg['B_T'],T)
                records.append(dict(n=n,rep=rep,T=T,variance_method=method,regret_method=correction,
                    center=result.center,center_mc_se=result.center_se,lambda_hat=result.lambda_hat,Sigma=result.Sigma,
                    gradient=result.gradient,simulation_backend=result.pi1_img.get('backend','python'),dispersion=model.variances if model.variances is not None else dict(type='conditional Bernoulli',beta=model.frozen_beta,scales=model.scales),
                    variance_uncertainty_propagated=model.propagate,primary_type='wald' if args.pi0 == 'uniform' else 'projection',
                    wald=wald,projection=proj,base=primary,corrected=corrected,regret=reg,inner_reps=M,selected_M=m_info,
                    bias=result.center-truth[T]['value'],covered=corrected[0]<=truth[T]['value']<=corrected[1],
                    base_covered=primary[0]<=truth[T]['value']<=primary[1],width=corrected[1]-corrected[0]))
    return records, baseline_records


def _safe_replication_task(task):
    try:
        records,baselines=_replication_task(task)
        return dict(records=records,baselines=baselines,error=None)
    except Exception as exc:
        return dict(records=[],baselines=[],error=dict(n=task[2],rep=task[3],
                    exception=type(exc).__name__,message=str(exc)))


def _truth_task(task):
    args,beta,T=task
    env = SubGaussianEnvironment(args.env,beta,args.scales,args.half_widths,
                                 None if args.mixtures_json is None else json.loads(args.mixtures_json))
    sampler = context_sampler(args.context_dim,args.context_var)
    behavior = policy_builder(args,args.pi0,args.epsilon0)
    target = policy_builder(args,args.pi1,args.epsilon1)
    return T,regret_rollouts(env,target,sampler,T,args.truth_reps,seed=args.seed+900000+T,
                            backend=args.backend,block_size=args.rollout_block_size,
                            progress=args.progress and args.n_jobs==1)


def _worker_init():
    # Avoid nested BLAS threads in each process (also documented shell exports).
    from threadpoolctl import threadpool_limits
    global _thread_limit
    _thread_limit=threadpool_limits(limits=1)


def _map_tasks(function,tasks,args,description):
    if args.n_jobs==1:
        for task in tqdm(tasks,desc=description,disable=not args.progress,unit='task'):
            yield function(task)
    else:
        with ProcessPoolExecutor(max_workers=min(args.n_jobs,len(tasks)),
                                 mp_context=mp.get_context('spawn'),initializer=_worker_init) as pool:
            futures=[pool.submit(function,task) for task in tasks]
            for future in tqdm(as_completed(futures),total=len(futures),desc=description,
                               disable=not args.progress,unit='task'):
                yield future.result()

def run(args):
    if args.n_actions < 1 or args.context_dim < 0 or not 0 < args.alpha < 1:
        raise ValueError('Invalid dimensions or alpha')
    if min(args.T_values+args.T_offline_values+[args.offline_reps]) < 1 or min(args.inner_reps,args.truth_reps) < 2:
        raise ValueError('Positive sample sizes and >=2 simulation replicates required')
    if args.env == 'gaussian_mixture' and 'hoeffding' in args.variance_methods:
        raise ValueError('Hoeffding is unavailable for Gaussian mixtures')
    if args.select_M and min(args.M_pilot,args.M_bootstrap_reps,args.Mmax) < 2:
        raise ValueError('Pilot, bootstrap and M budget must each be >=2')
    if args.propagate_variance_uncertainty and 'variance_proxy' in args.variance_methods:
        raise ValueError('Proxy uncertainty is unsupported; select empirical or Hoeffding')
    beta = np.zeros((args.n_actions,args.context_dim+1))
    beta[:,0] = np.linspace(-.5,.5,args.n_actions)
    if args.context_dim: beta[:,1:] = np.linspace(-.3,.3,args.n_actions)[:,None]
    if args.beta is not None: beta = np.asarray(args.beta).reshape(beta.shape)
    env = SubGaussianEnvironment(args.env,beta,args.scales,args.half_widths,
                                 None if args.mixtures_json is None else json.loads(args.mixtures_json))
    sampler = context_sampler(args.context_dim,args.context_var)
    behavior = policy_builder(args,args.pi0,args.epsilon0)
    target = policy_builder(args,args.pi1,args.epsilon1)
    if args.n_jobs == -1:
        args.n_jobs=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    if args.n_jobs < 1 or args.rollout_block_size < 1:
        raise ValueError('n_jobs must be positive or -1; rollout block size must be positive')
    if args.backend == 'numba':
        from .accelerated import NUMBA_AVAILABLE
        if not NUMBA_AVAILABLE: raise ImportError('Install numba to select the numba backend')
    truth=dict(_map_tasks(_truth_task,[(args,beta,T) for T in args.T_values],args,'True policy values'))
    records=[];baseline_records=[];failures=[]
    args.save_path.parent.mkdir(parents=True,exist_ok=True)
    checkpoint=args.save_path.with_suffix('.replications.jsonl')
    tasks=[(args,beta,n,rep,truth) for n in args.T_offline_values for rep in range(args.offline_reps)]
    with checkpoint.open('w') as log:
        log.write(json.dumps(clean_json(to_serializable(dict(config=vars(args),truth=truth))))+'\n')
        log.flush()
        for outcome in _map_tasks(_safe_replication_task,tasks,args,'Offline datasets'):
            log.write(json.dumps(clean_json(to_serializable(outcome)),allow_nan=False)+'\n');log.flush()
            if outcome['error'] is not None:
                failures.append(outcome['error'])
                if args.on_error=='raise':
                    raise RuntimeError(f"Replication failed: {outcome['error']}. Completed results: {checkpoint}")
            else:
                records.extend(outcome['records']);baseline_records.extend(outcome['baselines'])
    records.sort(key=lambda r:(r['n'],r['rep'],r['T'],r['variance_method'],r['regret_method']))
    baseline_records.sort(key=lambda r:(r['n'],r['rep']))
    failures.sort(key=lambda r:(r['n'],r['rep']))
    summaries=[]
    for n in args.T_offline_values:
        for T in args.T_values:
            for method in args.variance_methods:
                for correction in args.regret_methods:
                    group=[r for r in records if (r['n'],r['T'],r['variance_method'],r['regret_method'])==(n,T,method,correction)]
                    coverage=float(np.mean([r['covered'] for r in group])) if group else float('nan')
                    summaries.append(dict(n=n,T=T,variance_method=method,regret_method=correction,coverage=coverage,
                        coverage_mc_se=float(np.sqrt(coverage*(1-coverage)/len(group))) if group else float('nan'),
                        successful_reps=len(group),failed_reps=args.offline_reps-len(group),
                        coverage_all_requested=float(sum(r['covered'] for r in group)/args.offline_reps),
                        base_coverage=float(np.mean([r['base_covered'] for r in group])) if group else float('nan'),
                        mean_width=float(np.mean([r['width'] for r in group])) if group else float('nan'),mean_bias=float(np.mean([r['bias'] for r in group])) if group else float('nan')))
    payload=dict(config=vars(args),true_beta=beta,truth=truth,records=records,summary=summaries,baselines=baseline_records,failures=failures,checkpoint=str(checkpoint),
        baseline_summary=baseline_summary(baseline_records,truth),
        method='SVI',qualification='Empirical corrections with plug-in parameters; no certified worst-case coverage guarantee',
        context_distribution='known Gaussian',policy_model='linear working learner on every reward environment')
    args.save_path.parent.mkdir(parents=True,exist_ok=True)
    args.save_path.write_text(json.dumps(clean_json(to_serializable(payload)),indent=2,allow_nan=False))
    import csv
    with args.save_path.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(clean_json(summaries))
    return payload


def clean_json(value):
    """Represent unavailable intervals as JSON null rather than nonstandard NaN."""
    if isinstance(value, dict): return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def baseline_summary(records, truth):
    output=[]
    for n in sorted({r['n'] for r in records}):
        group=[r for r in records if r['n']==n]
        for T, target in truth.items():
            for name in ('ipw','dr','cadr','elfcb'):
                intervals=[r['intervals'][name] for r in group]
                valid=[ci for ci in intervals if ci is not None and np.isfinite(ci).all()]
                if not valid: continue
                output.append(dict(n=n,T=T,method=name,valid_reps=len(valid),
                    missing_reps=len(group)-len(valid),
                    coverage=float(np.mean([lo<=target['value']<=hi for lo,hi in valid])),
                    mean_width=float(np.mean([hi-lo for lo,hi in valid])),
                    target_interpretation=group[0]['target_interpretation']))
    return output


def main(): run(build_parser().parse_args())
if __name__ == '__main__': main()
