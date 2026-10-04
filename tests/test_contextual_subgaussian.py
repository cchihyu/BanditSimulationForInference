import unittest
import tempfile
from pathlib import Path
import numpy as np
from scipy.stats import norm
from contextual.dispersion import bernoulli_proxy, residual_proxy
from contextual.subgaussian_environments import SubGaussianEnvironment
from contextual.environments import ContextualSubGaussianWorkingModel, ContextualLinearGaussianRewardModel
from contextual.simulation import context_sampler, collect_subgaussian_data, UniformContextualPolicy, ContextualEpsilonGreedyPolicy, ContextualTSPolicy
from contextual.regret_corrections import regret_rollouts, search_mixture_regret, expand_interval, standardized_mixture
from contextual.contextual_bsi import ContextualParametricSVI, ContextualParametricBSI, contextual_bandit_exp_runner, estimate_contextual_svi_gradient
from contextual.select_inner_reps import per_trajectory_gradients
from contextual.run_subgaussian import build_parser,run
from contextual.baselines import cadr_interval, contextual_cadr_sigmas, contextual_cadr_sigmas_static, contextual_dr_bootstrap_interval, PerActionLinearRewardModel, compute_all_intervals, compute_all_bandit_intervals


class SubGaussianTests(unittest.TestCase):
    def logs(self,kind='uniform',d=1,adaptive=False):
        beta=np.zeros((2,d+1));beta[:,0]=[-.3,.4]
        if d: beta[:,1]=[.2,-.1]
        env=SubGaussianEnvironment(kind,beta,scales=[2,3])
        builder=lambda seed:UniformContextualPolicy(2)
        logs,_=collect_subgaussian_data(env,builder,context_sampler(d),600,37)
        return env,logs

    def fit(self,kind='uniform',method='empirical',joint=False,d=1,adaptive=False):
        env,data=self.logs(kind,d)
        model=ContextualSubGaussianWorkingModel(2,d,kind,method,scales=env.scales,
            support_widths=None if kind=='gaussian_mixture' else env.support_widths(),
            propagate_variance_uncertainty=joint,adaptive_behavior=adaptive,proxy_grid_size=101)
        model.fit(data['contexts'],data['actions'],data['rewards'],data['behavior_probs'])
        return model,data

    def test_dispersion(self):
        np.testing.assert_allclose(bernoulli_proxy(np.array([0,.5,1])),[0,.25,0])
        np.testing.assert_allclose(bernoulli_proxy(.2),bernoulli_proxy(.8))
        z=np.array([-1.,1.]*50)
        self.assertAlmostEqual(residual_proxy(z,grid_size=101),1.,places=7)
        self.assertAlmostEqual(residual_proxy(3*z,grid_size=101),9.,places=7)

    def test_true_means_variances(self):
        rng=np.random.default_rng(84)
        for kind in ['uniform','gaussian_mixture','scaled_bernoulli']:
            env=SubGaussianEnvironment(kind,np.array([[.2]]),scales=2.)
            ys=np.array([env.sample(np.empty(0),0,rng) for _ in range(15000)])
            self.assertLess(abs(ys.mean()-env.mean([],0)),.035)
            if kind=='uniform': self.assertLess(abs(ys.var()-1/3),.025)
            if kind=='scaled_bernoulli':
                p=env.mean([],0)/2;self.assertLess(abs(ys.var()-4*p*(1-p)),.025)
        env=SubGaussianEnvironment('beta',np.array([[.35]]),beta_alphas=[.35],beta_betas=[.65])
        ys=np.array([env.sample(np.empty(0),0,rng) for _ in range(15000)])
        self.assertLess(abs(ys.mean()-.35),.015)
        self.assertAlmostEqual(env.support_widths()[0],1.)

    def test_scores_finite_difference(self):
        for kind in ['uniform','gaussian_mixture','scaled_bernoulli']:
            for joint in [False,True]:
                model,_=self.fit(kind,joint=joint)
                theta=model.lambda_hat_.copy();x=np.array([.4]);a=1;r=.7
                def logdensity(t):
                    v=model.variance(x,a,t);e=r-model.mean(x,a,t)
                    return -.5*np.log(v)-e*e/(2*v)
                numerical=[]
                for j in range(len(theta)):
                    h=np.zeros_like(theta);h[j]=1e-5
                    numerical.append((logdensity(theta+h)-logdensity(theta-h))/2e-5)
                np.testing.assert_allclose(model.score(x,a,r,theta),numerical,atol=1e-6)
                self.assertGreaterEqual(np.linalg.eigvalsh(model.Sigma_).min(),-1e-9)
                self.assertEqual(len(theta),4+(2 if joint and kind!='scaled_bernoulli' else 0))

    def test_mab_covariance(self):
        model,data=self.fit(d=0,joint=True)
        a=data['actions'];r=data['rewards'];theta=model.lambda_hat_
        for arm in range(2):
            self.assertAlmostEqual(theta[arm],r[a==arm].mean())
            self.assertAlmostEqual(theta[2+arm],r[a==arm].var())
        n=len(r);h=[]
        for ai,ri in zip(a,r):
            row=np.zeros(4);pr=np.mean(a==ai);e=ri-theta[ai]
            row[ai]=e/pr;row[2+ai]=(e*e-theta[2+ai])/pr;h.append(row)
        h=np.array(h);np.testing.assert_allclose(model.Sigma_,h.T@h/n,atol=1e-10)

    def test_uniform_adaptive_covariance(self):
        m,_=self.fit(joint=True);other,_=self.fit(joint=True,adaptive=True)
        np.testing.assert_allclose(m.Sigma_,other.Sigma_)

    def test_invalid_methods(self):
        with self.assertRaises(ValueError):self.fit('gaussian_mixture','hoeffding')
        with self.assertRaises(ValueError):self.fit('uniform','variance_proxy',True)

    def test_gradients_and_alias(self):
        self.assertIs(ContextualParametricSVI,ContextualParametricBSI)
        m,data=self.fit(joint=True,d=0)
        sim=contextual_bandit_exp_runner(m,lambda seed:UniformContextualPolicy(2),context_sampler(0),4,5,m.lambda_hat_)
        np.testing.assert_allclose(per_trajectory_gradients(m,sim,m.lambda_hat_).mean(axis=0),
                                  estimate_contextual_svi_gradient(m,sim,m.lambda_hat_))

    def test_benchmarks(self):
        env=SubGaussianEnvironment('uniform',np.array([[0.],[2.]]))
        builder=lambda seed:UniformContextualPolicy(2)
        result=regret_rollouts(env,builder,context_sampler(0),7,4)
        self.assertAlmostEqual(result['regret'],7.)
        result=regret_rollouts(env,builder,context_sampler(0),7,4,'epsilon_floor',1.)
        self.assertAlmostEqual(result['regret'],0.)
        np.testing.assert_allclose(expand_interval([1,2],7,7),[0,3])

    def test_mixture_bound(self):
        rng=np.random.default_rng(6)
        for _ in range(10):
            w,m,s=standardized_mixture(rng,4)
            self.assertAlmostEqual(w@m,0.)
            self.assertLessEqual(max(s*s)+np.ptp(m)**2/4,1+1e-12)

    def test_cadr_and_bootstrap(self):
        y=np.array([1.,2.,3.,4.]);weights=np.ones(4);x=np.zeros((4,0));a=np.zeros(4,dtype=int);p=np.ones((4,1))
        ci=cadr_interval(y,weights,.95,conditional_sigmas=np.full(4,2.))
        np.testing.assert_allclose(ci,[y.mean()-norm.ppf(.975),y.mean()+norm.ppf(.975)])
        sig=contextual_cadr_sigmas(x,a,y,p,p,lambda t,x:np.ones((t,1)),min_samples=2)
        np.testing.assert_allclose(sig[2:],[np.std(y[:2]),np.std(y[:3])])
        static=contextual_cadr_sigmas_static(a,y,p,p,min_samples=2)
        np.testing.assert_allclose(static,sig)
        class Counter(PerActionLinearRewardModel):
            count=0
            def fit(self,*args,**kwargs):
                type(self).count+=1
                return super().fit(*args,**kwargs)
        ci=contextual_dr_bootstrap_interval(x,a,y,weights,p,.95,Counter(),20,7)
        self.assertEqual(Counter.count,21)
        rng=np.random.default_rng(7);samples=[y[rng.integers(4,size=4)].mean() for _ in range(20)]
        width=norm.ppf(.975)*np.std(samples)
        np.testing.assert_allclose(ci,[y.mean()-width,y.mean()+width])

    def test_mab_and_zero_context_use_identical_baselines(self):
        env=SubGaussianEnvironment('uniform',np.array([[.1],[-.2]]))
        behavior=lambda seed:ContextualEpsilonGreedyPolicy(
            2,0,epsilon=.2,reward_type='linear_gaussian',
            explore_untried=False,seed=seed)
        data,current=collect_subgaussian_data(
            env,behavior,context_sampler(0),80,37,retain_policy_states=True)
        target=lambda seed:ContextualTSPolicy(2,0,n_prob_mc=20,seed=seed)
        options=dict(
            reward_model=PerActionLinearRewardModel(),include_elfcb=False,
            dr_bootstrap_reps=20,dr_bootstrap_seed=91,
            cadr_current_behavior_probs=current,cadr_min_samples=5,
            cadr_target_is_fixed=False,allow_adaptive_cadr=True,
            cadr_behavior_is_static=False)
        for mode in ('one_step','cumulative'):
            contextual=compute_all_intervals(
                data['contexts'],data['actions'],data['rewards'],
                data['behavior_probs'],target(73),.9,weight_mode=mode,**options)
            bandit=compute_all_bandit_intervals(
                data['actions'],data['rewards'],data['behavior_probs'],
                target(73),.9,weight_mode=mode,**options)
            for name in ('ipw','dr','cadr'):
                np.testing.assert_allclose(getattr(contextual,name),getattr(bandit,name))

    def test_runner_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            for env in ['uniform','scaled_bernoulli','gaussian_mixture']:
                methods=['empirical','variance_proxy']+([] if env=='gaussian_mixture' else ['hoeffding'])
                args=build_parser().parse_args(['--env',env,'--n_actions','2','--context_dim','0',
                    '--T_values','2','--T_offline_values','100','--offline_reps','1','--inner_reps','4',
                    '--truth_reps','4','--candidate_count','2','--screen_rollouts','3','--refine_rollouts','3',
                    '--proxy_grid_size','101','--variance_methods',*methods,'--regret_methods','gaussian_mixture_search','minimax_bound',
                    '--no-include_baselines','--save_path',str(Path(tmp)/f'{env}.json')])
                payload=run(args);self.assertEqual(len(payload['records']),len(methods)*2)
            args=build_parser().parse_args(['--env','beta','--beta_alphas','.35','.5','.5',
                '--beta_betas','.65','.5','.5','--n_actions','3','--context_dim','0',
                '--T_values','2','--T_offline_values','100','--offline_reps','1','--inner_reps','4',
                '--truth_reps','4','--variance_methods','hoeffding','empirical',
                '--regret_methods','minimax_bound','--no-include_baselines',
                '--save_path',str(Path(tmp)/'beta.json')])
            payload=run(args)
            record=payload['records'][0]
            for name in ('svi','projection_svi','corrected_svi','corrected_projection_svi'):
                self.assertIn(name,record)
            args=build_parser().parse_args(['--env','uniform','--n_actions','2','--context_dim','1',
                '--pi0','contextual_epsilon','--pi1','uniform','--propagate_variance_uncertainty',
                '--T_values','2','--T_offline_values','100','--offline_reps','1','--inner_reps','4','--truth_reps','4',
                '--regret_methods','minimax_bound','--include_baselines','--dr_bootstrap_reps','3',
                '--save_path',str(Path(tmp)/'adaptive.json')])
            payload=run(args)
            self.assertEqual(payload['records'][0]['primary_type'],'projection')
            self.assertTrue(np.isfinite(payload['baselines'][0]['intervals']['cadr']).all())
            args=build_parser().parse_args(['--env','uniform','--n_actions','2','--context_dim','1',
                '--pi0','uniform','--pi1','contextual_ts','--ts_prob_mc','10',
                '--T_values','2','--T_offline_values','100','--offline_reps','1','--inner_reps','4','--truth_reps','4',
                '--regret_methods','minimax_bound','--dr_bootstrap_reps','3','--no-include_elfcb',
                '--save_path',str(Path(tmp)/'adaptive_target.json')])
            payload=run(args)
            self.assertTrue(np.isfinite(payload['baselines'][0]['intervals']['cadr']).all())
            self.assertEqual(payload['baselines'][0]['target_interpretation'],'adaptive contextual target')


class CompatibilityTests(unittest.TestCase):
    def test_ts_optimized_observation_scale(self):
        from unittest.mock import patch
        import contextual.contextual_bsi as core
        from contextual.algorithms import ContextualTSPolicy
        model=ContextualLinearGaussianRewardModel(2,1,sigma=3.)
        builder=lambda seed:ContextualTSPolicy(2,1,obs_sigma=.5,n_prob_mc=20,seed=seed)
        captured=[];original=core._batched_linear_ts_action_probs_from_precision
        def capture(*args,**kwargs):
            captured.append(kwargs['precision'].copy())
            return original(*args,**kwargs)
        with patch.object(core,'_batched_linear_ts_action_probs_from_precision',capture):
            result=core.contextual_bandit_exp_runner(model,builder,context_sampler(1),2,1,np.zeros(4))
        action=result['all_actions'][0,0];x=np.r_[1.,result['all_contexts'][0,0]]
        expected=captured[0].copy();expected[0,action]+=np.outer(x,x)/.5**2
        np.testing.assert_allclose(captured[1],expected)

    def test_pilot_and_bernoulli_joint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=build_parser().parse_args(['--env','scaled_bernoulli','--n_actions','2','--context_dim','1',
                '--propagate_variance_uncertainty','--pi1','contextual_ts','--ts_prob_mc','10',
                '--T_values','2','--T_offline_values','150','--offline_reps','1','--truth_reps','3',
                '--select_M','--M_pilot','3','--M_bootstrap_reps','3','--Mmax','5',
                '--regret_methods','minimax_bound','--no-include_baselines','--save_path',str(Path(tmp)/'pilot.json')])
            payload=run(args);self.assertTrue(payload['records'][0]['variance_uncertainty_propagated'])
            self.assertLessEqual(payload['records'][0]['inner_reps'],5)

if __name__=='__main__':unittest.main()
