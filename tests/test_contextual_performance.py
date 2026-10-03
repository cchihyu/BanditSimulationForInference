import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.stats import norm
from contextual.accelerated import NUMBA_AVAILABLE, simulate, posterior_update, linear_ts_probabilities, rollout_block
from contextual.algorithms import ContextualTSPolicy, ContextualEpsilonGreedyPolicy
from contextual.simulation import UniformContextualPolicy, context_sampler, collect_subgaussian_data
from contextual.subgaussian_environments import SubGaussianEnvironment
from contextual.environments import ContextualSubGaussianWorkingModel
from contextual.regret_corrections import regret_rollouts, MixtureCandidate, standardized_mixture
from contextual.run_subgaussian import build_parser,run


@unittest.skipUnless(NUMBA_AVAILABLE,'Numba not installed')
class CompiledTests(unittest.TestCase):
    def test_posterior_update_matches_existing_policy(self):
        policy=ContextualTSPolicy(2,2,obs_sigma=.7,prior_mean=.2,prior_var=1.3,n_prob_mc=10)
        mean=np.full(3,.2);cov=np.eye(3)*1.3;rng=np.random.default_rng(9)
        for _ in range(80):
            x=rng.normal(size=2);r=rng.normal()
            policy.update(x,0,r)
            mean,cov=posterior_update(mean,cov,np.r_[1.,x],r,.7**2)
            ref_cov=np.linalg.inv(policy._precision[0]);ref_mean=ref_cov@policy._info[0]
            np.testing.assert_allclose(mean,ref_mean,atol=1e-10)
            np.testing.assert_allclose(cov,ref_cov,atol=1e-10)

    def test_scalar_ts_probabilities(self):
        x=np.array([1.,.5]);means=np.array([[.2,.1],[-.3,.4]])
        cov=np.array([np.eye(2)*.6,np.eye(2)*.9])
        expected=norm.cdf(((means[0]-means[1])@x)/np.sqrt(x@(cov[0]+cov[1])@x))
        probs=linear_ts_probabilities(means,cov,x,100000,1e-8)
        self.assertLess(abs(probs[0]-expected),.009)
        self.assertTrue(linear_ts_probabilities.nopython_signatures)

    def test_block_invariance_and_all_environments(self):
        for kind in ['uniform','scaled_bernoulli','gaussian_mixture']:
            env=SubGaussianEnvironment(kind,np.array([[.1,.2],[-.2,.3]]))
            target=lambda seed:ContextualTSPolicy(2,1,n_prob_mc=20,seed=seed)
            a=simulate(env,target,context_sampler(1),8,9,17,backend='numba',block_size=2)
            b=simulate(env,target,context_sampler(1),8,9,17,backend='numba',block_size=5)
            for key in ['observed','expected','regrets']:
                np.testing.assert_array_equal(a[key],b[key])
        self.assertTrue(rollout_block.nopython_signatures)

    def test_reference_distribution(self):
        env=SubGaussianEnvironment('uniform',np.array([[.1],[-.2]]))
        for policy in ['ts','epsilon','uniform']:
            if policy=='ts':target=lambda seed:ContextualTSPolicy(2,0,n_prob_mc=20,seed=seed)
            elif policy=='epsilon':target=lambda seed:ContextualEpsilonGreedyPolicy(2,0,explore_untried=False,seed=seed)
            else:target=lambda seed:UniformContextualPolicy(2)
            a=regret_rollouts(env,target,context_sampler(0),8,500,seed=29,backend='numba')
            b=regret_rollouts(env,target,context_sampler(0),8,500,seed=29,backend='python')
            self.assertLessEqual(abs(a['value']-b['value']),6*np.hypot(a['value_se'],b['value_se'])+1e-10)

    def test_gradient_analytic_one_step(self):
        for kind in ['uniform','scaled_bernoulli']:
            env=SubGaussianEnvironment(kind,np.array([[.2],[-.3]]),scales=2.)
            target=lambda seed:UniformContextualPolicy(2)
            logs,_=collect_subgaussian_data(env,target,context_sampler(0),500,15)
            for joint in [False,True]:
                m=ContextualSubGaussianWorkingModel(2,0,kind,scales=2.,propagate_variance_uncertainty=joint)
                theta,_=m.fit(logs['contexts'],logs['actions'],logs['rewards'],logs['behavior_probs'])
                fast=simulate(m,target,context_sampler(0),1,12000,91,params=theta,gradient=True,backend='numba')
                gradients=fast['trajectory_gradients'];expected=np.zeros(len(theta))
                expected[:2]=.5
                if kind=='scaled_bernoulli':
                    from scipy.special import expit
                    p=expit(theta);expected[:2]=p*(1-p) # .5 * scale 2
                error=abs(gradients.mean(axis=0)-expected)
                self.assertTrue(np.all(error < 6*gradients.std(axis=0)/np.sqrt(len(gradients))+.005))
                # Mixture candidates exercise conditional scaling and packed arrays.
                c=MixtureCandidate(m,[standardized_mixture(np.random.default_rng(7+a),3) for a in range(2)])
                result=simulate(c,target,context_sampler(0),3,4,19,backend='numba')
                self.assertTrue(np.isfinite(result['observed']).all())

    def test_serial_parallel_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            def args(jobs):
                return build_parser().parse_args(['--env','uniform','--n_actions','2','--context_dim','1',
                    '--pi1','contextual_ts','--ts_prob_mc','10','--T_values','2','3',
                    '--T_offline_values','80','--offline_reps','3','--inner_reps','4','--truth_reps','4',
                    '--candidate_count','2','--screen_rollouts','3','--refine_rollouts','3',
                    '--regret_methods','gaussian_mixture_search','minimax_bound','--backend','numba',
                    '--truth_batch_size','1' if jobs==2 else '3','--n_jobs',str(jobs),'--no-progress','--save_path',str(Path(tmp)/f'j{jobs}.json')])
            one=run(args(1));two=run(args(2))
            self.assertEqual(one['truth'],two['truth'])
            self.assertEqual(one['summary'],two['summary'])
            for a,b in zip(one['records'],two['records']):
                np.testing.assert_array_equal(a['lambda_hat'],b['lambda_hat'])
                self.assertEqual(a['corrected'],b['corrected'])
                self.assertEqual(a['regret'],b['regret'])
            self.assertEqual(len(Path(two['checkpoint']).read_text().splitlines()),4)

    def test_truth_batches_match_unsplit_reference(self):
        env=SubGaussianEnvironment('uniform',np.array([[.1,.2],[-.2,.3]]))
        target=lambda seed:ContextualTSPolicy(2,1,n_prob_mc=10,seed=seed)
        for backend in ['python','numba']:
            full=regret_rollouts(env,target,context_sampler(1),4,7,seed=92,backend=backend)
            chunks=[regret_rollouts(env,target,context_sampler(1),4,min(3,7-start),
                       seed=92,backend=backend,trajectory_offset=start,return_samples=True)
                    for start in range(0,7,3)]
            for key,out in [('values','value'),('regrets','regret')]:
                values=np.concatenate([c[key] for c in chunks])
                self.assertEqual(float(values.mean()),full[out])
                self.assertEqual(float(values.std(ddof=1)/np.sqrt(7)),full[out+'_se'])

    def test_failure_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=build_parser().parse_args(['--env','scaled_bernoulli','--n_actions','2','--context_dim','0',
                '--beta','-100','-100','--T_values','2','--T_offline_values','30','--offline_reps','2',
                '--inner_reps','3','--truth_reps','3','--regret_methods','minimax_bound','--backend','numba',
                '--on_error','continue','--no-progress','--save_path',str(Path(tmp)/'fail.json')])
            result=run(args)
            self.assertEqual(len(result['failures']),2)
            self.assertEqual(result['summary'][0]['failed_reps'],2)
            self.assertEqual(json.loads(args.save_path.read_text())['summary'][0]['coverage'],None)

if __name__=='__main__':unittest.main()
