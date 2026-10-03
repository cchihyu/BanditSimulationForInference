# Contextual SVI

Simulation-based value inference (SVI) with contextual policies. Existing filenames,
including `contextual_bsi.py`, are unchanged. Public SVI classes are
`ContextualParametricSVI` and `ContextualSVIResult`; former BSI class/gradient names
remain aliases for existing callers. The legacy root MAB implementation is unchanged.

## Sub-Gaussian experiments

Run from the repository root, in the Python environment specified by `environment.yml`:

```bash
python -m contextual.run_subgaussian --help
python -m contextual.run_subgaussian \
  --env uniform --n_actions 3 --context_dim 2 \
  --variance_methods hoeffding variance_proxy empirical \
  --regret_methods gaussian_mixture_search minimax_bound \
  --T_values 25 50 --T_offline_values 300 \
  --offline_reps 10 --inner_reps 500 --truth_reps 2000 \
  --candidate_count 20 --screen_rollouts 50 --refine_rollouts 200 \
  --save_path results/uniform_svi.json
```

Use `--context_dim 0` for intercept-only MAB. `--beta` supplies flattened action-major
coefficients (K*(d+1) numbers). Without it, deterministic coefficients are generated
for any positive number of arms and nonnegative context dimension. Contexts are
independent Gaussian draws with known `--context_var`; context-distribution estimation
is not included.

### True environments

* `scaled_bernoulli`: `scale[a] * Bernoulli(expit(phi(x,a) @ beta))`.
  Configure `--scales` with one shared or K arm-specific positive values.
* `uniform`: linear conditional mean plus Uniform(-h[a],h[a]) noise;
  configure `--half_widths`. Noise width does not vary with context.
* `gaussian_mixture`: linear conditional mean plus centered finite Gaussian-mixture
  noise. `--mixtures_json` is a JSON list of K objects containing `weights`, `means`,
  and `sigmas`. Component offsets are centered automatically; shape parameters do
  not depend on context. For two arms, for example:
  `--mixtures_json '[{"weights":[0.7,0.3],"means":[-1,2],"sigmas":[0.3,0.8]},{"weights":[1],"means":[0],"sigmas":[1]}]'`.

### Dispersion and uncertainty

`--variance_methods` accepts several methods in the same run, sharing offline data:

* `hoeffding`: known support width squared / 4. Invalid for Gaussian mixtures.
  Uniform widths refer to additive noise, not a global reward bound with Gaussian contexts.
* `empirical`: weighted residual second moments per arm for additive noise;
  scale squared times fitted p(x)(1-p(x)) for Bernoulli (family-based conditional variance).
* `variance_proxy`: centered-residual empirical log-MGF grid for additive noise;
  the Bernoulli-specific optimal proxy evaluated at fitted p(x) for Bernoulli.
  The latter is model-based, not pooled empirical-MGF estimation.
  Control `--proxy_alpha` (default .49), `--proxy_grid_size` (odd; default 4001),
  and `--variance_floor` (default 1e-8).

All methods freeze fitted dispersion in the SVI gradient by default. With
`--propagate_variance_uncertainty`, empirical additive dispersion adds K variance
coordinates and joint sandwich covariance. Bernoulli empirical dispersion instead
adds its chain-rule derivative through the same probability coefficients. Known
Hoeffding bounds need no extra uncertainty; estimated-proxy uncertainty is unsupported
and raises an error. Adaptive fitting keeps the repository's uniform-reference
inverse-propensity weights. This code does not establish a new adaptive CLT.

Full-rank offline feature design and observations in each arm are required. Near
logistic separation raises an error instead of returning misleading intervals.

### Policies

Choose `--pi0` and `--pi1` from `uniform`, `contextual_epsilon`, `contextual_ts`.
The new runner uses linear working learners for epsilon-greedy and TS in **all**
true environments, so the same algorithm can consume real-valued Gaussian surrogate
rewards. A logistic conditional-mean fit for Bernoulli does not imply a binary-only
policy update. Binary logistic policies remain available in the original runner.

`--epsilon0`, `--epsilon1`, `--obs_sigma`, and `--ts_prob_mc` control policies.
Epsilon-greedy forced first-arm exploration is disabled to retain positive logging
probabilities. The TS observation scale remains fixed across candidate environments.

### Regret corrections

`--regret_methods` supports both methods in one run:

* `gaussian_mixture_search`: includes the fitted Gaussian environment, screens
  `--candidate_count` candidates, and independently refines `--n_refine` of them.
  `--mixture_components`, `--screen_rollouts`, `--refine_rollouts`, and
  `--mc_error_probability` configure the search. Candidates are analytically scaled
  using max component variance + component-mean range squared / 4, a sufficient
  proxy bound. They lie within the fitted dispersion budget; this is a restricted
  library, not a search over all sub-Gaussian distributions. Normal MC adjustments
  are approximate and do not account for omitted candidates or parameter error.
* `minimax_bound`: explicitly labeled **minimax-inspired sensitivity formula**, not
  a proven bound for the implemented policy. With proxy budget v and dimension p,
  `linear_dimension` uses C*p*sqrt(v*T*log(T)); `mab_rate` uses
  C*sqrt(K*v*T*log(T)). Configure `--bound_formula`, `--bound_constant`, and
  `--no-bound_log`. Gaussian contexts are not silently truncated to satisfy a theorem.

`--regret_benchmark unrestricted` uses the best conditional mean. `epsilon_floor`
uses the best policy constrained to put at least `--benchmark_epsilon / K` on each
arm. Probability-floor violations raise errors. The benchmark changes mixture
regret; the minimax-inspired formulas always use unrestricted regret.

Every candidate keeps the fitted means, context law, and policy settings fixed.
Regret uses policy-probability-weighted conditional-mean gaps; noisy rewards still
update the learner. The largest refined MC-adjusted regret expands each endpoint by
B_T/T. An empirical variance used as a proxy budget is an additional approximation.

### Results and baselines

JSON contains configuration, independent true-value MC estimates and SEs, per-run
parameters/covariances/gradients, fitted dispersion, base and corrected intervals,
regret diagnostics, and summaries. A companion CSV contains SVI coverage, width,
and bias summaries. Primary SVI intervals are Wald for uniform logging and projection
for adaptive logging. Reported coverage treats the independently simulated truth as
its reference; inspect its SE. SVI center MC error is recorded, not added to the CI.
Finite-search and plug-in corrections do not guarantee nominal coverage.

`--select_M` enables existing gradient-pilot replication selection, with controls
`--M_pilot`, `--M_bootstrap_reps`, `--Mmax`, `--M_tau`, `--M_rel_eps`. It monitors
width stability, not center MC error. Capped selections are recorded.

`--include_baselines` uses bootstrap DR by default (`--dr_ci_method wald` retains
Wald). Set `--dr_bootstrap_reps`; bootstrap seeds derive reproducibly from the run
seed. Scaled Bernoulli DR fits logistic probabilities on rewards divided by scale.
DR/IPW for adaptive evaluation policies are labeled logged-history comparisons,
not fresh-deployment estimators. CADR is included for the fixed uniform target and
omitted for adaptive evaluation targets. Pre-observation logging-policy snapshots
supply current-to-past propensity transport; `--ts_prob_mc` also controls these
probability estimates. CADR warm-up and floors have dedicated flags. ELFCB is opt-in
and still uses observed extrema, not certified population support bounds. Missing
intervals are JSON null; bootstrap fit failures raise rather than being discarded.
CADR snapshots and transport can require O(n^2) time/storage for growing policies.

## Existing runner and validation

The original parametric runner remains available as `python -m contextual.run_contextual`.
The new model follows its `fit`, `mean`, `sample`, and `score` interface, including
optional custom feature maps through the Python API. The sub-Gaussian runner now selects compiled simulation for supported built-in
environments and policies. Custom feature maps and models retain the Python path.
Legacy Gaussian simulation paths retain their existing behavior.

```bash
python -m unittest discover -s tests -p 'test_contextual_subgaussian.py' -v
```


## Progress and parallel simulation

The sub-Gaussian runner accepts `--n_jobs 8 --backend numba
--rollout_block_size 128 --progress`. Choose a worker count within your cluster CPU
allocation. `--n_jobs -1` uses CPUs visible to the process; it may exceed a scheduler
allocation without CPU affinity restrictions. Default worker count is one.

Progress bars are enabled by default (`--no-progress` disables them). Multiple
workers parallelize independent offline datasets and truth trajectories in batches;
worker BLAS threads are limited to one to avoid oversubscription. Install the
updated environment or run `python -m pip install numba tqdm threadpoolctl`.

`--backend auto` (default) uses Numba when available and supported; `--backend
python` selects the reference implementation. Explicit `--backend numba` raises
if unavailable or unsupported. Compiled kernels cover reward generation, policy
updates, rollout gradients, and regret simulation for built-in uniform,
contextual epsilon-greedy, and Gaussian Thompson policies. Model fitting and
baseline bootstrap calculations remain Python. Compilation adds startup time.

Rollouts run in blocks to avoid retaining all trajectory histories. Thompson
sampling computes scalar Gaussian predictions instead of sampling full coefficient
vectors, preserving the prediction law and the requested `--ts_prob_mc` budget.
For a fixed backend and seed, results are reproducible across worker counts and
block sizes; Python and Numba use different random streams.

Each completed dataset is flushed to a companion `.replications.jsonl` checkpoint.
This is a record of completed work, not automatic resume support; a new run with
the same output path overwrites it. The default `--on_error raise` stops on a failed
fit. Optional `--on_error continue` records failures and proceeds to other datasets.
This does not fix logistic separation. Summaries report successful and failed counts;
`coverage` uses successful datasets, while `coverage_all_requested` counts failed
datasets as not covered. Neither should hide the failure rate.

Validate both the statistical extensions and performance paths with:

```bash
python -m unittest discover -s tests -p 'test_contextual_*.py' -v
```

Truth simulation uses `--truth_batch_size 1000` by default. With five horizons and
50,000 truth trajectories each, 250 tasks share the same worker pool, so
`--n_jobs 64` can use 64 workers in this stage too. The progress bar counts completed
batches. A final shorter batch is included. Trajectory seeds and aggregate means
and standard errors are unchanged by batch size or worker count. There are no
nested process pools. Truth computations still restart on a new invocation.
