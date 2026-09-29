# BSI: Bandit Simulation for Average Reward Inference

Multi-arm bandit algorithms are increasingly used in online platforms, clinical trials, and social science experiments, but valid statistical inference on their performance remains an open challenge. After deploying bandits, a natural question is whether one can construct a confidence interval for its mean reward and assess whether it reliably outperforms a baseline policy. More broadly, one may wish to assess a new candidate algorithm's expected reward before committing to deployment. Standard inference methods break down because bandit algorithms introduce complex dependencies in the collected data, leading to inflated Type-I error. Moreover, existing inference methods for bandit data only apply to estimands such as the mean reward under a fixed action, which do not depend on the data-collection algorithm. We propose Bandit Simulation for Inference (BSI), a framework that fits a simulator of the bandit environment from observed data---either on-policy or off-policy---and uses it to estimate the mean reward under any evaluation policy, including adaptive blackbox algorithms. BSI formally propagates simulator estimation error into the confidence interval construction, requires only weak exploration assumptions on the behavior policy, and avoids importance weighting. We prove that BSI yields asymptotically valid confidence intervals, and demonstrate empirically that it maintains nominal coverage in settings where standard off-policy evaluation methods fail.

---

## Quick start

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate bsi
```

### 2. Run a small experiment

```bash
python run.py \
    --env normal \
    --mus 0.1 0.2 1.0 \
    --sigmas 1.0 1.0 1.0 \
    --pi0 epsilon_greedy \
    --pi1 ts_normal \
    --T 200 \
    --T_offline 500 \
    --offline_reps 200 \
    --alphas 0.05 0.10 \
    --n_jobs 4 \
    --save_dir results \
    --tag your_tag
```

---

## Reproducing Figure 2 and Figure 3 experiments

The following commands are written in the same style as the small experiment above. Each code box runs one panel/experiment family over the corresponding values of `T_offline`.

The full paper settings are computationally expensive: `T=500`, `offline_reps=1000`, `alphas=0.10` for 90% confidence intervals, and `infer_reps=20000` for BSI Monte Carlo rollouts. For a quick dry run, reduce `--offline_reps` and `--infer_reps` first.

Create the output folders once before running the commands:

```bash
mkdir -p results/figure2 results/figure3 figures
```

### Figure 2, left panel: Gaussian rewards, `pi0 = epsilon_greedy`, `pi1 = epsilon_greedy`

This reproduces the known-parametric Gaussian setting with arm means `(0.1, 0.2, 0.2)`, standard deviations `(1, 1, 1)`, and unknown reward variance.

```bash
TOFFS="100 150 250 500 1000 3000"

for TOFF in $TOFFS; do
  python run.py \
      --env normal \
      --mus 0.1 0.2 0.2 \
      --sigmas 1.0 1.0 1.0 \
      --estimate_sigma \
      --pi0 epsilon_greedy \
      --pi1 epsilon_greedy \
      --T 500 \
      --T_offline $TOFF \
      --offline_reps 1000 \
      --alphas 0.10 \
      --infer_reps 20000 \
      --no-select_M \
      --n_policy_value_mc 20000 \
      --n_eval_prob_mc 2000 \
      --weight_modes one_step \
      --run_naive_t_test \
      --no-save_logged_data \
      --n_jobs 8 \
      --save_dir results/figure2 \
      --tag fig2_gaussian_eps_eps
done
```

### Figure 2, middle panel: Bernoulli rewards, `pi0 = ts_bernoulli`, `pi1 = ts_bernoulli`

This reproduces the known-parametric Bernoulli setting with arm means `(0.35, 0.5, 0.6)` and Thompson Sampling for both the logging and evaluation policies.

```bash
TOFFS="100 150 250 500 1000 3000"

for TOFF in $TOFFS; do
  python run.py \
      --env bernoulli \
      --mus 0.35 0.5 0.6 \
      --pi0 ts_bernoulli \
      --pi1 ts_bernoulli \
      --T 500 \
      --T_offline $TOFF \
      --offline_reps 1000 \
      --alphas 0.10 \
      --infer_reps 20000 \
      --no-select_M \
      --n_policy_value_mc 20000 \
      --n_eval_prob_mc 2000 \
      --weight_modes one_step \
      --run_naive_t_test \
      --no-save_logged_data \
      --n_jobs 8 \
      --save_dir results/figure2 \
      --tag fig2_bernoulli_ts_ts
done
```

### Figure 2, right panel: Bernoulli rewards, `pi0 = uniform`, `pi1 = ts_bernoulli`

This reproduces the known-parametric Bernoulli setting with uniform logging and Thompson Sampling as the evaluation policy.

```bash
TOFFS="100 150 250 500 1000 3000"

for TOFF in $TOFFS; do
  python run.py \
      --env bernoulli \
      --mus 0.35 0.5 0.6 \
      --pi0 uniform \
      --pi1 ts_bernoulli \
      --T 500 \
      --T_offline $TOFF \
      --offline_reps 1000 \
      --alphas 0.10 \
      --infer_reps 20000 \
      --no-select_M \
      --n_policy_value_mc 20000 \
      --n_eval_prob_mc 2000 \
      --weight_modes one_step \
      --run_naive_t_test \
      --no-save_logged_data \
      --n_jobs 8 \
      --save_dir results/figure2 \
      --tag fig2_bernoulli_uniform_ts
done
```

### Figure 3: Beta rewards, `pi0 = ts_normal`, `pi1 = ts_normal`

This reproduces the sub-Gaussian reward-distribution experiment. The reward distributions are Beta with parameters `(0.35, 0.65)`, `(0.5, 0.5)`, and `(0.5, 0.5)`. The command below runs both BSI variance rules used in the figure: `hoeffding` and `empirical_variance`.

```bash
TOFFS="100 150 200 250 500"
VAR_RULES="hoeffding empirical_variance"

for VAR_RULE in $VAR_RULES; do
  for TOFF in $TOFFS; do
    python run.py \
        --env beta \
        --beta_alphas 0.35 0.5 0.5 \
        --beta_betas 0.65 0.5 0.5 \
        --var_estimation_beta $VAR_RULE \
        --pi0 ts_normal \
        --pi1 ts_normal \
        --T 500 \
        --T_offline $TOFF \
        --offline_reps 1000 \
        --alphas 0.10 \
        --infer_reps 20000 \
        --no-select_M \
        --n_policy_value_mc 20000 \
        --n_eval_prob_mc 2000 \
        --weight_modes one_step \
        --run_naive_t_test \
        --no-save_logged_data \
        --n_jobs 8 \
        --save_dir results/figure3 \
        --tag fig3_beta_ts_ts
  done
done
```


---

## Command-line arguments

`run.py` uses an `argparse` parser to define one experiment configuration. The
main groups of arguments are:

### Environment

| Argument | Description |
|---|---|
| `--env` | Reward model. Choose `normal`, `bernoulli`, or `beta`. |
| `--mus` | Arm means for `normal` and `bernoulli`. For `beta`, means are derived from `--beta_alphas` and `--beta_betas`. |
| `--sigmas` | Arm standard deviations for `normal`. |
| `--beta_alphas`, `--beta_betas` | Beta distribution parameters, one value per arm. Required when `--env beta`. |

### Logging and target policies

Use exactly one logging-policy specification:

- `--pi0`: adaptive or algorithmic logging policy, such as `uniform`,
  `epsilon_greedy`, `ts_normal`, or `ts_bernoulli`.
- `--behavior_policy`: static logging probabilities, such as
  `--behavior_policy 0.33 0.33 0.34`.

The target policy is set with `--pi1`. If omitted, it defaults to Thompson
Sampling: `ts_bernoulli` for Bernoulli environments and `ts_normal` otherwise.

Supported policy names:

| Argument | Choices |
|---|---|
| `--pi0` | `uniform`, `etc`, `batch_greedy`, `ts_normal`, `epsilon_greedy`, `ts_bernoulli` |
| `--pi1` | `uniform`, `etc`, `batch_greedy`, `ucb`, `epsilon_greedy`, `ts_normal`, `ts_bernoulli` |

### Horizons and replications

| Argument | Default | Description |
|---|---:|---|
| `--T` | `50` | Online horizon for the target policy. |
| `--T_offline` | `100` | Offline logged-data horizon. |
| `--offline_reps` | `100` | Number of Monte Carlo replications. |
| `--infer_reps` | selected automatically | Monte Carlo rollouts used inside BSI inference. Used directly only with `--no-select_M`. |
| `--alphas` | `0.05` | One or more significance levels, e.g. `--alphas 0.01 0.05 0.10`. |

### BSI-specific options

| Argument | Default | Description |
|---|---:|---|
| `--select_M` / `--no-select_M` | on | Select the BSI inner Monte Carlo size before running BSI. |
| `--Mmax` | `200000` | Upper budget for automatically selected `M`; if the estimated `M` is larger, the run uses `Mmax` and emits a warning. |
| `--M_m0` | `1000` | Pilot inner rollouts used for M selection. |
| `--M_bootstrap_reps` | `1000` | Bootstrap resamples used for M selection. |
| `--M_tau` | `0.05` | Tail probability for the M-selection error criterion. |
| `--M_rel_eps` | `0.05` | Target relative Monte Carlo error for the BSI width estimate. |
| `--estimate_sigma` | off | Estimate arm reward standard deviations in normal-style BSI. |
| `--var_estimation_beta` | `hoeffding` | Beta-normal BSI variance rule. Choose `hoeffding` or `empirical_variance`. |

For `--env beta`, BSI treats rewards through the normal-style BSI classes:
`AdaptiveNormalBSI` for adaptive logging and `NormalBSI` for non-adaptive
logging. The default `hoeffding` beta variance rule uses a fixed
sub-Gaussian standard deviation of `0.5`.

### Baseline controls

By default, the runner computes BSI and the OPE baselines. Use
`--baselines_only` to skip BSI and run only the baselines.

| Argument | Description |
|---|---|
| `--weight_modes` | Weighting rules for OPE baselines. Choices are `one_step` and `cumulative`; by default both are evaluated. |
| `--run_weighted_t_test` | Also report an IPW-style weighted t-test interval. |
| `--run_naive_t_test` | Also report a naive iid t-test interval on raw rewards. |
| `--cadr_min_samples` | Minimum number of samples before CADR starts using its running variance estimate. |
| `--n_eval_prob_mc` | Monte Carlo samples used to estimate target-policy action probabilities for stochastic policies. |

### Output, seeds, and parallelism

| Argument | Default | Description |
|---|---:|---|
| `--save_dir` | `results` | Directory for output JSON files. |
| `--tag` | `compare` | Prefix used in output filenames. |
| `--n_jobs` | `1` | Number of parallel workers. |
| `--rep_idx` | none | Run a single replication and save a partial JSON, useful for array jobs. |
| `--algo_seed` | `2026` | Seed for algorithm randomness. |
| `--table_seed` | `1013` | Seed for reward-table generation. |
| `--show_progress` | off | Show progress bars during replications. |
| `--no-save_logged_data` | off | Do not store logged actions, rewards, and behavior probabilities in the output JSON. |

To see the full parser directly, run:

```bash
python run.py --help
```

## Contextual BSI Extension

The original MAB implementation remains at the repository root. The contextual
bandit extension is kept in the `contextual/` package so its modules do not
collide with the original `algorithms.py`, `environments.py`, `baselines.py`,
or `inference.py` files.

Core contextual files:

| Path | Purpose |
|---|---|
| `contextual/contextual_bsi.py` | Contextual BSI estimator and inner simulation runner. |
| `contextual/algorithms.py` | Contextual ε-greedy and Thompson-sampling policies. |
| `contextual/environments.py` | Linear Gaussian and logistic Bernoulli contextual reward models. |
| `contextual/baselines.py` | Contextual IPW, DR, CADR, and ELFCB-style baselines. |
| `contextual/simulation.py` | Small simulation helpers used by examples and M-selection. |
| `contextual/select_inner_reps.py` | Pilot-bootstrap rule for selecting the BSI inner Monte Carlo size. |

Use `find_M.py` to estimate the inner Monte Carlo size before launching large
BSI experiments. It supports the original MAB setting, the sub-Gaussian Beta
MAB setting, and contextual BSI:

```bash
python find_M.py --help

python find_M.py \
    --mode mab \
    --env bernoulli \
    --mus 0.35,0.5,0.6 \
    --pi0 uniform \
    --pi1 ts_bernoulli \
    --T 500 \
    --T_offline_grid 100,500,1000 \
    --m0 1000 \
    --B 1000 \
    --out selected_M_mab.csv

python find_M.py \
    --mode subgaussian \
    --beta_alphas 0.35,0.5,0.5 \
    --beta_betas 0.65,0.5,0.5 \
    --pi0 ts_normal \
    --pi1 ts_normal \
    --T 500 \
    --T_offline_grid 100,500,1000 \
    --m0 1000 \
    --B 1000 \
    --out selected_M_subgaussian.csv

python find_M.py \
    --mode contextual \
    --contextual_envs linear_gaussian,logistic_bernoulli \
    --pairs uni_ts,ts_ts,eps_eps \
    --T 500 \
    --T_offline_grid 100,500,1000 \
    --m0 1000 \
    --B 1000 \
    --out selected_M_contextual.csv
```

## Methods

### BSI (this paper)

BSI constructs confidence intervals via the delta method / influence function,
using knowledge of the bandit algorithm's selection probabilities.

| Class | Environment | Logging policy |
|---|---|---|
| `AdaptiveNormalBSI` | Normal / Beta | Adaptive (TS, ε-greedy) |
| `AdaptiveBernoulliBSI` | Bernoulli | Adaptive (TS) |
| `NormalBSI` | Normal / Beta | Non-adaptive (uniform, static) |
| `BernoulliBSI` | Bernoulli | Non-adaptive (uniform, static) |

Each adaptive class produces two CI variants:

- **BSI** (`ci_width`): normal approximation — `norm.ppf(1-α/2) × se`
- **BSI (projection)** (`ci_width_proj`): chi-squared projection — `chi2.ppf(1-α, df) / T × √(gVg)`

### OPE baselines

| Key in output | Method |
|---|---|
| `elfcb_one_step` | ELFCB (empirical likelihood with covariate balancing) |
| `ipw_one_step` | IPW (inverse probability weighting) |
| `cadr_one_step` | CADR (covariate-adjusted doubly robust) |
| `dr_one_step` | DR (doubly robust) |
| `naive_t_test` | Naive t-test on raw rewards |

---
