#!/bin/bash
set -euo pipefail

# Natural contextual experiment:
#   K=3, d=2, X=(1,Z1,Z2), Z ~ N(0, I_2)
#   beta_1=(0.10, 0.45, -0.35)
#   beta_2=(0.00,-0.40,  0.40)
#   beta_3=(-0.05,0.20,  0.50)
#
# Policy pairs:
#   eps_eps: pi0=contextual_epsilon, pi1=contextual_epsilon
#   ts_ts:   pi0=contextual_ts,      pi1=contextual_ts
#   uni_ts:  pi0=uniform,            pi1=contextual_ts
#   uni_eps: pi0=uniform,            pi1=contextual_epsilon

ENVS="${ENVS:-linear_gaussian,logistic_bernoulli}"
PAIRS="${PAIRS:-eps_eps,ts_ts,uni_ts}"
T_VALUE="${T_VALUE:-500}"
T_OFFLINE_GRID="${T_OFFLINE_GRID:-300,500,1000,2000,5000}"
OUTER_REPS="${OUTER_REPS:-1000}"
INNER_REPS="${INNER_REPS:-50000}"
SELECT_INNER_REPS="${SELECT_INNER_REPS:-true}"
M_MAX="${M_MAX:-50000}"
M_M0="${M_M0:-1000}"
M_BOOTSTRAP_REPS="${M_BOOTSTRAP_REPS:-1000}"
M_TAU="${M_TAU:-0.05}"
M_REL_EPS="${M_REL_EPS:-0.1}"
TRUTH_REPS="${TRUTH_REPS:-50000}"
TRUTH_BATCH_REPS="${TRUTH_BATCH_REPS:-1000}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-250}"
TS_PROB_MC="${TS_PROB_MC:-300}"
CONF_LEVEL="${CONF_LEVEL:-0.95}"
ALPHAS="${ALPHAS:-0.01,0.05,0.1,0.2}"
SEED="${SEED:-20260821}"
PARAM_SCENARIO="${PARAM_SCENARIO:-natural}"
CONTEXT_VAR="${CONTEXT_VAR:-1.0}"
MICROMAMBA_ENV="${MICROMAMBA_ENV:-healthkit}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Leave PBS_QUEUE empty to match the older PBS files that successfully routed jobs.
PBS_QUEUE="${PBS_QUEUE:-}"
TRUTH_WALLTIME="${TRUTH_WALLTIME:-01:00:00}"
BASELINE_WALLTIME="${BASELINE_WALLTIME:-01:00:00}"
BSI_WALLTIME="${BSI_WALLTIME:-02:00:00}"
TRUTH_MEM="${TRUTH_MEM:-16gb}"
BASELINE_MEM="${BASELINE_MEM:-16gb}"
BSI_MEM="${BSI_MEM:-64gb}"
SUBMIT_TRUTH="${SUBMIT_TRUTH:-true}"
SUBMIT_BASELINES="${SUBMIT_BASELINES:-true}"
SUBMIT_BSI="${SUBMIT_BSI:-true}"

RUN_PREFIX="${RUN_PREFIX:-natural_T500_M50000}"
GENERATED_DIR="${GENERATED_DIR:-.pbs_generated_${RUN_PREFIX}}"

mkdir -p logs "${GENERATED_DIR}"

count_csv() {
  local text="$1"
  awk -F',' '{print NF}' <<< "$text"
}

contains_csv() {
  local needle="$1"
  local text="$2"
  local item
  IFS=, read -ra items <<< "$text"
  for item in "${items[@]}"; do
    if [ "$item" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

chunks_per_truth_setting() {
  echo $(((TRUTH_REPS + TRUTH_BATCH_REPS - 1) / TRUTH_BATCH_REPS))
}

queue_line() {
  if [ -n "${PBS_QUEUE}" ]; then
    echo "#PBS -q ${PBS_QUEUE}"
  fi
}

submit_truth() {
  local pair="$1"
  local pi0="$2"
  local pi1="$3"
  local out_dir="contextual_truth_${RUN_PREFIX}_${pair}"
  local n_envs chunks total_tasks pbs_path
  n_envs=$(count_csv "$ENVS")
  chunks=$(chunks_per_truth_setting)
  total_tasks=$((n_envs * chunks))
  pbs_path="${GENERATED_DIR}/truth_${pair}.pbs"

  cat > "$pbs_path" <<EOF
#!/bin/bash
#PBS -N truth_${pair}
$(queue_line)
#PBS -l select=1:ncpus=1:mem=${TRUTH_MEM}
#PBS -l walltime=${TRUTH_WALLTIME}
#PBS -j oe
#PBS -o logs/truth_${RUN_PREFIX}_${pair}.log

set -euo pipefail
cd "\${PBS_O_WORKDIR:-\$HOME/contextBSI}"
mkdir -p logs
if [ -f "\$HOME/.bashrc" ]; then
  source "\$HOME/.bashrc"
fi
micromamba activate "${MICROMAMBA_ENV}"
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ARRAY_ID="\${PBS_ARRAY_INDEX:-\${PBS_ARRAYID:-1}}"

"${PYTHON_BIN}" estimate_contextual_truth_array.py \\
  --array_id "\${ARRAY_ID}" \\
  --envs "${ENVS}" \\
  --param_scenario "${PARAM_SCENARIO}" \\
  --context_var "${CONTEXT_VAR}" \\
  --ts_prob_mc "${TS_PROB_MC}" \\
  --pi0s "${pi0}" \\
  --pi1 "${pi1}" \\
  --T "${T_VALUE}" \\
  --truth_reps "${TRUTH_REPS}" \\
  --truth_batch_reps "${TRUTH_BATCH_REPS}" \\
  --inner_reps "${INNER_REPS}" \\
  --seed "${SEED}" \\
  --out_dir "${out_dir}"
EOF

  echo "Submitting truth ${pair}: ${total_tasks} jobs -> ${out_dir}"
  qsub -J "1-${total_tasks}" "$pbs_path"
}

submit_baselines() {
  local pair="$1"
  local pi0="$2"
  local pi1="$3"
  local out_dir="contextual_baselines_${RUN_PREFIX}_${pair}"
  local n_methods n_envs n_toff total_rows total_tasks pbs_path
  n_methods=4
  n_envs=$(count_csv "$ENVS")
  n_toff=$(count_csv "$T_OFFLINE_GRID")
  total_rows=$((n_methods * n_envs * n_toff * OUTER_REPS))
  total_tasks=$(((total_rows + BASELINE_BATCH_SIZE - 1) / BASELINE_BATCH_SIZE))
  pbs_path="${GENERATED_DIR}/baselines_${pair}.pbs"

  cat > "$pbs_path" <<EOF
#!/bin/bash
#PBS -N base_${pair}
$(queue_line)
#PBS -l select=1:ncpus=1:mem=${BASELINE_MEM}
#PBS -l walltime=${BASELINE_WALLTIME}
#PBS -j oe
#PBS -o logs/baselines_${RUN_PREFIX}_${pair}.log

set -euo pipefail
cd "\${PBS_O_WORKDIR:-\$HOME/contextBSI}"
mkdir -p logs
if [ -f "\$HOME/.bashrc" ]; then
  source "\$HOME/.bashrc"
fi
micromamba activate "${MICROMAMBA_ENV}"
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ARRAY_ID="\${PBS_ARRAY_INDEX:-\${PBS_ARRAYID:-1}}"

"${PYTHON_BIN}" run_contextual_baseline_batch_array.py \\
  --array_id "\${ARRAY_ID}" \\
  --methods "ipw,dr,cadr,elfcb" \\
  --envs "${ENVS}" \\
  --param_scenario "${PARAM_SCENARIO}" \\
  --context_var "${CONTEXT_VAR}" \\
  --ts_prob_mc "${TS_PROB_MC}" \\
  --pi0s "${pi0}" \\
  --pi1 "${pi1}" \\
  --T "${T_VALUE}" \\
  --T_offline_grid "${T_OFFLINE_GRID}" \\
  --inner_reps "${INNER_REPS}" \\
  --outer_reps "${OUTER_REPS}" \\
  --truth_reps "${TRUTH_REPS}" \\
  --defer_truth \\
  --conf_level "${CONF_LEVEL}" \\
  --alphas "${ALPHAS}" \\
  --seed "${SEED}" \\
  --batch_size "${BASELINE_BATCH_SIZE}" \\
  --out_dir "${out_dir}"
EOF

  echo "Submitting baselines ${pair}: ${total_tasks} jobs, ${total_rows} rows -> ${out_dir}"
  qsub -J "1-${total_tasks}" "$pbs_path"
}

submit_bsi_one() {
  local pair="$1"
  local pi0="$2"
  local pi1="$3"
  local env="$4"
  local toff="$5"
  local out_dir="contextual_bsi_${RUN_PREFIX}_${pair}_${env}_Toff${toff}"
  local pbs_path="${GENERATED_DIR}/bsi_${pair}_${env}_Toff${toff}.pbs"
  local select_inner_args=""
  if [ "${SELECT_INNER_REPS}" = "true" ] || [ "${SELECT_INNER_REPS}" = "1" ] || [ "${SELECT_INNER_REPS}" = "yes" ]; then
    select_inner_args="--select_inner_reps"
  fi

  cat > "$pbs_path" <<EOF
#!/bin/bash
#PBS -N bsi_${pair}_${toff}
$(queue_line)
#PBS -l select=1:ncpus=1:mem=${BSI_MEM}
#PBS -l walltime=${BSI_WALLTIME}
#PBS -j oe
#PBS -o logs/bsi_${RUN_PREFIX}_${pair}_${env}_Toff${toff}.log

set -euo pipefail
cd "\${PBS_O_WORKDIR:-\$HOME/contextBSI}"
mkdir -p logs
if [ -f "\$HOME/.bashrc" ]; then
  source "\$HOME/.bashrc"
fi
micromamba activate "${MICROMAMBA_ENV}"
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ARRAY_ID="\${PBS_ARRAY_INDEX:-\${PBS_ARRAYID:-1}}"

"${PYTHON_BIN}" run_contextual_method_array.py \\
  --array_id "\${ARRAY_ID}" \\
  --methods "bsi" \\
  ${select_inner_args} \\
  --Mmax "${M_MAX}" \\
  --M_m0 "${M_M0}" \\
  --M_bootstrap_reps "${M_BOOTSTRAP_REPS}" \\
  --M_tau "${M_TAU}" \\
  --M_rel_eps "${M_REL_EPS}" \\
  --envs "${env}" \\
  --param_scenario "${PARAM_SCENARIO}" \\
  --context_var "${CONTEXT_VAR}" \\
  --ts_prob_mc "${TS_PROB_MC}" \\
  --pi0s "${pi0}" \\
  --pi1 "${pi1}" \\
  --T "${T_VALUE}" \\
  --T_offline_grid "${toff}" \\
  --inner_reps "${INNER_REPS}" \\
  --outer_reps "${OUTER_REPS}" \\
  --truth_reps "${TRUTH_REPS}" \\
  --defer_truth \\
  --conf_level "${CONF_LEVEL}" \\
  --alphas "${ALPHAS}" \\
  --seed "${SEED}" \\
  --out_dir "${out_dir}"
EOF

  echo "Submitting BSI ${pair} ${env} Toff=${toff}: ${OUTER_REPS} jobs -> ${out_dir}"
  qsub -J "1-${OUTER_REPS}" "$pbs_path"
}

echo "Natural contextual T=500 settings:"
echo "  ENVS=${ENVS}"
echo "  PAIRS=${PAIRS}"
echo "  T=${T_VALUE}"
echo "  T_OFFLINE_GRID=${T_OFFLINE_GRID}"
echo "  OUTER_REPS=${OUTER_REPS}"
echo "  INNER_REPS=${INNER_REPS}"
echo "  SELECT_INNER_REPS=${SELECT_INNER_REPS}"
echo "  M_MAX=${M_MAX}, M_M0=${M_M0}, M_BOOTSTRAP_REPS=${M_BOOTSTRAP_REPS}, M_TAU=${M_TAU}, M_REL_EPS=${M_REL_EPS}"
echo "  TRUTH_REPS=${TRUTH_REPS}"
echo "  PARAM_SCENARIO=${PARAM_SCENARIO}"
echo "  CONTEXT_VAR=${CONTEXT_VAR}"
echo "  TS_PROB_MC=${TS_PROB_MC}"
echo "  PBS_QUEUE=${PBS_QUEUE:-<auto>}"
echo "  TRUTH_WALLTIME=${TRUTH_WALLTIME}, TRUTH_MEM=${TRUTH_MEM}"
echo "  BASELINE_WALLTIME=${BASELINE_WALLTIME}, BASELINE_MEM=${BASELINE_MEM}"
echo "  BSI_WALLTIME=${BSI_WALLTIME}, BSI_MEM=${BSI_MEM}"
echo "  RUN_PREFIX=${RUN_PREFIX}"

POLICY_PAIRS=(
  "eps_eps:contextual_epsilon:contextual_epsilon"
  "ts_ts:contextual_ts:contextual_ts"
  "uni_ts:uniform:contextual_ts"
  "uni_eps:uniform:contextual_epsilon"
)

for spec in "${POLICY_PAIRS[@]}"; do
  IFS=: read -r pair pi0 pi1 <<< "$spec"
  if ! contains_csv "$pair" "$PAIRS"; then
    continue
  fi
  if [ "$SUBMIT_TRUTH" = "true" ]; then
    submit_truth "$pair" "$pi0" "$pi1"
  else
    echo "Skipping truth ${pair}."
  fi
  if [ "$SUBMIT_BASELINES" = "true" ]; then
    submit_baselines "$pair" "$pi0" "$pi1"
  else
    echo "Skipping baselines ${pair}."
  fi
  IFS=, read -ra env_arr <<< "$ENVS"
  IFS=, read -ra toff_arr <<< "$T_OFFLINE_GRID"
  if [ "$SUBMIT_BSI" = "true" ]; then
    for env in "${env_arr[@]}"; do
      for toff in "${toff_arr[@]}"; do
        submit_bsi_one "$pair" "$pi0" "$pi1" "$env" "$toff"
      done
    done
  else
    echo "Skipping BSI ${pair}."
  fi
done

cat <<EOF

Submitted requested arrays.

Truth folders:
  contextual_truth_${RUN_PREFIX}_eps_eps
  contextual_truth_${RUN_PREFIX}_ts_ts
  contextual_truth_${RUN_PREFIX}_uni_ts

Baseline folders:
  contextual_baselines_${RUN_PREFIX}_eps_eps
  contextual_baselines_${RUN_PREFIX}_ts_ts
  contextual_baselines_${RUN_PREFIX}_uni_ts

BSI folders:
  contextual_bsi_${RUN_PREFIX}_<pair>_<env>_Toff<T_off>
EOF
