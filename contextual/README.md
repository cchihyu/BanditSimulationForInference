# Contextual BSI

This package contains the contextual bandit extension of BSI. The original
multi-armed bandit implementation remains at the repository root; contextual
code is isolated here to avoid name collisions with the original modules.

## Main Modules

| File | Purpose |
|---|---|
| `contextual_bsi.py` | Contextual BSI estimator and inner Monte Carlo simulation. |
| `algorithms.py` | Contextual epsilon-greedy and Thompson sampling policies. |
| `environments.py` | Linear Gaussian and logistic Bernoulli contextual reward models. |
| `baselines.py` | Contextual IPW, DR, CADR, and ELFCB-style baselines. |
| `simulation.py` | Small simulation helpers used by examples and M-selection. |
| `select_inner_reps.py` | Pilot-bootstrap rule for selecting the BSI inner Monte Carlo size. |

## Running

Run commands from the repository root using module syntax:

```bash
python -m contextual.select_inner_reps --help
```
