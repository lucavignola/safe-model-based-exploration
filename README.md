# SBSRL: GP experiments

Anonymous repository for the paper submission. This codebase contains the GP-based implementation of SBSRL and of several other safe and exploratory model-based RL methods which we use as baselines.

## Overview

The repository is organized around two main components:

- `smbrl/`: algorithms, dynamics models, reward definitions, optimizers, and utilities.
- `experiments/`: runnable experiment entry points and launchers for pendulum, and cartpole.
- `configs/`: configuration files containing the hyperparameters used in our experiments.

The experiments are generated procedurally from the simulators in this repository; no external public dataset is required.

## Installation

Recommended environment:

```bash
conda create -n smbrl_gp python=3.11
conda activate smbrl_gp
pip install -r requirements_compatible.txt
pip install -e .
```

Important: The repository expects the `mbpo` and `mbrl` packages to be available in the environment. For double-blind review, anonymized mirrors of those dependencies can be provided to reviewers on request.

## Reproducing Experiments

The main experiment entry points live under `experiments/*_gp_full_exp/experiment.py`. Note that the racecar environment is not working due to reasons that are not related to this paper. Therefore, the reviewers should focus their attention on the pendulum and cartpole environments.

Examples of usage:

```bash
python experiments/pendulum_gp_full_exp/experiment.py --alg_name SBSRL --log_wandb 0
python experiments/cartpole_gp_full_exp/experiment.py --alg_name SBSRL --log_wandb 0
```

Launcher-based runs:

```bash
python experiments/pendulum_gp_full_exp/launcher.py --mode local --hardware 4090_rtx
python experiments/cartpole_gp_full_exp/launcher.py --mode local --hardware 4090_rtx
```

If you have a cluster environment that supports `sbatch`, the same launchers can be run with `--mode cluster`.

## Data Access and Preparation

There is no separate raw-data download step. The code generates its own training and evaluation data by rolling out the environments and sampling offline transitions inside the experiment scripts. Outputs are written to the following tracked locations:

- `runs/` for experiment artifacts.
- `logs/` for run logs.
- `saved_data/` for stored experiment data and derived outputs.

## Experimental Setting and Details

The  `configs/` folder contains all the hyperparameters that were used to obtain the results presented in the paper. The remaining training and test details are specified in the experiment files and launcher scripts.
