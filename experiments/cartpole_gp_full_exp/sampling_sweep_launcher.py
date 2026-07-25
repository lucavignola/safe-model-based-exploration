"""Euler launcher for the Cartpole GP sampling-method/M sweep."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiment

from smbrl.utils.experiment_utils import (
    dict_permutations,
    generate_base_command,
    generate_run_commands,
)


PROJECT_NAME = "CartPoleGP"
ENTITY_NAME = "lvignola-eth-z-rich"
PARTICLE_SWEEP = [1, 10, 20, 30, 40, 50]
SAMPLING_METHODS = ["marginal", "rff"]
TRUNCATION_MODES = ["none", "posterior", "prior"]

HARDWARE_CONFIGS = {
    "4090_rtx": {"gpu_type": "rtx_4090", "cpus_per_task": 10},
    "rtx_a6000": {"gpu_type": "rtxa6000", "cpus_per_task": 8},
}


def build_sweep_configs(
        sampling_methods=None,
        truncation_modes=None,
        num_rff_features=512,
        rff_path_scale=None,
        particle_sweep=None,
        seeds=None,
):
    """Returns the 2 x 6 x 5 sweep while holding paper settings fixed."""

    if sampling_methods is None:
        sampling_methods = SAMPLING_METHODS
    if particle_sweep is None:
        particle_sweep = PARTICLE_SWEEP
    if truncation_modes is None:
        truncation_modes = ["none"]
    if seeds is None:
        seeds = list(range(5))
    beta = 3.0
    if rff_path_scale is None:
        rff_path_scale = beta
    config = {
        "alg_name": ["SBSRL"],
        "project_name": [PROJECT_NAME],
        "entity_name": [ENTITY_NAME],
        "seed": list(seeds),
        "gp_sampling_method": list(sampling_methods),
        "gp_sample_truncation": list(truncation_modes),
        "num_rff_features": [num_rff_features],
        "rff_path_scale": [rff_path_scale],
        "num_particles": list(particle_sweep),
        "num_gpus": [1],
        "num_training_steps": [500],
        "num_samples": [1_000],
        "num_elites": [100],
        "num_steps": [5],
        "alpha": [0.2],
        "exponent": [1.0],
        "lambda_constraint": [1e8],
        "icem_horizon": [30],
        "episode_length": [50],
        "action_repeat": [2],
        "num_offline_data": [20],
        "num_traj": [0],
        "max_position": [1.5],
        "violation_eps": [0.6],
        "beta": [beta],
        "use_precomputed_kernel_params": [0],
        "use_function_norms": [0],
        "use_optimism": [0],
        "use_pessimism": [1],
        "lambda_sigma": [0.0],
        "action_cost": [0.0],
        "uncertainty_eps": [300.0],
        "uncertainty_decay_factor": [10.0],
        "uncertainty_decay_mode": ["linear"],
        "uncertainty_constraint_threshold": [50.0],
        "default_task_index": [0],
        "optimizer": ["icem"],
        "log_wandb": [1],
    }
    return dict_permutations(config)


def main(args):
    methods = SAMPLING_METHODS if args.gp_sampling_method == "both" else [args.gp_sampling_method]
    flags_combinations = build_sweep_configs(
        sampling_methods=methods,
        truncation_modes=args.gp_sample_truncation,
        num_rff_features=args.num_rff_features,
        rff_path_scale=args.rff_path_scale,
        particle_sweep=args.particles,
        seeds=args.seeds,
    )
    logs_dir = "../"
    if args.mode == "euler":
        logs_dir = f"/cluster/scratch/lvignola/{PROJECT_NAME}/"

    commands = []
    for flags in flags_combinations:
        flags["logs_dir"] = logs_dir
        flags["wandb_notes"] = args.wandb_notes
        command = generate_base_command(experiment, flags=flags)
        commands.append(f"PYTHONPATH={REPO_ROOT} {command}")

    hardware = HARDWARE_CONFIGS[args.hardware]
    generate_run_commands(
        commands,
        num_cpus=hardware["cpus_per_task"],
        num_gpus=1,
        mode=args.mode,
        duration=getattr(args, "duration", "4:00:00"),
        prompt=not args.dry_run,
        dry=args.dry_run,
        gpu_type=hardware["gpu_type"],
        partition=getattr(args, "partition", None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["euler", "local", "local_async"], default="euler")
    parser.add_argument("--hardware", choices=list(HARDWARE_CONFIGS), default="4090_rtx")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--duration",
        default="4:00:00",
        help="Slurm wall time, for example 24:00:00.",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help=(
            "Optional explicit Slurm partition, for example gpuhe.24h. "
            "If omitted, Slurm selects the partition."
        ),
    )
    parser.add_argument("--gp_sampling_method", choices=["marginal", "rff", "both"], default="both")
    parser.add_argument(
        "--gp_sample_truncation",
        choices=TRUNCATION_MODES,
        nargs="+",
        default=["none"],
        help=(
            "One or more sample-projection modes: none, posterior "
            "(beta*sigma_n), or prior (beta*sqrt(k(z,z)))"
        ),
    )
    parser.add_argument("--num_rff_features", type=int, default=512)
    parser.add_argument(
        "--rff_path_scale",
        type=float,
        default=None,
        help="RFF residual scale; defaults to beta for a scale-matched comparison",
    )
    parser.add_argument("--particles", type=int, nargs="+", default=PARTICLE_SWEEP)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--wandb_notes", type=str, default="gp-sampling-M-sweep")
    main(parser.parse_args())
