"""Euler launcher for the theory-aligned Pendulum GP/M sweep."""

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


PROJECT_NAME = "PendulumGPTheoryAligned"
ENTITY_NAME = "lvignola-eth-z-rich"
PARTICLE_SWEEP = [1, 10, 20, 30, 40, 50]
# Practical clipping/calibration hyperparameter sweep. Edit this list or
# override it with, for example, ``--function_norm 0.5 1.0 2.1``.
FUNCTION_NORM_SWEEP = [1.0]

HARDWARE_CONFIGS = {
    "4090_rtx": {"gpu_type": "rtx_4090", "cpus_per_task": 10},
    "rtx_a6000": {"gpu_type": "rtxa6000", "cpus_per_task": 8},
}


def build_sweep_configs(
        *,
        particle_sweep=None,
        seeds=None,
        num_rff_features=512,
        function_norm=FUNCTION_NORM_SWEEP,
        num_samples=1_000,
        num_elites=100,
        num_steps=5,
        rkhs_norm_safety_factor=1.0,
        confidence_delta=0.05,
        information_gain_bound="diagonal",
):
    """Builds the fixed-prior, recursive-truncation, hard-iCEM sweep.

    ``function_norm`` is exposed as a practical confidence-width
    hyperparameter. It is not automatically a certified continuous-domain
    RKHS bound.
    """

    if particle_sweep is None:
        particle_sweep = PARTICLE_SWEEP
    if seeds is None:
        seeds = list(range(5))
    function_norms = (
        list(function_norm)
        if isinstance(function_norm, (list, tuple))
        else [function_norm]
    )
    config = {
        "alg_name": ["SBSRL"],
        "project_name": [PROJECT_NAME],
        "entity_name": [ENTITY_NAME],
        "seed": list(seeds),
        "num_particles": list(particle_sweep),
        "gp_sampling_method": ["rff"],
        "gp_path_source": ["prior"],
        "gp_sample_truncation": ["recursive"],
        "aleatoric_noise_in_prediction": [0],
        "gp_prior_condition_on_initial_data": [0],
        "num_rff_features": [num_rff_features],
        "rff_path_scale": [1.0],
        "gp_beta_mode": ["theorem"],
        "confidence_delta": [confidence_delta],
        "information_gain_bound": [information_gain_bound],
        "function_norm": function_norms,
        "rkhs_norm_safety_factor": [rkhs_norm_safety_factor],
        "constraint_mode": ["hard"],
        "constraint_tolerance": [1e-6],
        "constraint_failure_mode": ["recovery"],
        "reward_dynamics_source": ["posterior_mean"],
        "violation_eps": [0.0],
        "num_gpus": [1],
        "num_training_steps": [0],
        "num_samples": [num_samples],
        "num_elites": [num_elites],
        "num_steps": [num_steps],
        "alpha": [0.2],
        "exponent": [0.2],
        "lambda_constraint": [0.0],
        "icem_horizon": [20],
        "episode_length": [50],
        "num_episodes": [10],
        "action_repeat": [2],
        "num_offline_data": [0],
        "max_abs_velocity": [6.0],
        "env_margin_factor": [10.0],
        "reward_source": ["gym"],
        "use_optimism": [0],
        "use_pessimism": [1],
        "lambda_sigma": [0.0],
        "uncertainty_eps": [0.0],
        "uncertainty_decay_factor": [10.0],
        "uncertainty_decay_mode": ["linear"],
        "uncertainty_constraint_threshold": [0.0],
        "action_cost": [0.0],
        "default_task_index": [0],
        "log_wandb": [1],
    }
    return dict_permutations(config)


def main(args):
    configs = build_sweep_configs(
        particle_sweep=args.particles,
        seeds=args.seeds,
        num_rff_features=args.num_rff_features,
        function_norm=args.function_norm,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_steps=args.num_steps,
        rkhs_norm_safety_factor=args.rkhs_norm_safety_factor,
        confidence_delta=args.confidence_delta,
        information_gain_bound=args.information_gain_bound,
    )
    logs_dir = "../"
    if args.mode == "euler":
        logs_dir = f"/cluster/scratch/lvignola/{PROJECT_NAME}/"

    commands = []
    for flags in configs:
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
        duration="4:00:00",
        prompt=not args.dry_run,
        dry=args.dry_run,
        gpu_type=hardware["gpu_type"],
        partition=None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["euler", "local", "local_async"],
        default="euler",
    )
    parser.add_argument(
        "--hardware",
        choices=list(HARDWARE_CONFIGS),
        default="4090_rtx",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--particles", type=int, nargs="+", default=PARTICLE_SWEEP)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--num_rff_features", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=1_000)
    parser.add_argument("--num_elites", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument(
        "--function_norm",
        type=float,
        nargs="+",
        default=FUNCTION_NORM_SWEEP,
    )
    parser.add_argument("--rkhs_norm_safety_factor", type=float, default=1.0)
    parser.add_argument("--confidence_delta", type=float, default=0.05)
    parser.add_argument(
        "--information_gain_bound",
        choices=["diagonal", "observed"],
        default="diagonal",
    )
    parser.add_argument(
        "--wandb_notes",
        type=str,
        default="theory-aligned-prior-hard-recovery-zero-tightening",
    )
    main(parser.parse_args())
