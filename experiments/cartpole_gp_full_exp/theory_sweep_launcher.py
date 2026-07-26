"""Euler launcher for the theory-aligned Cartpole GP/M sweep."""

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


PROJECT_NAME = "CartPoleGPTheoryAligned"
ENTITY_NAME = "lvignola-eth-z-rich"
PARTICLE_SWEEP = [1, 10, 20, 30, 40, 50]
# Practical clipping/calibration hyperparameter sweep. Edit this list or
# override it with, for example, ``--function_norm 0.5 1.0 2.0``.
FUNCTION_NORM_SWEEP = [1.0]
SAMPLING_MODE_CONFIGS = {
    "prior": {
        "gp_sampling_method": "rff",
        "gp_path_source": "prior",
    },
    "posterior": {
        "gp_sampling_method": "rff",
        "gp_path_source": "posterior",
    },
    "ts1": {
        "gp_sampling_method": "marginal",
        "gp_path_source": "posterior",
    },
}
TRUNCATION_MODES = ("recursive", "posterior", "none", "prior")

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
        sampling_modes=("prior", "posterior", "ts1"),
        truncation_modes=("recursive",),
        num_offline_data=20,
        num_safe_offline_data=0,
        num_samples=1_000,
        num_elites=50,
        num_steps=5,
        constraint_mode="penalty",
        lambda_constraint=1e8,
        reward_dynamics_source="particles",
        aleatoric_noise_in_prediction=True,
        gp_marginal_sample_scale=3.0,
        gp_hyperparameter_updates=("freeze_after_d0",),
        num_training_steps=500,
        use_precomputed_kernel_params=False,
        num_evaluation_trajectories=5,
        log_gp_diagnostics=False,
        use_empirical_function_norms=False,
        rkhs_norm_safety_factor=1.0,
        confidence_delta=0.05,
        information_gain_bound="diagonal",
):
    """Builds matched legacy-penalty iCEM sweeps over GP path modes."""

    if particle_sweep is None:
        particle_sweep = PARTICLE_SWEEP
    if seeds is None:
        seeds = list(range(5))
    function_norms = (
        list(function_norm)
        if isinstance(function_norm, (list, tuple))
        else [function_norm]
    )
    sampling_modes = (
        list(sampling_modes)
        if isinstance(sampling_modes, (list, tuple))
        else [sampling_modes]
    )
    truncation_modes = (
        list(truncation_modes)
        if isinstance(truncation_modes, (list, tuple))
        else [truncation_modes]
    )
    gp_hyperparameter_updates = (
        list(gp_hyperparameter_updates)
        if isinstance(gp_hyperparameter_updates, (list, tuple))
        else [gp_hyperparameter_updates]
    )
    lifecycle_aliases = {
        "none": "none",
        "d0": "freeze_after_d0",
        "every": "every_episode",
        "freeze_after_d0": "freeze_after_d0",
        "every_episode": "every_episode",
    }
    gp_hyperparameter_updates = [
        lifecycle_aliases[mode] for mode in gp_hyperparameter_updates
    ]
    offline_data_sweep = (
        list(num_offline_data)
        if isinstance(num_offline_data, (list, tuple))
        else [num_offline_data]
    )
    safe_offline_data_sweep = (
        list(num_safe_offline_data)
        if isinstance(num_safe_offline_data, (list, tuple))
        else [num_safe_offline_data]
    )
    unknown_modes = set(sampling_modes) - set(SAMPLING_MODE_CONFIGS)
    if unknown_modes:
        raise ValueError(f"Unknown sampling modes: {sorted(unknown_modes)}")
    unknown_truncation_modes = set(truncation_modes) - set(
        TRUNCATION_MODES
    )
    if unknown_truncation_modes:
        raise ValueError(
            "Unknown GP sample truncation modes: "
            f"{sorted(unknown_truncation_modes)}"
        )
    config = {
        "alg_name": ["SBSRL"],
        "project_name": [PROJECT_NAME],
        "entity_name": [ENTITY_NAME],
        "seed": list(seeds),
        "num_particles": list(particle_sweep),
        "aleatoric_noise_in_prediction": [
            int(aleatoric_noise_in_prediction)
        ],
        "gp_marginal_sample_scale": [gp_marginal_sample_scale],
        "num_rff_features": [num_rff_features],
        "rff_path_scale": [1.0],
        "gp_beta_mode": ["theorem"],
        "confidence_delta": [confidence_delta],
        "information_gain_bound": [information_gain_bound],
        "rkhs_norm_safety_factor": [rkhs_norm_safety_factor],
        "use_precomputed_kernel_params": [
            int(use_precomputed_kernel_params)
        ],
        "use_function_norms": [int(use_empirical_function_norms)],
        "function_norm": function_norms,
        "constraint_mode": [constraint_mode],
        "constraint_tolerance": [1e-6],
        "constraint_failure_mode": ["recovery"],
        "reward_dynamics_source": [reward_dynamics_source],
        "violation_eps": [0.0],
        "num_gpus": [1],
        "num_training_steps": [num_training_steps],
        "gp_hyperparameter_update": gp_hyperparameter_updates,
        "num_samples": [num_samples],
        "num_elites": [num_elites],
        "num_steps": [num_steps],
        "alpha": [0.2],
        "exponent": [1.0],
        "lambda_constraint": [lambda_constraint],
        "icem_horizon": [30],
        "episode_length": [50],
        "num_episodes": [5],
        "action_repeat": [2],
        "num_traj": [0],
        "max_position": [1.5],
        "use_optimism": [0],
        "use_pessimism": [1],
        "lambda_sigma": [0.0],
        "uncertainty_eps": [0.0],
        "uncertainty_scale_with_beta": [1],
        "uncertainty_decay_factor": [10.0],
        "uncertainty_decay_mode": ["linear"],
        "uncertainty_constraint_threshold": [0.0],
        "action_cost": [0.0],
        "default_task_index": [0],
        "optimizer": ["icem"],
        "num_evaluation_trajectories": [num_evaluation_trajectories],
        "log_gp_diagnostics": [int(log_gp_diagnostics)],
        "log_wandb": [1],
    }
    configs = []
    for base_config in dict_permutations(config):
        for total_data in offline_data_sweep:
            for safe_data in safe_offline_data_sweep:
                if not 0 <= safe_data <= total_data:
                    continue
                for sampling_mode in sampling_modes:
                    mode_config = SAMPLING_MODE_CONFIGS[sampling_mode]
                    for truncation_mode in truncation_modes:
                        configs.append({
                            **base_config,
                            **mode_config,
                            "gp_sample_truncation": truncation_mode,
                            "num_offline_data": total_data,
                            "num_safe_offline_data": safe_data,
                            "gp_prior_condition_on_initial_data": int(
                                total_data > 0
                            ),
                        })
    if not configs:
        raise ValueError(
            "No valid D0 configurations: require "
            "0 <= num_safe_offline_data <= num_offline_data."
        )
    return configs


def main(args):
    configs = build_sweep_configs(
        particle_sweep=args.particles,
        seeds=args.seeds,
        num_rff_features=args.num_rff_features,
        function_norm=args.function_norm,
        sampling_modes=getattr(
            args,
            "sampling_modes",
            ["prior", "posterior", "ts1"],
        ),
        truncation_modes=getattr(
            args, "gp_sample_truncation", ["recursive"]
        ),
        num_offline_data=args.num_offline_data,
        num_safe_offline_data=args.num_safe_offline_data,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_steps=args.num_steps,
        constraint_mode=getattr(args, "constraint_mode", "penalty"),
        lambda_constraint=getattr(args, "lambda_constraint", 1e8),
        reward_dynamics_source=getattr(
            args, "reward_dynamics_source", "particles"
        ),
        aleatoric_noise_in_prediction=bool(
            getattr(args, "aleatoric_noise_in_prediction", 1)
        ),
        gp_hyperparameter_updates=getattr(
            args,
            "gp_hyperparameter_update",
            ["freeze_after_d0"],
        ),
        gp_marginal_sample_scale=getattr(
            args, "gp_marginal_sample_scale", 3.0
        ),
        num_training_steps=getattr(args, "num_training_steps", 500),
        use_precomputed_kernel_params=bool(
            getattr(args, "use_precomputed_kernel_params", 0)
        ),
        num_evaluation_trajectories=getattr(
            args, "num_evaluation_trajectories", 5
        ),
        log_gp_diagnostics=getattr(args, "log_gp_diagnostics", False),
        use_empirical_function_norms=args.use_empirical_function_norms,
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
        duration=getattr(args, "duration", "4:00:00"),
        prompt=not args.dry_run,
        dry=args.dry_run,
        gpu_type=hardware["gpu_type"],
        partition=getattr(args, "partition", None),
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
    parser.add_argument("--particles", type=int, nargs="+", default=PARTICLE_SWEEP)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--num_rff_features", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=1_000)
    parser.add_argument("--num_elites", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument(
        "--num_evaluation_trajectories",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--log_gp_diagnostics",
        action="store_true",
        help="Enable expensive visited-path clipping/calibration diagnostics.",
    )
    parser.add_argument(
        "--sampling_modes",
        choices=list(SAMPLING_MODE_CONFIGS),
        nargs="+",
        default=["prior", "posterior", "ts1"],
        help=(
            "prior: one fixed prior RFF path bank; posterior: new fixed "
            "posterior RFF paths per episode; ts1: stepwise marginal "
            "posterior sampling. Clipping is selected independently with "
            "--gp_sample_truncation."
        ),
    )
    parser.add_argument(
        "--gp_sample_truncation",
        choices=TRUNCATION_MODES,
        nargs="+",
        default=["recursive"],
        help=(
            "recursive intersects the initial and all previous confidence "
            "tubes; posterior clips only to the current beta*sigma_n tube; "
            "none disables clipping; prior uses the current mean with the "
            "initial kernel standard deviation."
        ),
    )
    parser.add_argument(
        "--constraint_mode",
        choices=["penalty", "hard"],
        default="penalty",
    )
    parser.add_argument(
        "--lambda_constraint",
        type=float,
        default=1e8,
        help="Penalty coefficient in reward-lambda*relu(cost).",
    )
    parser.add_argument(
        "--reward_dynamics_source",
        choices=["particles", "posterior_mean"],
        default="particles",
        help=(
            "particles restores the legacy iCEM reward aggregation; "
            "posterior_mean performs a separate deterministic reward rollout."
        ),
    )
    parser.add_argument(
        "--aleatoric_noise_in_prediction",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Sample the learned GP likelihood scale as process noise inside "
            "planning rollouts. The submitted experiments used 1."
        ),
    )
    parser.add_argument(
        "--gp_hyperparameter_update",
        choices=[
            "none",
            "d0",
            "every",
            "freeze_after_d0",
            "every_episode",
        ],
        nargs="+",
        default=["freeze_after_d0"],
        help=(
            "Fit once on D0 and freeze, or refit after every accumulated-data "
            "update. Passing both creates a matched lifecycle ablation."
        ),
    )
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=500,
        help="Kernel-hyperparameter optimization steps at each enabled fit.",
    )
    parser.add_argument(
        "--gp_marginal_sample_scale",
        type=float,
        default=3.0,
        help=(
            "TS1 epistemic scale: 3 reproduces the submitted heuristic; "
            "1 is an uninflated GP marginal draw. This does not change the "
            "beta used by truncation."
        ),
    )
    parser.add_argument(
        "--use_precomputed_kernel_params",
        type=int,
        choices=[0, 1],
        default=0,
        help="Initialize kernel parameters from the stored Cartpole values.",
    )
    parser.add_argument(
        "--num_offline_data",
        type=int,
        nargs="+",
        default=[20],
    )
    parser.add_argument(
        "--num_safe_offline_data",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "D0 points assigned to the local safe-equilibrium design; the "
            "remaining points stay uniformly random."
        ),
    )
    parser.add_argument(
        "--function_norm",
        type=float,
        nargs="+",
        default=FUNCTION_NORM_SWEEP,
    )
    parser.add_argument(
        "--use_empirical_function_norms",
        action="store_true",
        help=(
            "Use the stored finite-design norm estimate instead of "
            "--function_norm."
        ),
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
        default="theory-sampling-M-sweep-D0-penalty-zero-tightening",
    )
    main(parser.parse_args())
