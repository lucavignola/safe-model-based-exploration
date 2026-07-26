"""Euler launcher for Pendulum ActSafe sweeps."""

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


PROJECT_NAME = "PendulumGP"
ENTITY_NAME = "lvignola-eth-z-rich"
NUM_GPUS = 1
DEFAULT_PARTICLES = [30]

HARDWARE_CONFIGS = {
    "4090_rtx": {
        "gpu_type": "rtx_4090",
        "cpus_per_task": 10,
        "timeout_min": 240,
    },
    "rtx_a6000": {
        "gpu_type": "rtxa6000",
        "cpus_per_task": 8,
        "timeout_min": 240,
    },
    "cpu_only": {
        "gpu_type": None,
        "cpus_per_task": 4,
        "timeout_min": 300,
    },
}


def build_sweep_configs(
    *,
    particles=DEFAULT_PARTICLES,
    seeds=range(5),
    num_offline_data=0,
    num_safe_offline_data=0,
    match_theory_config=False,
):
    """Build an ActSafe-only sweep, optionally matching the SBSRL setup."""

    if not 0 <= num_safe_offline_data <= num_offline_data:
        raise ValueError(
            "Require 0 <= num_safe_offline_data <= num_offline_data."
        )

    config = {
        "alg_name": ["ActSafe"],
        "project_name": [PROJECT_NAME],
        "entity_name": [ENTITY_NAME],
        "seed": list(seeds),
        "num_particles": list(particles),
        "use_optimism": [1],
        "use_pessimism": [1],
        "actsafe_index": [-1],
        "env_margin_factor": [10.0],
        "reward_source": ["gym"],
        "num_training_steps": [1_000],
        "num_gpus": [NUM_GPUS],
        "num_samples": [1_000],
        "lambda_constraint": [1e7],
        "icem_horizon": [20],
        "num_elites": [100],
        "num_offline_data": [num_offline_data],
        "num_safe_offline_data": [num_safe_offline_data],
        "violation_eps": [0.0],
        "beta": [3.0],
        "log_wandb": [1],
    }

    if match_theory_config:
        # Match theory_sweep_launcher.py in every common environment, GP,
        # iCEM, D0 and evaluation setting. ActSafe's exploration objective and
        # optimism/pessimism choices remain the algorithmic differences.
        config.update({
            "project_name": ["PendulumGPTheoryAligned"],
            "num_training_steps": [0],
            "num_episodes": [10],
            "episode_length": [50],
            "action_repeat": [2],
            "max_abs_velocity": [6.0],
            "num_evaluation_trajectories": [5],
            "function_norm": [1.0],
            "gp_beta_mode": ["theorem"],
            "confidence_delta": [0.05],
            "information_gain_bound": ["diagonal"],
            "aleatoric_noise_in_prediction": [0],
            "constraint_mode": ["penalty"],
            "constraint_tolerance": [1e-6],
            "constraint_failure_mode": ["recovery"],
            "reward_dynamics_source": ["particles"],
            "lambda_constraint": [1e7],
            "violation_eps": [0.0],
            "gp_sampling_method": ["marginal"],
            "alpha": [0.2],
            "exponent": [0.2],
            "num_steps": [5],
        })

    return dict_permutations(config)


def main(args):
    configs = build_sweep_configs(
        particles=args.particles,
        seeds=args.seeds,
        num_offline_data=args.num_offline_data,
        num_safe_offline_data=args.num_safe_offline_data,
        match_theory_config=args.match_theory_config,
    )

    commands = []
    for flags in configs:
        logs_dir = "../"
        if args.mode == "euler":
            logs_dir = (
                f"/cluster/scratch/lvignola/"
                f"{flags['project_name']}/"
            )
        flags["logs_dir"] = logs_dir
        if args.wandb_notes:
            flags["wandb_notes"] = args.wandb_notes
        command = generate_base_command(experiment, flags=flags)
        commands.append(f"PYTHONPATH={REPO_ROOT} {command}")

    hardware = HARDWARE_CONFIGS[args.hardware]
    if args.duration is not None:
        duration = args.duration
    elif args.long_run:
        duration = "24:00:00"
    else:
        hours, minutes = divmod(hardware["timeout_min"], 60)
        duration = f"{hours}:{minutes:02d}:00"

    generate_run_commands(
        commands,
        num_cpus=hardware["cpus_per_task"],
        num_gpus=NUM_GPUS if hardware["gpu_type"] is not None else 0,
        mode=args.mode,
        duration=duration,
        prompt=not args.dry_run,
        dry=args.dry_run,
        gpu_type=hardware["gpu_type"],
        partition=args.partition,
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
    parser.add_argument("--long_run", action="store_true")
    parser.add_argument(
        "--duration",
        default=None,
        help=(
            "Explicit Slurm wall time, for example 24:00:00. Overrides "
            "--long_run and the hardware default."
        ),
    )
    parser.add_argument(
        "--partition",
        default="gpuhe.24h",
        help="Explicit Slurm partition.",
    )
    parser.add_argument(
        "--particles", type=int, nargs="+", default=DEFAULT_PARTICLES,
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(5)),
    )
    parser.add_argument("--num_offline_data", type=int, default=0)
    parser.add_argument("--num_safe_offline_data", type=int, default=0)
    parser.add_argument(
        "--match_theory_config",
        action="store_true",
        help=(
            "Match the common D0, GP, iCEM, hard-constraint and five-trajectory "
            "evaluation settings used by theory_sweep_launcher.py."
        ),
    )
    parser.add_argument(
        "--wandb_notes",
        default=None,
        help="W&B note/tag used to group the sweep.",
    )
    main(parser.parse_args())
