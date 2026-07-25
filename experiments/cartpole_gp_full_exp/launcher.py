from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiment
from smbrl.utils.experiment_utils import (
    generate_run_commands,
    generate_base_command,
    dict_permutations,
)
import argparse

PROJECT_NAME = "CartPoleGP"
ENTITY = "lvignola-eth-z-rich"
NUM_GPUS = 1

# Hardware configurations matching working Hydra setup
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
        "timeout_min": 300,  # Increased from 240 to 300 min (5h)
    },
}

_applicable_configs = {
    "project_name": [PROJECT_NAME],
    "num_training_steps": [500],
    "episode_length": [50],
    "action_repeat": [2],
    "seed": list(range(5)),
    "entity_name": [ENTITY],
    "num_gpus": [NUM_GPUS],
    "beta": [3.0],
    "use_precomputed_kernel_params": [
        0,
    ],
    "use_function_norms": [0],
    "num_offline_data": [20],
    "max_position": [1.5],
    "num_samples": [1000],
    "icem_horizon": [
        30,
    ],
    "num_elites": [100],
    "num_steps": [5],
    "violation_eps":  [0.6],
    "num_traj": [0],  # 0=uniform grid sampling, >0=trajectory-based sampling
    "gp_sample_truncation": ["none"],
}
num_particles = [30]
_applicable_configs_actsafe = {
    "alg_name": ["ActSafe"],
    "use_optimism": [1],
    "use_pessimism": [1],
    "num_particles": num_particles,
    "actsafe_index": [-1],
} | _applicable_configs

_applicable_configs_actsafe_no_pessimism = {
    "alg_name": ["ActSafe"],
    "use_optimism": [0],
    "use_pessimism": [0],
    "num_particles": [1],
} | _applicable_configs

_applicable_configs_opax = {
    "alg_name": ["OPAX"],
    "use_optimism": [1],
    "use_pessimism": [1],
    "num_particles": num_particles,
} | _applicable_configs

_applicable_configs_sbsrl = {
    "alg_name": ["SBSRL"],
    "use_optimism": [0],
    "use_pessimism": [1],
    "num_particles": num_particles + [1],
    "lambda_sigma": [0],
    "action_cost": [0.0, 0.01],
    "uncertainty_eps": [300],
    "uncertainty_decay_factor": [10.0],
    "uncertainty_decay_mode": ["linear", "log_sigma_eps"],
    "uncertainty_constraint_threshold": [10.0],
    "default_task_index": [0],
} | _applicable_configs

_applicable_configs_safehucrl = {
    "alg_name": ["SafeHUCRL"],
    "use_optimism": [1],
    "use_pessimism": [1],
    "num_particles": num_particles,
} | _applicable_configs

def main(args):
    command_list = []
    sweep_config = dict(_applicable_configs_actsafe)
    sweep_config["num_particles"] = list(args.particles)
    sweep_config["seed"] = list(args.seeds)
    run_configs = dict_permutations(sweep_config)

    for flags in run_configs:
        flags = dict(flags)
        if args.match_theory_config:
            # Match the common environment, GP, iCEM, D0 and evaluation
            # settings of theory_sweep_launcher.py. The remaining differences
            # are the algorithm (ActSafe) and its exploration objective.
            flags.update({
                "project_name": "CartPoleGPTheoryAligned",
                "num_training_steps": 0,
                "num_episodes": 5,
                "num_safe_offline_data": 1,
                "num_evaluation_trajectories": 5,
                "use_precomputed_kernel_params": 1,
                "function_norm": 1.0,
                "gp_beta_mode": "theorem",
                "confidence_delta": 0.05,
                "information_gain_bound": "diagonal",
                "aleatoric_noise_in_prediction": 0,
                "constraint_mode": "hard",
                "constraint_tolerance": 1e-6,
                "constraint_failure_mode": "recovery",
                "reward_dynamics_source": "posterior_mean",
                "violation_eps": 0.0,
                "gp_sampling_method": "marginal",
                "gp_path_source": "posterior",
                "gp_sample_truncation": "recursive",
            })
        logs_dir = "../"
        if args.mode == "euler":
            logs_dir = (
                f"/cluster/scratch/lvignola/"
                f"{flags['project_name']}/"
            )
        flags["logs_dir"] = logs_dir
        # Add wandb notes if specified
        if args.wandb_notes:
            flags["wandb_notes"] = args.wandb_notes
        cmd = generate_base_command(experiment, flags=flags)
        command_list.append(f"PYTHONPATH={REPO_ROOT} {cmd}")

    # submit jobs - using exact working Hydra configuration
    hw_config = HARDWARE_CONFIGS.get(args.hardware, HARDWARE_CONFIGS["4090_rtx"])
    if args.duration is not None:
        duration = args.duration
    elif args.long_run:
        duration = "24:00:00"
    else:
        duration_hours = hw_config["timeout_min"] // 60
        duration_mins = hw_config["timeout_min"] % 60
        duration = f"{duration_hours}:{duration_mins:02d}:00"

    generate_run_commands(
        command_list,
        num_cpus=hw_config["cpus_per_task"],
        num_gpus=NUM_GPUS, #if hw_config["gpu_type"] is not None else 0,
        mode=args.mode,
        duration=duration,
        prompt=not args.dry_run,
        dry=args.dry_run,
        gpu_type=hw_config["gpu_type"],
        partition=args.partition,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="euler", help="how to launch the experiments"
    )
    parser.add_argument("--long_run", default=False, action="store_true")
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
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--particles", type=int, nargs="+", default=num_particles,
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(5)),
    )
    parser.add_argument(
        "--match_theory_config",
        action="store_true",
        help=(
            "Match the common D0, GP, iCEM, hard-constraint and five-trajectory "
            "evaluation settings used by theory_sweep_launcher.py."
        ),
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="4090_rtx",
        choices=["4090_rtx", "rtx_a6000", "cpu_only"],
        help="hardware configuration (similar to Hydra +hardware=4090_rtx)",
    )
    parser.add_argument(
        "--wandb_notes",
        type=str,
        default=None,
        help="wandb notes for grouping runs (e.g. Mar01-cartpole_sbsrl)",
    )

    args = parser.parse_args()
    main(args)
