import experiment
from smbrl.utils.experiment_utils import (
    generate_run_commands,
    generate_base_command,
    dict_permutations,
)
import argparse

PROJECT_NAME = "ExplorationCartPoleGP"
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
    "entity": [ENTITY],
    "num_gpus": [NUM_GPUS],
    "lambda_constraint": [0],
    "use_precomputed_kernel_params": [
        0,
    ],
    "use_function_norms": [0],
    "num_offline_data": [0],
    "max_position": [1.5],
    "num_samples": [1000],
    "alpha": [0.8],
    "init_std": [1.0],
    "icem_horizon": [50],
    "num_elites": [20],
    "num_steps": [5],
    "exponent": [1.0],
    "violation_eps":  [0.6],
    "num_traj": [0],  # 0=uniform grid sampling, >0=trajectory-based sampling
    "process_noise_scale": [1e-3],
    "model_noise_scale": [1e-3],
    "reward_source": ["sparse"],
    "sparse_task": [True],
    "sparse_reward_lower_bound": [0.5],
    "action_cost": [0.0],
    "use_mean_dynamics": [True],
    "aleatoric_noise_in_prediction": [True],
}
num_particles = [10]
_applicable_configs_actsafe = _applicable_configs | {
    "alg_name": ["ActSafe"],
    "use_optimism": [1],
    "use_pessimism": [1],
    "num_particles": num_particles,
    "beta": [3.0],
    "prior_knowledge": ["none", "cartpole"],
    "actsafe_index": [-1],
}

_applicable_configs_actsafe_no_pessimism = _applicable_configs | {
    "alg_name": ["ActSafe"],
    "use_optimism": [0],
    "use_pessimism": [0],
    "num_particles": [1],
    "beta": [3.0],
    "prior_knowledge": ["none", "cartpole"],
}

_applicable_configs_opax = _applicable_configs | {
    "alg_name": ["OPAX"],
    "use_optimism": [1],
    "use_pessimism": [1],
    "num_particles": num_particles,
    "beta": [3.0],
    "prior_knowledge": ["none", "cartpole"],
}

_applicable_configs_sbsrl = _applicable_configs | {
    "alg_name": ["SBSRL"],
    "use_optimism": [0],
    "use_pessimism": [0],
    "num_particles": num_particles,
    "beta": [3.0],
    "prior_knowledge": ["none", "cartpole"],
    "lambda_sigma": [0],
    "uncertainty_eps": [0],
    "uncertainty_decay_factor": [1],
    "uncertainty_decay_mode": ["linear"],
    "uncertainty_constraint_threshold": [0],
    "default_task_index": [0],
}

_applicable_configs_safehucrl = _applicable_configs | {
    "alg_name": ["SafeHUCRL"],
    "use_optimism": [1],
    "use_pessimism": [1],
    "num_particles": num_particles,
    "beta": [3.0],
    "prior_knowledge": ["none", "cartpole"],
}

_applicable_configs_hucrl = _applicable_configs | {
    "alg_name": ["HUCRL"],
    "use_optimism": [0],
    "use_pessimism": [0],
    "num_particles": num_particles,
    "beta": [3.0],
    "prior_knowledge": ["none", "cartpole"],
}

_applicable_configs_ground_truth = _applicable_configs | {
    "alg_name": ["GroundTruth"],
    "use_optimism": [0],
    "use_pessimism": [0],
    "num_particles": [1],
    "beta": [0.0],
    "prior_knowledge": ["none"],
}

all_flags_combinations = (
    dict_permutations(_applicable_configs_sbsrl)
    + dict_permutations(_applicable_configs_hucrl)
    + dict_permutations(_applicable_configs_ground_truth)
)


def main(args):
    command_list = []

    logs_dir = "../"
    if args.mode == "euler":
        logs_dir = "/cluster/scratch/"
        logs_dir += "lvignola" + "/" + PROJECT_NAME + "/"

    for flags in all_flags_combinations:
        flags["logs_dir"] = logs_dir
        # Add wandb notes if specified
        if args.wandb_notes:
            flags["wandb_notes"] = args.wandb_notes
        cmd = generate_base_command(experiment, flags=flags)
        command_list.append(cmd)

    # submit jobs - using exact working Hydra configuration
    hw_config = HARDWARE_CONFIGS.get(args.hardware, HARDWARE_CONFIGS["4090_rtx"])
    duration_hours = hw_config["timeout_min"] // 60 if not args.long_run else 23
    duration_mins = hw_config["timeout_min"] % 60 if not args.long_run else 59

    generate_run_commands(
        command_list,
        num_cpus=hw_config["cpus_per_task"],
        num_gpus=NUM_GPUS, #if hw_config["gpu_type"] is not None else 0,
        mode=args.mode,
        duration=f"{duration_hours}:{duration_mins:02d}:00",
        prompt=True,
        gpu_type=hw_config["gpu_type"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="euler", help="how to launch the experiments"
    )
    parser.add_argument("--long_run", default=False, action="store_true")
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
