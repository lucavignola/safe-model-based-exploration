# %% Setup (run once): imports, config, and plotting helpers
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn.objects as so
import wandb
from seaborn import axes_style
from tueplots import bundles, figsizes

warnings.filterwarnings("ignore")


# =========================
# User-facing configuration
# =========================
ENTITY = "lvignola-eth-z-rich"
PROJECT_NAME = "PendulumGP"

# Optional global pre-filter. Use a list to include multiple note groups.
WANDB_NOTES: str | list[str] | None = [
    "May6-sbsrl-pend-sparse2",
    #"Apr21-actsafe_exploit2",
]

# Additional W&B run filters (restored from previous plotting setup).
GLOBAL_WANDB_FILTERS: dict[str, Any] = {
    "config.num_offline_data": {"$in": [0]},
    "config.violation_eps": {"$in": [1]},
    "config.action_cost": {"$in": [0.01]}
}

# Number of episodes to plot from each run.
NUM_EPISODES = 7

# If True, runs missing any episode in [0, NUM_EPISODES-1] are discarded.
STRICT_EPISODES = False

# Metrics to read from W&B history and corresponding y-axis labels.
METRICS = [
    ("extrinsic_rewards", r"$\hat{J}_r(\pi_n)$"),
    ("constraint_cost", r"$\hat{J}_c(\pi_n)-d$"),
]

# Horizontal line on cost subplot (set to None to disable).
COST_LIMIT = None


@dataclass
class GroupSpec:
    name: str
    label: str
    color: str
    marker: str
    match: dict[str, Any]
    wandb_notes: str | list[str] | None = None
    where: Callable[[Any], bool] | None = None


#
# Define comparison groups here.
# - You can have multiple groups for the same algorithm (e.g., SBSRL with different params).
# - "match" keys can be dotted config paths, e.g. "training.safe" or flat keys like "violation_eps".
# - You can add a custom Python predicate with "where" for full if-style control.
#
# Example parameter-split groups (kept as commented examples):
# GroupSpec(
#     name="SBSRL eps=0.5",
#     label=r"\textsf{SBSRL} ($\epsilon_v=0.5$)",
#     color="#5F4690",
#     marker="o",
#     match={"alg_name": "SBSRL", "violation_eps": 0.5},
# ),
# GroupSpec(
#     name="SBSRL eps=1.0",
#     label=r"\textsf{SBSRL} ($\epsilon_v=1.0$)",
#     color="#1D6996",
#     marker="x",
#     match={"alg_name": "SBSRL", "violation_eps": 1.0},
# ),
# GroupSpec(
#     name="ActSafe eps=1.0",
#     label=r"\textsf{ActSafe} ($\epsilon_v=1.0$)",
#     color="#38A6A5",
#     marker="^",
#     match={"alg_name": "ActSafe", "violation_eps": 1.0},
# ),

GROUPS: list[GroupSpec] = [
    GroupSpec(
        name=r"SBSRL (d_\sigma^n high=20)",
        label=r"\textsf{SBSRL} ($d_\sigma^n=20$)",
        color="#5F4690",
        marker="o",
        match={"alg_name": "SBSRL", "num_particles": 30, "lambda_sigma": 100, "uncertainty_eps": 0, "uncertainty_decay_factor": 1},
        wandb_notes="May6-sbsrl-pend-sparse2",
    ),
    GroupSpec(
        name=r"SBSRL (d_\sigma^n=10)",
        label=r"\textsf{SBSRL} ($d_\sigma^n=30$)",
        color="#1D6996",
        marker="x",
        match={"alg_name": "SBSRL", "num_particles": 30, "lambda_sigma": 100, "uncertainty_eps": 30, "uncertainty_decay_factor": 1},
        wandb_notes="May6-sbsrl-pend-sparse2",
    ),
    GroupSpec(
        name=r"SBSRL (d_\sigma^n=10)",
        label=r"\textsf{SBSRL} ($d_\sigma^n=50$)",
        color="#1D6996",
        marker="x",
        match={"alg_name": "SBSRL", "num_particles": 30, "lambda_sigma": 100, "uncertainty_eps": 50, "uncertainty_decay_factor": 1},
        wandb_notes="May6-sbsrl-pend-sparse2",
    ),

]


def get_config_value(config: dict[str, Any], key_path: str) -> Any:
    if key_path in config:
        return config[key_path]

    current: Any = config
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def values_match(actual: Any, expected: Any, atol: float = 1e-9) -> bool:
    if isinstance(expected, (list, tuple, set)):
        return any(values_match(actual, item, atol=atol) for item in expected)

    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return abs(float(actual) - float(expected)) <= atol
        except Exception:
            return False

    return actual == expected


def notes_match(run, wanted: str | list[str] | None) -> bool:
    if wanted is None:
        return True

    if isinstance(wanted, (list, tuple, set)):
        return any(notes_match(run, note) for note in wanted)

    run_notes = run.notes or ""
    run_tags = run.tags or []
    return (run_notes == wanted) or (wanted in run_tags)


def run_matches_group(run, group: GroupSpec) -> bool:
    if run.config is None:
        return False

    if not notes_match(run, group.wandb_notes):
        return False

    for key_path, expected in group.match.items():
        actual = get_config_value(run.config, key_path)
        if not values_match(actual, expected):
            return False

    if group.where is not None and not group.where(run):
        return False

    return True


def collect_candidate_runs(entity: str, project_name: str, wandb_notes: str | list[str] | None):
    api = wandb.Api()
    filters: dict[str, Any] = dict(GLOBAL_WANDB_FILTERS)
    if wandb_notes is not None:
        notes = list(wandb_notes) if isinstance(wandb_notes, (list, tuple, set)) else [wandb_notes]
        filters["$or"] = (
            [{"notes": note} for note in notes]
            + [{"tags": {"$in": [note]}} for note in notes]
        )

    if filters:
        return list(api.runs(f"{entity}/{project_name}", filters=filters))
    return list(api.runs(f"{entity}/{project_name}"))


def load_metrics_by_episode(run, metric_names: list[str], num_episodes: int, strict_episodes: bool) -> dict[int, dict[str, float]]:
    rows = list(run.scan_history(keys=["_step", "episode_idx", *metric_names]))
    if not rows:
        raise ValueError(f"No history rows in run {run.id}")

    history_df = pd.DataFrame(rows)
    if "episode_idx" not in history_df.columns:
        raise ValueError(f"No episode_idx in run {run.id}")

    missing_cols = [m for m in metric_names if m not in history_df.columns]
    if missing_cols:
        raise ValueError(f"Missing metrics {missing_cols} in run {run.id}")

    metric_mask = history_df[metric_names].notna().any(axis=1)
    metric_rows = history_df[metric_mask].copy()
    if metric_rows.empty:
        raise ValueError(f"No valid metric rows for run {run.id}")

    episode_values: dict[int, dict[str, float]] = {}
    for episode_idx, group in metric_rows.groupby("episode_idx", sort=True):
        if pd.isna(episode_idx):
            continue
        ep = int(episode_idx)
        if ep < 0 or ep >= num_episodes:
            continue

        episode_values[ep] = {}
        for metric in metric_names:
            valid_rows = group[group[metric].notna()]
            if len(valid_rows) > 0:
                episode_values[ep][metric] = float(valid_rows[metric].iloc[-1])

    if strict_episodes:
        missing_eps = [ep for ep in range(num_episodes) if ep not in episode_values]
        if missing_eps:
            raise ValueError(f"Missing episodes {missing_eps} in run {run.id}")

    return episode_values


def build_dataframe(runs, groups: list[GroupSpec], metric_names: list[str], num_episodes: int, strict_episodes: bool):
    records = []
    loaded_counts = {group.name: 0 for group in groups}

    for run in runs:
        if not notes_match(run, WANDB_NOTES):
            continue

        matched_group = None
        for group in groups:
            if run_matches_group(run, group):
                matched_group = group
                break

        if matched_group is None:
            continue

        seed = get_config_value(run.config or {}, "seed")
        if seed is None:
            seed = run.id

        try:
            episode_values = load_metrics_by_episode(
                run=run,
                metric_names=metric_names,
                num_episodes=num_episodes,
                strict_episodes=strict_episodes,
            )
        except Exception as exc:
            print(f"Skipping run {run.id} ({matched_group.name}): {exc}")
            continue

        loaded_counts[matched_group.name] += 1

        for episode_idx in sorted(episode_values.keys()):
            row = {
                "step": episode_idx,
                "seed": seed,
                "method": matched_group.name,
            }
            for metric in metric_names:
                if metric in episode_values[episode_idx]:
                    row[metric] = episode_values[episode_idx][metric]
            records.append(row)

    if not records:
        raise ValueError("No runs matched the configured groups and metric requirements.")

    data = pd.DataFrame(records)
    return data, loaded_counts


def configure_plot_theme():
    theme = bundles.neurips2024()
    so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
    plt.rcParams.update(bundles.neurips2024())
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})


def make_plot(data: pd.DataFrame, groups: list[GroupSpec], metrics: list[tuple[str, str]], cost_limit: float | None):
    present_methods = [group.name for group in groups if group.name in set(data["method"].unique())]
    if not present_methods:
        raise ValueError("No configured groups are present in the assembled dataframe.")

    colors = {group.name: group.color for group in groups}
    markers = {group.name: group.marker for group in groups}

    configure_plot_theme()
    plt.rcParams.update(
        figsizes.neurips2024(
            nrows=len(metrics),
            ncols=1,
            rel_width=0.40,
            height_to_width_ratio=0.5,
        )
    )
    fig, axes = plt.subplots(len(metrics), 1, sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for metric_idx, (metric_name, y_label) in enumerate(metrics):
        so.Plot(
            data,
            x="step",
            marker="method",
            color="method",
            y=metric_name,
        ).add(
            so.Line(linewidth=1.0, pointsize=2.5, edgewidth=0.1),
            so.Agg("mean"),
            legend=False,
        ).add(
            so.Band(alpha=0.15),
            so.Est("mean", errorbar=("pi", 95)),
            legend=False,
        ).scale(
            color=so.Nominal(
                values=[colors[m] for m in present_methods],
                order=present_methods,
            ),
            marker=so.Nominal(
                values=[markers[m] for m in present_methods],
                order=present_methods,
            ),
        ).label(
            x="",
            y=lambda _: y_label,
        ).theme(
            axes_style("ticks")
        ).on(
            axes[metric_idx]
        ).plot()

    baseline = data.groupby(["seed", "method"])[metrics[0][0]].first().mean()
    for idx, ax in enumerate(axes):
        ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
        if idx == 0:
            ax.axhline(
                baseline,
                linestyle="--",
                color="#666666",
                linewidth=1.0,
            )
            # Get normalized y position for the baseline
            y_min, y_max = ax.get_ylim()
            y_norm = (baseline - y_min) / (y_max - y_min)
            ax.text(
                -0.05,
                y_norm,
                r"$\pi_0$",
                transform=ax.transAxes,
                fontsize=7,
                va="center",
                ha="right",
                color="#666666",
                clip_on=False,
            )
        if idx == len(axes) - 1 and cost_limit is not None:
            ax.axhline(y=cost_limit, color="black", linestyle=(0, (1, 1)), linewidth=1.25)

        if idx < len(axes) - 1:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax.set_xlabel("")

        for spine in ax.spines.values():
            spine.set_linewidth(1.25)

    axes[-1].set_xlabel(r"Episode $n$")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors[m],
            marker=markers[m],
            linewidth=1.5,
            markersize=5,
        )
        for m in present_methods
    ]
    legend_texts = [group.label for group in groups if group.name in present_methods]

    fig.legend(
        legend_handles,
        legend_texts,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.5,
        handlelength=1.4,
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    #fig.suptitle(r"\textsc{PendulumSwingup}", y=1.1)
    return fig


def make_reward_cost_bar_plot(
    data: pd.DataFrame,
    groups: list[GroupSpec],
    metrics: list[tuple[str, str]],
):
    """Two-panel bar plot: mean max reward (normalized) and cumulative cost."""
    present_methods = [group.name for group in groups if group.name in set(data["method"].unique())]
    if not present_methods:
        raise ValueError("No configured groups are present in the assembled dataframe.")

    colors = {group.name: group.color for group in groups}

    reward_metric = metrics[0][0]
    cost_metric = metrics[1][0]

    per_run = (
        data.groupby(["seed", "method"], as_index=False)
        .agg(
            cumulative_reward=(reward_metric, "sum"),
            max_cost=(cost_metric, "max"),
        )
    )

    global_min_reward = float(per_run["cumulative_reward"].min())
    global_max_reward = float(per_run["cumulative_reward"].max())
    if global_max_reward > global_min_reward:
        per_run["cumulative_reward_normalized"] = (
            per_run["cumulative_reward"] - global_min_reward
        ) / (global_max_reward - global_min_reward)
    else:
        per_run["cumulative_reward_normalized"] = 0.0

    grouped = per_run.groupby("method")
    reward_mean = grouped["cumulative_reward_normalized"].mean().reindex(present_methods)
    reward_std = grouped["cumulative_reward_normalized"].std().fillna(0.0).reindex(present_methods)
    cost_mean = grouped["max_cost"].mean().reindex(present_methods)
    cost_std = grouped["max_cost"].std().fillna(0.0).reindex(present_methods)

    configure_plot_theme()
    plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=2, rel_width=0.48))
    fig, axes = plt.subplots(1, 2)

    bar_x = list(range(len(present_methods)))
    bar_colors = [colors[m] for m in present_methods]

    summary_frames = [
        pd.DataFrame(
            {
                "method": present_methods,
                "x_pos": bar_x,
                "mean": reward_mean.values,
                "std": reward_std.values,
            }
        ),
        pd.DataFrame(
            {
                "method": present_methods,
                "x_pos": bar_x,
                "mean": cost_mean.values,
                "std": cost_std.values,
            }
        ),
    ]

    max_cost_with_error = float((cost_mean + cost_std).max())
    cost_upper = 1.1 * max_cost_with_error if max_cost_with_error > 0 else 1.0

    plot_specs = [
        (axes[0], summary_frames[0], r"Cumulative Reward", r"$\sum_n \hat{J}_r(\pi_n)$", (0.0, 1.1)),
        (axes[1], summary_frames[1], r"Max Cost", r"$\max_n \hat{J}_c(\pi_n)-d$", (0.0, cost_upper)),
    ]

    for ax, summary_df, title, ylabel, ylim in plot_specs:
        (
            so.Plot(summary_df, x="x_pos", y="mean", color="method")
            .add(so.Bar(), legend=False)
            .scale(color=so.Nominal(values=bar_colors, order=present_methods))
            .label(x="", y=lambda _: ylabel)
            .theme(axes_style("ticks"))
            .on(ax)
            .plot()
        )

        ax.errorbar(
            summary_df["x_pos"],
            summary_df["mean"],
            yerr=summary_df["std"],
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=6,
        )
        ax.set_xticks(bar_x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        #ax.set_title(title, pad=2.0)
        ax.set_axisbelow(True) # keep grid behind the plot
        if ylim is not None:
            ax.set_ylim(*ylim)
        if ylabel.endswith("(normalized)"):
            ax.set_yticks([0.0, 1.0])

    for ax in axes:
        ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.25)

    legend_handles = [
        Line2D([0], [0], color=colors[m], marker="s", linestyle="None", markersize=8)
        for m in present_methods
    ]
    legend_labels = [group.label for group in groups if group.name in present_methods]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=max(2, len(legend_labels)),
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.25,
        handlelength=1.2,
    )

    #fig.suptitle(r"\textsc{PendulumSwingup}", y=1.145)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.30, top=0.80, wspace=0.28)
    return fig


def sanitize_filename(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(c if c in allowed else "_" for c in name)


# %% Load data once (run this cell when data/config changes)
if "METRICS" not in globals():
    raise RuntimeError("Setup is not loaded. Run the first '# %% Setup' cell first.")

metric_names = [metric for metric, _ in METRICS]

runs = collect_candidate_runs(
    entity=ENTITY,
    project_name=PROJECT_NAME,
    wandb_notes=WANDB_NOTES,
)
print(f"Found {len(runs)} candidate runs in {ENTITY}/{PROJECT_NAME}.")

data, loaded_counts = build_dataframe(
    runs=runs,
    groups=GROUPS,
    metric_names=metric_names,
    num_episodes=NUM_EPISODES,
    strict_episodes=STRICT_EPISODES,
)

print("Loaded runs per group:")
for group in GROUPS:
    print(f"  {group.name}: {loaded_counts[group.name]}")


# %% Plot and save (rerun this cell as often as needed)
def _refresh_plot_definitions_from_disk() -> None:
    module_path = (
        Path(__file__)
        if "__file__" in globals()
        else Path("/Users/lucav/Documents/safe-model-based-exploration/analysis/plot_episode_rewards_pendulum_exploration.py")
    )
    source = module_path.read_text()
    setup_source, _, _ = source.partition("\n# %% Load data once")
    if not setup_source.strip():
        raise RuntimeError(f"Could not refresh setup definitions from {module_path}")

    exec(compile(setup_source, str(module_path), "exec"), globals())


_refresh_plot_definitions_from_disk()

if "data" not in globals() or "loaded_counts" not in globals():
    raise RuntimeError("Data is not loaded. Run the '# %% Load data once' cell first.")

fig = make_plot(data=data, groups=GROUPS, metrics=METRICS, cost_limit=COST_LIMIT)
fig_reward_cost_legend = make_reward_cost_bar_plot(data=data, groups=GROUPS, metrics=METRICS)

if WANDB_NOTES is None:
    notes_stub = "all"
elif isinstance(WANDB_NOTES, (list, tuple, set)):
    notes_stub = "multi_notes"
else:
    notes_stub = sanitize_filename(WANDB_NOTES)

output_stub = f"pend_exp/pend_curves_{sanitize_filename(PROJECT_NAME)}_{notes_stub}"
Path(output_stub).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{output_stub}.pdf", bbox_inches="tight", pad_inches=0.12)
print(f"Saved line plot to {output_stub}.pdf")

bar_output_stub_legend = f"pend_exp/pend_barplot_legend_{sanitize_filename(PROJECT_NAME)}_{notes_stub}"
fig_reward_cost_legend.savefig(f"{bar_output_stub_legend}.pdf", bbox_inches="tight", pad_inches=0.12)
print(f"Saved reward/cost bar plot (legend version) to {bar_output_stub_legend}.pdf")

plt.show()
