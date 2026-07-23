import copy
import os.path
import pickle
from typing import Tuple, NamedTuple, List

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import wandb
from brax.envs import Env as BraxEnv
from brax.envs import State
from brax.training.types import Metrics
from bsm.statistical_model import StatisticalModel, GPStatisticalModel
from bsm.utils.normalization import Data
from bsm.utils.type_aliases import ModelState
from distrax import Distribution, Normal
from flax import struct
from jaxtyping import Key, Array, PyTree, Float
from smbrl.mbpo_stubs import Reward, RewardParams
from optax import Schedule, constant_schedule

from smbrl.model_based_rl.active_exploration_system import ExplorationSystem, ExplorationReward, ExplorationDynamics
from smbrl.dynamics_models.gp_sampling import (
    RFFPriorState,
    RFFPosteriorState,
    append_rff_prior_confidence,
    sample_rff_prior,
    sample_rff_posterior,
)
from smbrl.optimizer.icem import iCemParams, iCemTO, AbstractCost
# from smbrl.optimizer.ipopt_optimizer import IPOPTOptimizer, IPOPTParams
from smbrl.utils.tolerance_reward import ToleranceReward
from smbrl.utils.utils import create_folder, ExplorationTrajectory


class Task(NamedTuple):
    reward: Reward
    name: str
    env: BraxEnv


class SafeModelBasedAgent:
    def __init__(self,
                 env: BraxEnv,
                 model: StatisticalModel,
                 episode_length: int,
                 action_repeat: int,
                 cost_fn: AbstractCost,
                 test_tasks: List[Task],
                 predict_difference: bool = True,
                 num_training_steps: Schedule = constant_schedule(1000),
                 icem_horizon: int = 20,
                 icem_params: iCemParams = iCemParams(),
                 ipopt_params: iCemParams = iCemParams(),
                 # ipopt_params: IPOPTParams = IPOPTParams(),
                 saving_frequency: int = 5,
                 log_to_wandb: bool = False,
                 train_task_index: int = -1,
                 use_optimism: bool = True,
                 use_pessimism: bool = True,
                 optimizer: str = 'icem',  # can be 'icem' or 'ipopt'
                 gp_sampling_method: str = 'marginal',
                 gp_path_source: str = 'posterior',
                 num_rff_features: int = 512,
                 rff_path_scale: float | None = None,
                 gp_sample_truncation: str = 'none',
                 aleatoric_noise_in_prediction: bool = True,
                 constraint_failure_mode: str = 'recovery',
                 ):
        assert train_task_index >= -1
        assert train_task_index <= len(test_tasks)
        self.env = env
        if isinstance(model, GPStatisticalModel):
            jax.config.update("jax_enable_x64", True)
        self.model = model
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.cost_fn = cost_fn
        self.cost_fn_env = copy.deepcopy(cost_fn)
        self.cost_fn_env.horizon = self.episode_length
        if hasattr(self.cost_fn_env, 'violation_eps'):
            self.cost_fn_env.violation_eps = 0
        self.test_tasks = test_tasks

        self.use_optimism = use_optimism
        self.use_pessimism = use_pessimism

        self.train_task_index = train_task_index

        self.predict_difference = predict_difference
        self.num_training_steps = num_training_steps
        self.icem_horizon = icem_horizon
        self.icem_params = icem_params
        self.saving_frequency = saving_frequency
        self.log_to_wandb = log_to_wandb
        self.optimizer = optimizer
        self.ipopt_params = ipopt_params
        if gp_sampling_method not in {'marginal', 'rff'}:
            raise ValueError(
                "gp_sampling_method must be one of {'marginal', 'rff'}, "
                f"got {gp_sampling_method!r}."
            )
        if num_rff_features < 1:
            raise ValueError(
                f'num_rff_features must be positive, got {num_rff_features}.'
            )
        self.gp_sampling_method = gp_sampling_method
        if gp_path_source not in ExplorationDynamics.GP_PATH_SOURCES:
            raise ValueError(
                "gp_path_source must be one of "
                f"{sorted(ExplorationDynamics.GP_PATH_SOURCES)}, "
                f"got {gp_path_source!r}."
            )
        if gp_path_source == 'prior' and gp_sampling_method != 'rff':
            raise ValueError(
                "Whole-run prior paths require gp_sampling_method='rff'."
            )
        self.gp_path_source = gp_path_source
        self.num_rff_features = num_rff_features
        if rff_path_scale is not None and rff_path_scale < 0:
            raise ValueError(
                f'rff_path_scale must be non-negative, got {rff_path_scale}.'
            )
        if (
                gp_path_source == 'prior'
                and rff_path_scale is not None
                and rff_path_scale != 1.0
        ):
            raise ValueError(
                "Whole-run prior RFF paths are uninflated prior draws and require "
                "rff_path_scale=1."
            )
        if gp_path_source == 'prior' and rff_path_scale is None:
            rff_path_scale = 1.0
        self.rff_path_scale = rff_path_scale
        if gp_sample_truncation not in ExplorationDynamics.GP_SAMPLE_TRUNCATION_MODES:
            raise ValueError(
                "gp_sample_truncation must be one of "
                f"{sorted(ExplorationDynamics.GP_SAMPLE_TRUNCATION_MODES)}, "
                f"got {gp_sample_truncation!r}."
            )
        self.gp_sample_truncation = gp_sample_truncation
        self.aleatoric_noise_in_prediction = aleatoric_noise_in_prediction
        if (
                gp_sample_truncation == 'recursive'
                and gp_path_source != 'prior'
        ):
            raise ValueError(
                "Recursive confidence truncation requires "
                "gp_path_source='prior'."
            )
        self._fixed_prior_path_state: RFFPriorState | None = None
        if constraint_failure_mode not in {'recovery', 'raise'}:
            raise ValueError(
                "constraint_failure_mode must be one of "
                "{'recovery', 'raise'}, "
                f"got {constraint_failure_mode!r}."
            )
        self.constraint_failure_mode = constraint_failure_mode
        self.latest_planning_feasible_fraction = 0.0
        self.latest_planning_any_feasible_fraction = 0.0
        self.latest_planning_solver_failure_fraction = 0.0
        self.latest_planning_max_selected_cost = 0.0

    def get_planning_dynamics(self,
                              use_log: bool = True,
                              scale_with_aleatoric_std: bool = True) -> ExplorationDynamics:
        """Constructs the learned dynamics used by every planning rollout."""

        return ExplorationDynamics(
            x_dim=self.env.observation_size,
            u_dim=self.env.action_size,
            model=self.model,
            use_log=use_log,
            scale_with_aleatoric_std=scale_with_aleatoric_std,
            predict_difference=self.predict_difference,
            gp_sampling_method=self.gp_sampling_method,
            gp_path_source=self.gp_path_source,
            rff_path_scale=self.rff_path_scale,
            gp_sample_truncation=self.gp_sample_truncation,
            aleatoric_noise_in_prediction=(
                self.aleatoric_noise_in_prediction
            ),
        )

    def sample_episode_posterior_paths(
            self,
            model_state: ModelState,
            episode_key: Key[Array, '2'],
    ) -> RFFPosteriorState | RFFPriorState | None:
        """Returns the path bank used throughout one episode.

        Posterior paths are redrawn after each GP update.  Prior paths are
        created once by ``_initialize_fixed_prior_paths`` and reused for the
        entire online run.
        """

        if self.gp_sampling_method == 'marginal':
            return None
        if self.gp_path_source == 'prior':
            if self._fixed_prior_path_state is None:
                raise RuntimeError(
                    "Fixed prior paths must be initialized before planning."
                )
            return self._fixed_prior_path_state
        # A fixed tag keeps path sampling separate from control/environment RNGs
        # and makes changing the sampling method or M leave those streams intact.
        path_key = jr.fold_in(episode_key, 0x524646)
        return sample_rff_posterior(
            model=self.model,
            model_state=model_state,
            key=path_key,
            num_paths=self.icem_params.num_particles,
            num_features=self.num_rff_features,
        )

    def _initial_prior_beta(self) -> chex.Array:
        """Returns the output-wise ``B`` used for the episode-zero tube."""

        if hasattr(self.model, "theorem_f_norm_bound"):
            return jnp.asarray(self.model.theorem_f_norm_bound)
        if hasattr(self.model, "f_norm_bound"):
            return jnp.asarray(self.model.f_norm_bound)
        raise TypeError(
            "Recursive prior truncation requires a GP model exposing "
            "an RKHS/function norm bound as f_norm_bound."
        )

    def _empty_gp_history(self, model_state: ModelState) -> ModelState:
        """Removes BSM's synthetic zero datum for a true no-data prior."""

        gp_state = model_state.model_state
        empty_history = Data(
            inputs=jnp.zeros(
                (0, self.model.input_dim),
                dtype=gp_state.history.inputs.dtype,
            ),
            outputs=jnp.zeros(
                (0, self.model.output_dim),
                dtype=gp_state.history.outputs.dtype,
            ),
        )
        replacements = {"history": empty_history}
        if hasattr(gp_state, "alphas"):
            replacements["alphas"] = jnp.zeros(
                (self.model.output_dim, 0),
                dtype=gp_state.history.outputs.dtype,
            )
        return model_state.replace(
            model_state=gp_state.replace(**replacements)
        )

    def _initialize_fixed_prior_paths(
            self,
            model_state: ModelState,
            episode_key: Key[Array, '2'],
            *,
            condition_on_initial_data: bool,
    ) -> ModelState:
        """Samples and freezes the online prior before the first rollout."""

        if not isinstance(self.model, GPStatisticalModel):
            raise TypeError(
                "Whole-run RFF prior paths require GPStatisticalModel."
            )
        if not condition_on_initial_data:
            model_state = self._empty_gp_history(model_state)

        path_key = jr.fold_in(episode_key, 0x5052494F)
        self._fixed_prior_path_state = sample_rff_prior(
            model=self.model,
            model_state=model_state,
            key=path_key,
            num_paths=self.icem_params.num_particles,
            num_features=self.num_rff_features,
            initial_beta=self._initial_prior_beta(),
            condition_on_initial_data=condition_on_initial_data,
        )

        # A fixed function-space prior requires a fixed kernel and coordinate
        # system.  Future updates only condition this same GP on more data.
        self.model.fixed_kernel_params = True
        self.model.normalization_stats = model_state.model_state.data_stats
        return model_state

    def _append_fixed_prior_confidence(
            self,
            model_state: ModelState,
    ) -> None:
        """Adds the latest post-online-data confidence tube exactly once."""

        if self._fixed_prior_path_state is None:
            raise RuntimeError("Fixed prior paths have not been initialized.")
        self._fixed_prior_path_state = append_rff_prior_confidence(
            self._fixed_prior_path_state,
            model_state,
        )

    def train_dynamics_model(self,
                             model_state: ModelState,
                             data: Data,
                             episode_idx: int) -> ModelState:
        model_state = self.model.update(data=data,
                                        stats_model_state=model_state)
        return model_state

    def _handle_constraint_solver_failure(
            self,
            optimizer_state,
            *,
            step: int,
            context: str,
    ) -> None:
        """Prevents an infeasible recovery sequence from masquerading as safe."""

        failed = bool(
            jax.device_get(optimizer_state.constraint_solver_failed)
        )
        if not failed or self.constraint_failure_mode == 'recovery':
            return
        raise RuntimeError(
            "Hard-constrained iCEM found no feasible candidate at "
            f"{context} step {step}. The optimizer's recovery sequence is "
            "infeasible and was not executed. Increase the planning budget "
            "or provide a separately verified safe fallback controller."
        )

    def test_a_task(self,
                    model_state: ModelState,
                    key: Key[Array, '2'],
                    task: Task,
                    posterior_path_state:
                    RFFPosteriorState | RFFPriorState | None = None,
                    ) -> Tuple[State, Float[Array, '... action_dim'], Float[Array, 'episode_length 1'], Metrics]:
        if posterior_path_state is None:
            posterior_path_state = self.sample_episode_posterior_paths(
                model_state=model_state,
                episode_key=key,
            )
        exploration_dynamics = self.get_planning_dynamics()
        learned_system = ExplorationSystem(
            dynamics=exploration_dynamics,
            reward=task.reward,
        )
        key, subkey = jr.split(key)

        if self.optimizer == 'icem':
            optimizer = iCemTO(
                horizon=self.icem_horizon,
                action_dim=self.env.action_size,
                key=subkey,
                opt_params=self.icem_params,
                system=learned_system,
                cost_fn=self.cost_fn,
                use_optimism=self.use_optimism,
                use_pessimism=self.use_pessimism,
            )
        # elif self.optimizer == 'ipopt':
        #     optimizer = IPOPTOptimizer(
        #         horizon=self.icem_horizon,
        #         action_dim=self.env.action_size,
        #         key=subkey,
        #         opt_params=self.ipopt_params,
        #         system=learned_system,
        #         cost_fn=self.cost_fn,
        #         use_optimism=self.use_optimism,
        #         use_pessimism=self.use_pessimism,
        #     )

        key, subkey = jr.split(key)
        optimizer_state = optimizer.init(key=subkey)

        dynamics_params = optimizer_state.system_params.dynamics_params.replace(
            model_state=model_state,
            posterior_path_state=posterior_path_state,
        )
        system_params = optimizer_state.system_params.replace(dynamics_params=dynamics_params)
        optimizer_state = optimizer_state.replace(system_params=system_params)

        env_state = task.env.reset(rng=key)

        collected_states = [env_state]
        actions = []

        for i in range(self.episode_length):
            action, optimizer_state = optimizer.act(env_state.obs, optimizer_state)
            self._handle_constraint_solver_failure(
                optimizer_state,
                step=i,
                context=f"evaluation task {task.name!r}",
            )
            for _ in range(self.action_repeat):
                env_state = self.env.step(env_state, action)
            collected_states.append(env_state)
            actions.append(action)

        collected_states = jt.map(lambda *xs: jnp.stack(xs), *collected_states)
        actions = jt.map(lambda *xs: jnp.stack(xs), *actions)
        # get task reward
        state = collected_states.obs[:-1]
        next_state = collected_states.obs[1:]
        reward_params = system_params.reward_params
        rewards_dist, _ = jax.vmap(task.reward, in_axes=(0, 0, None, 0))(state, actions, reward_params, next_state)
        rewards = rewards_dist.mean()
        costs = self.cost_fn_env(state, actions)
        metrics = {f'total_reward_{task.name}': jnp.sum(rewards).item(), f'cost_{task.name}': costs.item()}
        return collected_states, actions, rewards, metrics

    def get_train_rewards(self) -> Reward:
        if self.train_task_index == -1:
            exploration_reward = ExplorationReward(x_dim=self.env.observation_size,
                                                   u_dim=self.env.action_size, )
            return exploration_reward
        else:
            return self.test_tasks[self.train_task_index].reward

    def get_episode_wandb_metrics(self, episode_idx: int) -> dict:
        return {}

    def on_episode_end(self, episode_idx: int) -> None:
        return None

    def on_exploration_rollout_end(self,
                                   episode_idx: int,
                                   intrinsic_rewards: chex.Array,
                                   extrinsic_rewards: chex.Array) -> None:
        return None

    def get_train_env_state(self, rng: jax.Array) -> State:
        if self.train_task_index == -1:
            return self.env.reset(rng=rng) #TODO: what does this return?
        else:
            env = self.test_tasks[self.train_task_index].env
            return env.reset(rng=rng)

    def simulate_on_true_env(self,
                             model_state: ModelState,
                             key: Key[Array, '2'],
                             posterior_path_state:
                             RFFPosteriorState | RFFPriorState | None = None,
                             ) -> Tuple[
        PyTree[Array, 'episode_length ...'], Float[Array, 'episode_length action_dim'], Float[
            Array, 'episode_length 1'], Float[
            Array, 'episode_length 1'], Float[Array, '1']]:
        reward = self.get_train_rewards()

        if posterior_path_state is None:
            posterior_path_state = self.sample_episode_posterior_paths(
                model_state=model_state,
                episode_key=key,
            )
        exploration_dynamics = self.get_planning_dynamics()
        learned_system = ExplorationSystem(
            dynamics=exploration_dynamics,
            reward=reward,
        )
        key, subkey = jr.split(key)

        if self.optimizer == 'icem':
            optimizer = iCemTO(
                horizon=self.icem_horizon,
                action_dim=self.env.action_size,
                key=subkey,
                opt_params=self.icem_params,
                system=learned_system,
                cost_fn=self.cost_fn,
                use_optimism=self.use_optimism,
                use_pessimism=self.use_pessimism,
            )
        # elif self.optimizer == 'ipopt':
        #     optimizer = IPOPTOptimizer(
        #         horizon=self.icem_horizon,
        #         action_dim=self.env.action_size,
        #         key=subkey,
        #         opt_params=self.ipopt_params,
        #         system=learned_system,
        #         cost_fn=self.cost_fn,
        #         use_optimism=self.use_optimism,
        #         use_pessimism=self.use_pessimism,
        #     )

        key, subkey = jr.split(key)
        optimizer_state = optimizer.init(key=subkey)

        dynamics_params = optimizer_state.system_params.dynamics_params.replace(
            model_state=model_state,
            posterior_path_state=posterior_path_state,
        )
        system_params = optimizer_state.system_params.replace(dynamics_params=dynamics_params)
        optimizer_state = optimizer_state.replace(system_params=system_params)

        env_state = self.get_train_env_state(rng=key)

        collected_states = [env_state]
        actions = []
        intrinsic_rewards = []
        extrinsic_rewards = []
        planning_costs = []
        planning_feasible = []
        planning_any_feasible = []
        planning_solver_failed = []
        # TODO: Should implement treatment of done flags
        for i in range(self.episode_length):
            action, optimizer_state = optimizer.act(env_state.obs, optimizer_state)
            self._handle_constraint_solver_failure(
                optimizer_state,
                step=i,
                context="training rollout",
            )
            print(f'Step {i}: reward is {optimizer_state.best_reward}')
            planning_costs.append(optimizer_state.best_cost)
            planning_feasible.append(optimizer_state.best_feasible)
            planning_any_feasible.append(optimizer_state.any_feasible)
            planning_solver_failed.append(
                optimizer_state.constraint_solver_failed
            )
            for _ in range(self.action_repeat):
                env_state = self.env.step(env_state, action)
                extrinsic_rewards.append(env_state.reward)
            # Calculate intrinsic reward
            z = jnp.concatenate([env_state.obs, action])
            pred = self.model(z, model_state)
            epistemic_std, aleatoric_std = pred.epistemic_std, pred.aleatoric_std
            intrinsic_reward = learned_system.dynamics.get_intrinsic_reward(epistemic_std=epistemic_std,
                                                                            aleatoric_std=aleatoric_std)
            intrinsic_rewards.append(intrinsic_reward)
            collected_states.append(env_state)
            actions.append(action)

        collected_states = jt.map(lambda *xs: jnp.stack(xs), *collected_states)
        actions = jt.map(lambda *xs: jnp.stack(xs), *actions)
        intrinsic_rewards = jt.map(lambda *xs: jnp.stack(xs), *intrinsic_rewards)
        extrinsic_rewards = jt.map(lambda *xs: jnp.stack(xs), *extrinsic_rewards)
        planning_costs = jnp.stack(planning_costs)
        planning_feasible = jnp.stack(planning_feasible)
        planning_any_feasible = jnp.stack(planning_any_feasible)
        planning_solver_failed = jnp.stack(planning_solver_failed)
        self.latest_planning_feasible_fraction = float(
            jnp.mean(planning_feasible.astype(jnp.float32))
        )
        self.latest_planning_any_feasible_fraction = float(
            jnp.mean(planning_any_feasible.astype(jnp.float32))
        )
        self.latest_planning_solver_failure_fraction = float(
            jnp.mean(planning_solver_failed.astype(jnp.float32))
        )
        self.latest_planning_max_selected_cost = float(jnp.max(planning_costs))
        costs = self.cost_fn_env(collected_states.obs[:-1], actions)
        position = collected_states.obs[:-1][:, 0]
        trajectory_constraint = jnp.abs(position) - 1.5
        return collected_states, actions, intrinsic_rewards, extrinsic_rewards, costs, jnp.sum(trajectory_constraint)

    def from_collected_transitions_to_data(self,
                                           collected_states: PyTree[Array, 'episode_length ...'],
                                           actions: Float[Array, 'episode_length action_dim']) -> Data:
        # TODO: Isn't this wrong, if we have a done flag in collected_states?
        states = collected_states.obs[:-1]
        next_states = collected_states.obs[1:]
        inputs = jnp.concatenate([states, actions], axis=-1)
        if self.predict_difference:
            outputs = next_states - states
        else:
            outputs = next_states
        return Data(inputs=inputs, outputs=outputs)

    def do_episode(self,
                   model_state: ModelState,
                   episode_idx: int,
                   data: Data,
                   key: Key[Array, '2'],
                   save_agent: bool = True,
                   train_model: bool = True,
                   folder_name: str = 'experiment_2024'
                   ) -> (ModelState, Data):
        if (
                self.gp_path_source == 'prior'
                and self._fixed_prior_path_state is None
        ):
            # Theory-aligned lifecycle: draw the fixed finite-RFF prior paths
            # before any D0 update. If offline data are supplied, their
            # posterior confidence tube is appended below before episode-0
            # planning.
            model_state = self._initialize_fixed_prior_paths(
                model_state=model_state,
                episode_key=key,
                condition_on_initial_data=False,
            )

        if train_model:
            # If we collected some data already then we train dynamics model and the policy
            print(f'Start of dynamics training')
            model_state = self.train_dynamics_model(model_state=model_state,
                                                    data=data,
                                                    episode_idx=episode_idx)

        if self.gp_path_source == 'prior' and train_model:
            # Every real-data update, including an optional offline D0 update,
            # contributes the next beta_n sigma_n tube.
            self._append_fixed_prior_confidence(model_state)

        posterior_path_state = self.sample_episode_posterior_paths(
            model_state=model_state,
            episode_key=key,
        )

        # We collect new data with the current policy
        print(f'Start of data collection')
        exploration_states, exploration_actions, intrinsic_rewards, extrinsic_rewards, cost, trajectory_constraint = self.simulate_on_true_env(
            model_state=model_state,
            key=key,
            posterior_path_state=posterior_path_state)

        self.on_exploration_rollout_end(
            episode_idx=episode_idx,
            intrinsic_rewards=intrinsic_rewards,
            extrinsic_rewards=extrinsic_rewards,
        )

        # import matplotlib.pyplot as plt
        # plt.plot(exploration_states.obs)
        # plt.axhline(y=-1.5, color='r', linestyle='-')
        # plt.axhline(y=1.5, color='r', linestyle='-')
        # plt.show()

        if self.log_to_wandb:
            metrics = {
                'episode_idx': episode_idx,
                'intrinsic_rewards': jnp.sum(intrinsic_rewards).item(),
                'extrinsic_rewards': jnp.sum(extrinsic_rewards).item(),
                'constraint_cost': cost.item(),
                'trajectory_constraint': trajectory_constraint.item(),
                'planning_feasible_fraction': self.latest_planning_feasible_fraction,
                'planning_any_feasible_fraction': self.latest_planning_any_feasible_fraction,
                'planning_solver_failure_fraction':
                    self.latest_planning_solver_failure_fraction,
                'planning_max_selected_cost': self.latest_planning_max_selected_cost,
            }
            beta_values = jnp.atleast_1d(model_state.beta)
            for output_idx, beta_value in enumerate(beta_values):
                metrics[f'gp/beta_{output_idx}'] = float(beta_value)
            if self._fixed_prior_path_state is not None:
                metrics['gp/recursive_confidence_tubes'] = len(
                    self._fixed_prior_path_state.confidence_model_states
                )
            if hasattr(self, 'action_cost'):
                action_tolerance = ToleranceReward(bounds=(-0.1, 0.1), margin=0.1, sigmoid='gaussian')
                action_penalty = getattr(self, 'action_cost') * jnp.sum(1 - action_tolerance(exploration_actions))
                metrics['sbsrl_action_penalty'] = action_penalty.item()
                metrics['sbsrl_reward_no_exploration_penalty'] = (
                    jnp.sum(extrinsic_rewards) - action_penalty
                ).item()
            metrics.update(self.get_episode_wandb_metrics(episode_idx))
            wandb.log(metrics)

        task_outputs = []
        for task in self.test_tasks:
            task_output = self.test_a_task(
                model_state=model_state,
                key=key,
                task=task,
                posterior_path_state=posterior_path_state,
            )
            task_metrics = task_output[-1]
            task_outputs.append(task_output[:-1])
            if self.log_to_wandb:
                task_metrics['episode_idx'] = episode_idx
                wandb.log(task_metrics)
            else:
                print(task_metrics)
            print(f'End of task {task.name} evaluation')

        new_data = self.from_collected_transitions_to_data(exploration_states, exploration_actions)
        data = Data(inputs=jnp.concatenate([data.inputs, new_data.inputs]),
                    outputs=jnp.concatenate([data.outputs, new_data.outputs]), )

        # We save everything with pickle
        folder_name = os.path.join(folder_name, f'episode_{episode_idx}')
        create_folder(folder_name)

        if (not self.log_to_wandb) and save_agent:
            # Saving data to a pickle file
            with open(os.path.join(folder_name, 'data.pkl'), 'wb') as file:
                pickle.dump(data, file)

            with open(os.path.join(folder_name, 'model_state.pkl'), 'wb') as file:
                pickle.dump(model_state, file)

            with open(os.path.join(folder_name, 'exploration_trajectory.pkl'), 'wb') as file:
                pickle.dump(ExplorationTrajectory(states=exploration_states, actions=exploration_actions,
                                                  intrinsic_rewards=intrinsic_rewards,
                                                  extrinsic_rewards=extrinsic_rewards), file)

            with open(os.path.join(folder_name, 'task_outputs.pkl'), 'wb') as file:
                pickle.dump(task_outputs, file)

        if self.log_to_wandb and save_agent:
            folder_name = os.path.join(wandb.run.dir, 'saved_data', f'episode_{episode_idx}')
            create_folder(folder_name)

            # Saving data to a pickle file
            with open(os.path.join(folder_name, 'data.pkl'), 'wb') as file:
                pickle.dump(data, file)
            wandb.save(os.path.join(folder_name, 'data.pkl'), wandb.run.dir)

            with open(os.path.join(folder_name, 'model_state.pkl'), 'wb') as file:
                pickle.dump(model_state, file)

            wandb.save(os.path.join(folder_name, 'model_state.pkl'), wandb.run.dir)

            with open(os.path.join(folder_name, 'exploration_trajectory.pkl'), 'wb') as file:
                pickle.dump(ExplorationTrajectory(states=exploration_states, actions=exploration_actions,
                                                  intrinsic_rewards=intrinsic_rewards,
                                                  extrinsic_rewards=extrinsic_rewards,
                                                  ), file)

            wandb.save(os.path.join(folder_name, 'exploration_trajectory.pkl'), wandb.run.dir)

            with open(os.path.join(folder_name, 'task_outputs.pkl'), 'wb') as file:
                pickle.dump(task_outputs, file)

            wandb.save(os.path.join(folder_name, 'task_outputs.pkl'), wandb.run.dir)

        return model_state, data

    def run_episodes(self,
                     num_episodes: int,
                     key: Key[Array, '2'] = jr.PRNGKey(0),
                     model_state: ModelState | None = None,
                     data: Data | None = None,
                     folder_name: str = 'experiment_2024') -> (ModelState, Data):
        create_folder(folder_name)
        train_model = True
        if data is None:
            data = Data(inputs=jnp.zeros(shape=(0, self.env.observation_size + self.env.action_size)),
                        outputs=jnp.zeros(shape=(0, self.env.observation_size)))
            train_model = False

        for episode_idx in range(num_episodes):
            key, subkey = jr.split(key)
            train_model = train_model or episode_idx > 0
            self.current_episode_idx = episode_idx
            print(f'Starting with Episode {episode_idx}')
            save_agent = episode_idx % self.saving_frequency == 0
            model_state, data = self.do_episode(model_state=model_state,
                                                episode_idx=episode_idx,
                                                data=data,
                                                key=subkey,
                                                train_model=train_model,
                                                save_agent=save_agent,
                                                folder_name=folder_name)
            self.on_episode_end(episode_idx)
            print(f'End of Episode {episode_idx}')
        return model_state, data


class ActSafeAgent(SafeModelBasedAgent):
    def __init__(self,
                 actsafe_index: int = -1,
                 actsafe_task_index: int = 0,
                 *args,
                 **kwargs):
        super().__init__(train_task_index=-1, *args, **kwargs)
        self.actsafe_index = actsafe_index
        self.actsafe_task_index = actsafe_task_index
        self.current_episode_idx = 0

    def get_train_rewards(self) -> Reward:
        if self.actsafe_index >= 0 and self.current_episode_idx >= self.actsafe_index:
            return self.test_tasks[self.actsafe_task_index].reward
        return super().get_train_rewards()


class SafeHUCRL(SafeModelBasedAgent):
    def __init__(self, train_task_index: int = 0, *args, **kwargs):
        assert train_task_index >= 0
        super().__init__(train_task_index=train_task_index, *args, **kwargs)


if __name__ == '__main__':
    from smbrl.envs.pendulum import PendulumEnv
    from smbrl.playground.pendulum_icem import VelocityBound
    from smbrl.dynamics_models.gps import ARD
    import optax

    from mbrl.utils.offline_data import PendulumOfflineData

    # num_offline_data = 100
    offline_data_gen = PendulumOfflineData()
    key = jr.PRNGKey(0)
    offline_data = None
    # offline_data_key, key = jr.split(key)
    # offline_data = offline_data_gen.sample_transitions(key=offline_data_key,
    #                                                    num_samples=num_offline_data)
    #
    # offline_data = Data(inputs=jnp.concatenate([offline_data.observation, offline_data.action], axis=-1),
    #                     outputs=offline_data.next_observation,
    #                     )

    env = PendulumEnv()
    log_wandb = True

    model = GPStatisticalModel(
        kernel=ARD(input_dim=env.observation_size + env.action_size),
        input_dim=env.observation_size + env.action_size,
        output_dim=env.observation_size,
        output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
        logging_wandb=False,
        f_norm_bound=3 * jnp.ones(shape=(env.observation_size,)),
        beta=None,
        num_training_steps=optax.constant_schedule(1000)
    )

    # model = DeterministicEnsemble(
    #     features=(256, 256),
    #     num_particles=5,
    #     input_dim=env.observation_size + env.action_size,
    #     output_dim=env.observation_size,
    #     output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
    #     logging_wandb=log_wandb)

    icem_horizon = 20


    @chex.dataclass
    class PendulumRewardParams:
        control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.02))
        angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
        target_angle: chex.Array = struct.field(default_factory=lambda: jnp.array(0.0))


    class PendulumReward(Reward):
        def __init__(self, target_angle: float = 0.0):
            super().__init__(x_dim=3, u_dim=1)
            self.target_angle = jnp.array(target_angle)

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: PendulumRewardParams,
                     x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
            chex.assert_shape(x, (self.x_dim,))
            chex.assert_shape(u, (self.u_dim,))
            chex.assert_shape(x_next, (self.x_dim,))
            # get intrinsic reward out
            theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
            target_angle = reward_params.target_angle
            diff_th = theta - target_angle
            diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
            reward = -(reward_params.angle_cost * diff_th ** 2 +
                       0.1 * omega ** 2) - reward_params.control_cost * u ** 2
            reward = reward.squeeze()
            return Normal(loc=reward, scale=jnp.zeros_like(reward)), reward_params

        def init_params(self, key: chex.PRNGKey) -> PendulumRewardParams:
            default_reward_params = PendulumRewardParams()
            return default_reward_params.replace(target_angle=self.target_angle)


    class PendulumEnvBalance(PendulumEnv):
        def reset(self,
                  rng: jax.Array) -> State:
            # set initial state to upright
            state = State(pipeline_state=None,
                          obs=jnp.array([1.0, 0.0, 0.0]),
                          reward=jnp.array(0.0),
                          done=jnp.array(0.0), )
            if self.add_process_noise:
                state.info['process_noise_key'] = rng
            return state


    icem_params = iCemParams(
        num_particles=10,
        num_samples=500,
        alpha=0.2,
        num_steps=5,
        exponent=2,
        lambda_constraint=1e6
    )

    agent = ActSafeAgent(
        env=PendulumEnv(),
        model=model,
        episode_length=50,
        action_repeat=2,
        # cost_fn=None,
        cost_fn=VelocityBound(horizon=icem_horizon,
                              max_abs_velocity=6.0 - 10 ** (-3),
                              violation_eps=1e-3, ),
        test_tasks=[Task(reward=PendulumReward(), name='Swing up', env=env),
                    Task(reward=PendulumReward(), name='Balance', env=PendulumEnvBalance()),
                    Task(reward=PendulumReward(target_angle=jnp.pi), name='Keep down', env=env),
                    ],
        predict_difference=True,
        num_training_steps=constant_schedule(1000),
        icem_horizon=icem_horizon,
        icem_params=icem_params,
        log_to_wandb=log_wandb,
    )

    model_state = model.init(jr.PRNGKey(0))
    if log_wandb:
        wandb.init(project='act safe test')
    agent.run_episodes(num_episodes=20,
                       key=key,
                       model_state=model_state,
                       folder_name='Cost30Aug2024',
                       data=offline_data,
                       )
