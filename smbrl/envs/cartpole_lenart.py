from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from brax.envs.base import State, Env
from bsm.utils import Data
from flax import struct
from jax import vmap
from jaxtyping import Float, Array, Scalar


@chex.dataclass
class CartPoleDynamicsParams:
    max_torque: chex.Array = struct.field(default_factory=lambda: jnp.array(15.0))
    dt: chex.Array = struct.field(default_factory=lambda: jnp.array(0.05))
    g: chex.Array = struct.field(default_factory=lambda: jnp.array(9.81))
    m_1: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    m_c: chex.Array = struct.field(default_factory=lambda: jnp.array(5.0))
    l_1: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))


@chex.dataclass
class CartPoleRewardParams:
    control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.01))
    angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    pos_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    vel_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.1))
    target_angle: chex.Array = struct.field(default_factory=lambda: jnp.array(jnp.pi))


class CartPoleEnv(Env):
    def __init__(self,
                 reward_source: str = 'gym',
                 init_angle: float = 0.0,
                 ):
        self.dynamics_params = CartPoleDynamicsParams()
        self.reward_params = CartPoleRewardParams()
        self.init_angle = init_angle
        self.reward_source = reward_source

    def reset(self,
              rng: jax.Array) -> State:
        state = State(pipeline_state=None,
                      obs=jnp.array([0.0, jnp.cos(self.init_angle), jnp.sin(self.init_angle), 0.0, 0.0]),
                      reward=jnp.array(0.0),
                      done=jnp.array(0.0), )
        return state

    @staticmethod
    def angle_to_cos_sin_representation(angle: Scalar) -> Float[Array, '2']:
        return jnp.array([jnp.cos(angle), jnp.sin(angle)])

    @staticmethod
    def cos_sin_to_angle_representation(cos_sin_angle: Float[Array, '2']) -> Scalar:
        return jnp.arctan2(cos_sin_angle[1], cos_sin_angle[0])

    def from_state_to_obs(self, obs: Float[Array, '4']) -> Float[Array, '5']:
        assert obs.shape == (4,)
        position, angle, linear_velocity, angular_velocity = obs[0], obs[1], obs[2], obs[3]
        cos_sin_angle = self.angle_to_cos_sin_representation(angle)
        return jnp.array([position, cos_sin_angle[0], cos_sin_angle[1], linear_velocity, angular_velocity])

    def from_obs_to_state(self, state: Float[Array, '5']) -> Float[Array, '4']:
        assert state.shape == (5,)
        position, cos, sin, linear_velocity, angular_velocity = state[0], state[1], state[2], state[3], state[4]
        angle = self.cos_sin_to_angle_representation(jnp.array([cos, sin]))
        return jnp.array([position, angle, linear_velocity, angular_velocity])

    def reward(self,
               x: Float[Array, 'observation_dim'],
               u: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        x_compressed = self.from_obs_to_state(x)
        position, angle = x_compressed[0], x_compressed[1]
        linear_velocity, angular_velocity = x_compressed[2], x_compressed[3]

        target_angle = self.reward_params.target_angle
        diff_th = angle - target_angle
        diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        reward = -(self.reward_params.angle_cost * diff_th ** 2 + self.reward_params.pos_cost * position ** 2 +
                   self.reward_params.vel_cost * (
                           linear_velocity ** 2 + angular_velocity ** 2)) - self.reward_params.control_cost * u[
                     0] ** 2
        reward = reward.squeeze()
        return reward

    @partial(jax.jit, static_argnums=0)
    def step(self,
             state: State,
             action: jax.Array) -> State:
        x = state.obs
        chex.assert_shape(x, (self.observation_size,))
        chex.assert_shape(action, (self.action_size,))

        dt = self.dynamics_params.dt
        x_compressed = self.from_obs_to_state(x)

        dx = self.ode(x_compressed, action)

        next_x_compressed = x_compressed + dx * dt
        next_obs = self.from_state_to_obs(next_x_compressed)

        if self.reward_source == 'gym':
            next_reward = self.reward(x, action)
        elif self.reward_source == 'dm-control':
            raise NotImplementedError(f'{self.reward_source} not implemented')
        else:
            raise NotImplementedError(f'Unknown reward source {self.reward_source}')

        next_state = State(pipeline_state=state.pipeline_state,
                           obs=next_obs,
                           reward=next_reward,
                           done=state.done,
                           metrics=state.metrics,
                           info=state.info)
        return next_state

    def ode(self,
            state: Float[Array, '4'],
            action: Float[Array, '1'], ) -> Float[Array, '4']:

        g = self.dynamics_params.g
        m_1 = self.dynamics_params.m_1
        m_c = self.dynamics_params.m_c
        l_1 = self.dynamics_params.l_1

        u = jnp.clip(action, -1, 1) * self.dynamics_params.max_torque

        x, theta1, x_dot, theta1_dot = state

        cos_theta1 = jnp.cos(theta1)
        sin_theta1 = jnp.sin(theta1)

        A = jnp.array([
            [m_1 + m_c, m_1 * l_1 * cos_theta1],
            [m_1 * l_1 * cos_theta1, m_1 * l_1 * l_1]
        ])
        b = jnp.array([
            u[0] + sin_theta1 * l_1 * m_1 * theta1_dot * theta1_dot,
            -m_1 * l_1 * g * sin_theta1
        ])

        der = jnp.linalg.inv(A).dot(b)

        return jnp.array([x_dot, theta1_dot, der[0], der[1]])

    @property
    def dt(self):
        return self.dynamics_params.dt

    @property
    def observation_size(self) -> int:
        return 5

    @property
    def action_size(self) -> int:
        return 1

    def backend(self) -> str:
        return 'positional'


class CartPoleOfflineData:
    def __init__(self,
                 action_repeat: int,
                 predict_difference: bool = True):
        self.env = CartPoleEnv()
        self.action_repeat = action_repeat
        self.predict_difference = predict_difference

    def dynamics_fn(self,
                    z: Float[Array, '6']):
        obs, action = z[:self.env.observation_size], z[self.env.observation_size:]
        state = State(pipeline_state=None,
                      obs=obs,
                      reward=jnp.array(0.0),
                      done=jnp.array(0.0), )
        for _ in range(self.action_repeat):
            state = self.env.step(state, action)
        return state.obs - obs

    def sample(self,
               key: Float[Array, '2'],
               num_samples: int = 1,
               max_abs_lin_position: float = 0.5,
               max_abs_angle: float = jnp.pi,
               max_abs_lin_velocity: float = 1.0,
               max_abs_ang_velocity: float = 4.0,
               max_abs_action: float = 1.0,
               ):
        key_lin_pos, key_ang, key_lin_vel, key_ang_vel, key_action = jr.split(key, 5)
        lin_pos = jr.uniform(key_lin_pos, minval=-max_abs_lin_position, maxval=max_abs_lin_position,
                             shape=(num_samples,))
        ang = jr.uniform(key_ang, minval=-max_abs_angle, maxval=max_abs_angle, shape=(num_samples,))
        lin_velocity = jr.uniform(key_lin_vel, minval=-max_abs_lin_velocity, maxval=max_abs_lin_velocity,
                                  shape=(num_samples,))
        ang_velocity = jr.uniform(key_ang_vel, minval=-max_abs_ang_velocity, maxval=max_abs_ang_velocity,
                                  shape=(num_samples,))
        actions = jr.uniform(key_action, minval=-max_abs_action, maxval=max_abs_action, shape=(num_samples,))

        inputs = jnp.stack([lin_pos, jnp.cos(ang), jnp.sin(ang), lin_velocity, ang_velocity, actions], axis=-1)
        outputs = vmap(self.dynamics_fn)(inputs)

        return Data(inputs=inputs, outputs=outputs)


class CartPoleTrajectoryOfflineData:
    """
    Alternative offline data generation that samples contiguous trajectories
    instead of uniform grid points, providing more realistic temporal structure.
    """
    def __init__(self,
                 action_repeat: int,
                 predict_difference: bool = True):
        self.env = CartPoleEnv()
        self.action_repeat = action_repeat
        self.predict_difference = predict_difference

    def _is_state_in_bounds(self, obs, bounds):
        """Check if observation is within specified bounds"""
        state = self.env.from_obs_to_state(obs)
        pos, angle, lin_vel, ang_vel = state[0], state[1], state[2], state[3]

        return (jnp.abs(pos) <= bounds['max_abs_lin_position'] and
                jnp.abs(angle) <= bounds['max_abs_angle'] and
                jnp.abs(lin_vel) <= bounds['max_abs_lin_velocity'] and
                jnp.abs(ang_vel) <= bounds['max_abs_ang_velocity'])

    def dynamics_fn(self, z: Float[Array, '6']):
        """Same dynamics function as original for compatibility"""
        obs, action = z[:self.env.observation_size], z[self.env.observation_size:]
        state = State(pipeline_state=None,
                      obs=obs,
                      reward=jnp.array(0.0),
                      done=jnp.array(0.0), )
        for _ in range(self.action_repeat):
            state = self.env.step(state, action)
        return state.obs - obs

    def sample_trajectory(self,
                         key: Float[Array, '2'],
                         trajectory_length: int = 50,
                         max_abs_lin_position: float = 0.5,
                         max_abs_angle: float = jnp.pi,
                         max_abs_lin_velocity: float = 1.0,
                         max_abs_ang_velocity: float = 4.0,
                         max_abs_action: float = 1.0,
                         use_env_reset: bool = True,
                         init_noise_std: float = 0.0,
                         ) -> Data:
        """
        Sample a single contiguous trajectory with random actions

        Args:
            use_env_reset: If True, use env.reset() for initial state (consistent with online)
            init_noise_std: If >0, add Gaussian noise to initial state for diversity
        """

        bounds = {
            'max_abs_lin_position': max_abs_lin_position,
            'max_abs_angle': max_abs_angle,
            'max_abs_lin_velocity': max_abs_lin_velocity,
            'max_abs_ang_velocity': max_abs_ang_velocity
        }

        key_init, key_noise, key_actions = jr.split(key, 3)

        if use_env_reset:
            # Use same initial state as online episodes for consistency
            reset_state = self.env.reset(key_init)
            init_obs = reset_state.obs

        else:
            # Original random sampling approach (for comparison)
            key_lin_pos, key_ang, key_lin_vel, key_ang_vel = jr.split(key_init, 4)

            init_pos = jr.uniform(key_lin_pos, minval=-max_abs_lin_position*0.8, maxval=max_abs_lin_position*0.8)
            init_ang = jr.uniform(key_ang, minval=-max_abs_angle*0.8, maxval=max_abs_angle*0.8)
            init_lin_vel = jr.uniform(key_lin_vel, minval=-max_abs_lin_velocity*0.5, maxval=max_abs_lin_velocity*0.5)
            init_ang_vel = jr.uniform(key_ang_vel, minval=-max_abs_ang_velocity*0.5, maxval=max_abs_ang_velocity*0.5)

            init_obs = self.env.from_state_to_obs(jnp.array([init_pos, init_ang, init_lin_vel, init_ang_vel]))

        # Initialize state
        state = State(pipeline_state=None, obs=init_obs, reward=jnp.array(0.0), done=jnp.array(0.0))

        # Sample random actions for trajectory - ensure each trajectory has different actions
        action_keys = jr.split(key_actions, trajectory_length)
        actions = vmap(lambda k: jr.uniform(k, minval=-max_abs_action, maxval=max_abs_action, shape=(1,)))(action_keys)

        # Roll out trajectory
        states_list = []
        actions_list = []

        for i in range(trajectory_length):
            action = actions[i]
            current_obs = state.obs

            # Check bounds before stepping
            if self._is_state_in_bounds(current_obs, bounds):
                states_list.append(current_obs)
                actions_list.append(action)

                # Step environment
                for _ in range(self.action_repeat):
                    state = self.env.step(state, action)
            else:
                # Stop if we go out of bounds
                break

        if len(states_list) == 0:
            # Fallback to single transition if trajectory too short
            states_list = [init_obs]
            actions_list = [actions[0]]

        # Convert to arrays and create input/output pairs
        trajectory_obs = jnp.array(states_list)
        trajectory_actions = jnp.array(actions_list).squeeze(-1)

        # Create inputs (obs + action)
        inputs = jnp.concatenate([trajectory_obs, trajectory_actions[:, None]], axis=-1)

        # Compute outputs using dynamics function
        outputs = vmap(self.dynamics_fn)(inputs)

        return Data(inputs=inputs, outputs=outputs)

    def sample(self,
               key: Float[Array, '2'],
               num_samples: int = 1,
               max_abs_lin_position: float = 0.5,
               max_abs_angle: float = jnp.pi,
               max_abs_lin_velocity: float = 1.0,
               max_abs_ang_velocity: float = 4.0,
               max_abs_action: float = 1.0,
               num_trajectories: int = None,
               trajectory_length: int = 50,
               use_env_reset: bool = True,
               init_noise_std: float = 0.0,
               ) -> Data:
        """
        Sample offline data using contiguous trajectories.

        Args:
            num_samples: Total number of transitions to collect
            num_trajectories: Number of trajectories to sample (if None, auto-computed)
            trajectory_length: Length of each trajectory
            use_env_reset: If True, use env.reset() for initial states (consistent with online)
            init_noise_std: If >0, add Gaussian noise to initial state for diversity
        """

        if num_trajectories is None:
            num_trajectories = max(1, num_samples // trajectory_length)

        keys = jr.split(key, num_trajectories)

        # Sample multiple trajectories
        all_inputs = []
        all_outputs = []

        for traj_key in keys:
            traj_data = self.sample_trajectory(
                traj_key,
                trajectory_length=trajectory_length,
                max_abs_lin_position=max_abs_lin_position,
                max_abs_angle=max_abs_angle,
                max_abs_lin_velocity=max_abs_lin_velocity,
                max_abs_ang_velocity=max_abs_ang_velocity,
                max_abs_action=max_abs_action,
                use_env_reset=use_env_reset,
                init_noise_std=init_noise_std,
            )
            all_inputs.append(traj_data.inputs)
            all_outputs.append(traj_data.outputs)

        # Concatenate all trajectory data
        if all_inputs:
            combined_inputs = jnp.concatenate(all_inputs, axis=0)
            combined_outputs = jnp.concatenate(all_outputs, axis=0)

            # Trim to requested number of samples
            if combined_inputs.shape[0] > num_samples:
                combined_inputs = combined_inputs[:num_samples]
                combined_outputs = combined_outputs[:num_samples]

            return Data(inputs=combined_inputs, outputs=combined_outputs)
        else:
            # Fallback to empty data
            empty_input = jnp.zeros((0, self.env.observation_size + self.env.action_size))
            empty_output = jnp.zeros((0, self.env.observation_size))
            return Data(inputs=empty_input, outputs=empty_output)


if __name__ == '__main__':
    from jax import jit
    import matplotlib.pyplot as plt

    jax.config.update("jax_enable_x64", True)

    # Test environment simulation
    env = CartPoleEnv(init_angle=jnp.pi)
    state = env.reset(jr.PRNGKey(0))
    action = jnp.array([0.0, ])

    obs = []
    rewards = []
    step_fn = jit(env.step)
    for i in range(100):
        state = step_fn(state, action)
        obs.append(state.obs)
        rewards.append(state.reward)

    obs = jnp.array(obs)

    # Create figure for environment simulation
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    for i in range(env.observation_size):
        plt.plot(obs[:, i], label=f'State {i}')
    plt.legend()
    plt.title('Environment Simulation (Zero Actions)')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')

    # Compare offline data collection methods
    print("Comparing offline data collection methods...")

    # Parameters for comparison
    num_samples = 200  # Total samples to collect
    action_repeat = 2
    key = jr.PRNGKey(42)
    key1, key2 = jr.split(key)

    # Original uniform sampling
    print("Generating uniform grid samples...")
    uniform_sampler = CartPoleOfflineData(action_repeat=action_repeat)
    uniform_data = uniform_sampler.sample(
        key=key1,
        num_samples=20,
        max_abs_lin_position=0.5,
        max_abs_angle=jnp.pi,
        max_abs_lin_velocity=1.0,
        max_abs_ang_velocity=4.0,
        max_abs_action=1.0
    )

    # New trajectory-based sampling
    print("Generating trajectory-based samples...")
    trajectory_sampler = CartPoleTrajectoryOfflineData(action_repeat=action_repeat)
    trajectory_data = trajectory_sampler.sample(
        key=key2,
        num_samples=num_samples,
        num_trajectories=2,  # 5 trajectories to check diversity
        trajectory_length=50,  # Longer trajectories
        use_env_reset=True,
        init_noise_std=0.0,  # No initial noise - OK if same start state
        max_abs_lin_position=0.5,
        max_abs_angle=jnp.pi,
        max_abs_lin_velocity=1.0,
        max_abs_ang_velocity=4.0,
        max_abs_action=1.0
    )

    # Verify action diversity between trajectories
    print("\n=== ACTION DIVERSITY CHECK ===")
    traj_actions_all = trajectory_data.inputs[:, -1]  # Extract all actions
    samples_per_traj = len(traj_actions_all) // 5  # Assuming 5 trajectories

    for i in range(5):  # Check each trajectory
        start_idx = i * samples_per_traj
        end_idx = start_idx + min(5, samples_per_traj)  # First 5 actions of each trajectory
        if end_idx <= len(traj_actions_all):
            actions_slice = traj_actions_all[start_idx:end_idx]
            print(f"  Trajectory {i+1} - First 5 actions: [{', '.join([f'{a:.3f}' for a in actions_slice])}]")

    print(f"\nUniform samples collected: {uniform_data.inputs.shape[0]}")
    print(f"Trajectory samples collected: {trajectory_data.inputs.shape[0]}")

    # Extract state components for visualization
    # Uniform data states (from inputs, excluding action)
    uniform_states = uniform_data.inputs[:, :-1]  # Remove last column (action)
    uniform_pos = uniform_states[:, 0]
    uniform_cos = uniform_states[:, 1]
    uniform_sin = uniform_states[:, 2]
    uniform_lin_vel = uniform_states[:, 3]
    uniform_ang_vel = uniform_states[:, 4]
    uniform_actions = uniform_data.inputs[:, -1]

    # Trajectory data states
    traj_states = trajectory_data.inputs[:, :-1]
    traj_pos = traj_states[:, 0]
    traj_cos = traj_states[:, 1]
    traj_sin = traj_states[:, 2]
    traj_lin_vel = traj_states[:, 3]
    traj_ang_vel = traj_states[:, 4]
    traj_actions = trajectory_data.inputs[:, -1]

    # Convert cos/sin back to angles for plotting
    uniform_angles = jnp.arctan2(uniform_sin, uniform_cos)
    traj_angles = jnp.arctan2(traj_sin, traj_cos)

    # Plot comparisons
    plt.subplot(2, 3, 2)
    plt.scatter(uniform_pos, uniform_angles, alpha=0.6, s=20, label='Uniform Grid', color='red')
    plt.scatter(traj_pos, traj_angles, alpha=0.6, s=20, label='Trajectories', color='blue')
    plt.xlabel('Position')
    plt.ylabel('Angle (rad)')
    plt.title('Position vs Angle Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    plt.scatter(uniform_lin_vel, uniform_ang_vel, alpha=0.6, s=20, label='Uniform Grid', color='red')
    plt.scatter(traj_lin_vel, traj_ang_vel, alpha=0.6, s=20, label='Trajectories', color='blue')
    plt.xlabel('Linear Velocity')
    plt.ylabel('Angular Velocity')
    plt.title('Velocity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    plt.hist(uniform_actions, bins=30, alpha=0.7, label='Uniform Grid', color='red', density=True)
    plt.hist(traj_actions, bins=30, alpha=0.7, label='Trajectories', color='blue', density=True)
    plt.xlabel('Action')
    plt.ylabel('Density')
    plt.title('Action Distribution')
    plt.legend()

    # Plot temporal structure for trajectory data (first 50 samples)
    n_plot = min(50, trajectory_data.inputs.shape[0])
    plt.subplot(2, 3, 5)
    plt.plot(traj_pos[:n_plot], 'b-', linewidth=2, label='Position')
    plt.plot(traj_angles[:n_plot], 'g-', linewidth=2, label='Angle')
    plt.plot(traj_actions[:n_plot], 'r--', alpha=0.7, label='Actions')
    plt.xlabel('Trajectory Step')
    plt.ylabel('Value')
    plt.title('Trajectory Temporal Structure')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Compare dynamics prediction differences
    plt.subplot(2, 3, 6)
    uniform_delta_pos = uniform_data.outputs[:, 0]  # Position change
    traj_delta_pos = trajectory_data.outputs[:, 0]

    plt.hist(uniform_delta_pos, bins=30, alpha=0.7, label='Uniform Grid', color='red', density=True)
    plt.hist(traj_delta_pos, bins=30, alpha=0.7, label='Trajectories', color='blue', density=True)
    plt.xlabel('Position Change (Δpos)')
    plt.ylabel('Density')
    plt.title('Dynamics Output Distribution')
    plt.legend()

    plt.tight_layout()
    plt.suptitle('Offline Data Collection Comparison: Uniform Grid vs Trajectories', y=1.02)
    plt.show()

    # Print statistics
    print("\n=== DATA COLLECTION STATISTICS ===")
    print(f"Total samples - Uniform: {len(uniform_data.inputs)}, Trajectory: {len(trajectory_data.inputs)}")
    print(f"\nPosition range:")
    print(f"  Uniform: [{uniform_pos.min():.3f}, {uniform_pos.max():.3f}]")
    print(f"  Trajectory: [{traj_pos.min():.3f}, {traj_pos.max():.3f}]")
    print(f"\nAngle range:")
    print(f"  Uniform: [{uniform_angles.min():.3f}, {uniform_angles.max():.3f}]")
    print(f"  Trajectory: [{traj_angles.min():.3f}, {traj_angles.max():.3f}]")
    print(f"\nAction statistics:")
    print(f"  Uniform - mean: {uniform_actions.mean():.3f}, std: {uniform_actions.std():.3f}")
    print(f"  Trajectory - mean: {traj_actions.mean():.3f}, std: {traj_actions.std():.3f}")
