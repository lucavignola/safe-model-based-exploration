from typing import NamedTuple
from typing import Union, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from brax.envs.base import State, Env
from distrax import Normal
from flax import struct
from mbpo.systems import DynamicsParams, RewardParams
from mbpo.systems.base_systems import System, SystemParams, SystemState
from mbpo.systems.dynamics.base_dynamics import Dynamics
from mbpo.systems.rewards.base_rewards import Reward


@chex.dataclass
class BicycleCarRewardParams:
    action_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.02))
    goal: chex.Array = struct.field(default_factory=lambda: jnp.zeros(3))


class BicycleCarReward(Reward):
    def __init__(self,
                 encode_angle: bool = True):
        super().__init__(x_dim=6, u_dim=2)
        self.encode_angle = encode_angle
        self.angle_idx = 2

    @staticmethod
    def decode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
        """ Decodes the angle (theta) from sin(theta) and cos(theta)"""
        assert angle_idx < state.shape[0] - 1
        theta = jnp.arctan2(state[angle_idx:angle_idx + 1],
                            state[angle_idx + 1:angle_idx + 2])
        state_decoded = jnp.concatenate([state[:angle_idx], theta, state[angle_idx + 2:]], axis=-1)
        assert state_decoded.shape[-1] == state.shape[-1] - 1
        return state_decoded

    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 reward_params: BicycleCarRewardParams,
                 x_next: chex.Array | None = None):
        if self.encode_angle:
            x = self.decode_angles(x, angle_idx=self.angle_idx)
        pos_diff = x[:2] - reward_params.goal[:2]
        theta_diff = x[2] - reward_params.goal[2]
        pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_diff), axis=-1))
        theta_dist = jnp.abs(((theta_diff + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        total_dist = jnp.sqrt(pos_dist ** 2 + theta_dist ** 2)
        reward = - total_dist - reward_params.action_cost * jnp.sum(jnp.square(u), axis=-1)
        reward = reward.reshape(-1).squeeze()
        return Normal(reward, 1e-3 * jnp.ones_like(reward)), reward_params

    def init_params(self, key: chex.PRNGKey) -> RewardParams:
        return BicycleCarRewardParams()


class CarParams(NamedTuple):
    """
    d_f, d_r : Represent grip of the car. Range: [0.015, 0.025]
    b_f, b_r: Slope of the pacejka. Range: [2.0 - 4.0].

    delta_limit: [0.3 - 0.5] -> Limit of the steering angle.

    c_m_1: Motor parameter. Range [0.2, 0.5]
    c_m_1: Motor friction, Range [0.00, 0.007]
    c_f, c_r: [1.0 2.0] # motor parameters: source https://web.mit.edu/drela/Public/web/qprop/motor1_theory.pdf,
    https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Embedded
    -Control-Systems/LectureNotes/6_Motor_Control.pdf # or look at:
    https://video.ethz.ch/lectures/d-mavt/2021/spring/151-0593-00L/00718f4f-116b-4645-91da-b9482164a3c7.html :
    lecture 2 part 2
    c_m_1: max current of motor: [0.2 - 0.5] c_m_2: motor resistance due to shaft: [0.01 - 0.15]
    """
    m: Union[jax.Array, float] = jnp.array(1.65)  # [0.04, 0.08]
    i_com: Union[jax.Array, float] = jnp.array(2.78e-05)  # [1e-6, 5e-6]
    l_f: Union[jax.Array, float] = jnp.array(0.13)  # [0.025, 0.05]
    l_r: Union[jax.Array, float] = jnp.array(0.17)  # [0.025, 0.05]
    g: Union[jax.Array, float] = jnp.array(9.81)

    d_f: Union[jax.Array, float] = jnp.array(0.02)  # [0.015, 0.025]
    c_f: Union[jax.Array, float] = jnp.array(1.2)  # [1.0, 2.0]
    b_f: Union[jax.Array, float] = jnp.array(2.58)  # [2.0, 4.0]

    d_r: Union[jax.Array, float] = jnp.array(0.017)  # [0.015, 0.025]
    c_r: Union[jax.Array, float] = jnp.array(1.27)  # [1.0, 2.0]
    b_r: Union[jax.Array, float] = jnp.array(3.39)  # [2.0, 4.0]

    c_m_1: Union[jax.Array, float] = jnp.array(10.431917)  # [0.2, 0.5]
    c_m_2: Union[jax.Array, float] = jnp.array(1.5003588)  # [0.00, 0.007]
    c_d: Union[jax.Array, float] = jnp.array(0.0)  # [0.01, 0.1]
    steering_limit: Union[jax.Array, float] = jnp.array(0.19989373)
    use_blend: Union[jax.Array, float] = jnp.array(0.0)  # 0.0 -> (only kinematics), 1.0 -> (kinematics + dynamics)

    # parameters used to compute the blend ratio characteristics
    blend_ratio_ub: Union[jax.Array, float] = jnp.array([0.5477225575])
    blend_ratio_lb: Union[jax.Array, float] = jnp.array([0.4472135955])
    angle_offset: Union[jax.Array, float] = jnp.array([0.02791893])


class BicycleCarModel(Dynamics):
    """
    local_coordinates: bool
        Used to indicate if local or global coordinates shall be used.
        If local, the state x is
            x = [0, 0, theta, vel_r, vel_t, angular_velocity_z]
        else:
            x = [x, y, theta, vel_x, vel_y, angular_velocity_z]
    u = [steering_angle, throttle]
    encode_angle: bool
        Encodes angle to sin and cos if true
    """

    def __init__(self, dt: float = 1 / 30., encode_angle: bool = True):
        self.encode_angle = encode_angle
        x_dim = 6
        super().__init__(x_dim=x_dim + int(encode_angle), u_dim=2)
        self._x_dim = x_dim
        self.dt = dt
        self.params = CarParams()

        self.dt_integration = 1 / 90.
        assert dt >= self.dt_integration
        assert (dt / self.dt_integration - int(
            dt / self.dt_integration)) < 1e-4, 'dt must be multiple of dt_integration'
        self._num_steps_integrate = int(dt / self.dt_integration)
        self.angle_idx = 2
        self.velocity_start_idx = 4 if self.encode_angle else 3
        self.velocity_end_idx = 5 if self.encode_angle else 4

    def rk_integration(self, x: jnp.array, u: jnp.array, params: CarParams) -> jnp.array:
        integration_factors = jnp.asarray([self.dt_integration / 2.,
                                           self.dt_integration / 2., self.dt_integration,
                                           self.dt_integration])
        integration_weights = jnp.asarray([self.dt_integration / 6.,
                                           self.dt_integration / 3., self.dt_integration / 3.0,
                                           self.dt_integration / 6.0])

        def body(carry, _):
            """one step of rk integration.
            k_0 = self.ode(x, u)
            k_1 = self.ode(x + self.dt_integration / 2. * k_0, u)
            k_2 = self.ode(x + self.dt_integration / 2. * k_1, u)
            k_3 = self.ode(x + self.dt_integration * k_2, u)

            x_next = x + self.dt_integration * (k_0 / 6. + k_1 / 3. + k_2 / 3. + k_3 / 6.)
            """

            def rk_integrate(carry, ins):
                k = self.ode(carry, u, params)
                carry = carry + k * ins
                outs = k
                return carry, outs

            _, dxs = jax.lax.scan(rk_integrate, carry, xs=integration_factors, length=4)
            dx = (dxs.T * integration_weights).sum(axis=-1)
            q = carry + dx
            return q, None

        next_state, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        if self.angle_idx is not None:
            theta = next_state[self.angle_idx]
            sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
            next_state = next_state.at[self.angle_idx].set(jnp.arctan2(sin_theta, cos_theta))
        return next_state

    def next_state(self, x: chex.Array,
                   u: chex.Array,
                   dynamics_params: CarParams) -> jnp.array:
        theta_x = jnp.arctan2(x[..., self.angle_idx], x[..., self.angle_idx + 1]) if self.encode_angle else \
            x[..., self.angle_idx]
        offset = jnp.clip(dynamics_params.angle_offset, -jnp.pi, jnp.pi)
        theta_x = theta_x + offset
        # rotate velocity to local frame to compute dx
        velocity_global = x[..., self.velocity_start_idx: self.velocity_end_idx + 1]
        rotated_vel = self.rotate_vector(velocity_global,
                                         -theta_x)
        x = x.at[..., self.velocity_start_idx: self.velocity_end_idx + 1].set(rotated_vel)
        if self.encode_angle:
            x_reduced = self.reduce_x(x)
            x_reduced = self.rk_integration(x_reduced, u, dynamics_params)

            next_theta = jnp.atleast_1d(x_reduced[..., self.angle_idx])
            next_x = jnp.concatenate([x_reduced[..., 0:self.angle_idx], jnp.sin(next_theta), jnp.cos(next_theta),
                                      x_reduced[..., self.angle_idx + 1:]], axis=-1)
        else:
            next_x = self.rk_integration(x, u, dynamics_params)

        # convert velocity to global frame
        new_theta_x = jnp.arctan2(next_x[..., self.angle_idx], next_x[..., self.angle_idx + 1]) \
            if self.encode_angle else next_x[..., self.angle_idx]
        new_theta_x = new_theta_x + offset
        velocity = next_x[..., self.velocity_start_idx: self.velocity_end_idx + 1]
        rotated_vel = self.rotate_vector(velocity, new_theta_x)
        next_x = next_x.at[..., self.velocity_start_idx: self.velocity_end_idx + 1].set(rotated_vel)
        return Normal(next_x, 1e-3 * jnp.ones_like(next_x)), dynamics_params

    def reduce_x(self, x):
        theta = jnp.arctan2(x[..., self.angle_idx], x[..., self.angle_idx + 1])

        x_reduced = jnp.concatenate([x[..., 0:self.angle_idx], jnp.atleast_1d(theta),
                                     x[..., self.velocity_start_idx:]],
                                    axis=-1)
        return x_reduced

    @staticmethod
    def rotate_vector(v, theta):
        v_x, v_y = v[..., 0], v[..., 1]
        rot_x = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        rot_y = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        return jnp.concatenate([jnp.atleast_1d(rot_x), jnp.atleast_1d(rot_y)], axis=-1)

    def _accelerations(self, x, u, params: CarParams):
        """Compute acceleration forces for dynamic model.
        Inputs
        -------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, velocity_r, velocity_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        acceleration: jnp.ndarray,
            shape = (3, ) -> [a_r, a_t, a_theta]
        """
        i_com = params.i_com
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        d_f = params.d_f * params.g
        d_r = params.d_r * params.g
        c_f = params.c_f
        c_r = params.c_r
        b_f = params.b_f
        b_r = params.b_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2

        c_d = params.c_d

        delta, d = u[0], u[1]

        alpha_f = - jnp.arctan(
            (w * l_f + v_y) /
            (v_x + 1e-6)
        ) + delta
        alpha_r = jnp.arctan(
            (w * l_r - v_y) /
            (v_x + 1e-6)
        )
        f_f_y = d_f * jnp.sin(c_f * jnp.arctan(b_f * alpha_f))
        f_r_y = d_r * jnp.sin(c_r * jnp.arctan(b_r * alpha_r))
        f_r_x = (c_m_1 * d - (c_m_2 ** 2) * v_x - (c_d ** 2) * (v_x * jnp.abs(v_x)))

        v_x_dot = (f_r_x - f_f_y * jnp.sin(delta) + m * v_y * w) / m
        v_y_dot = (f_r_y + f_f_y * jnp.cos(delta) - m * v_x * w) / m
        w_dot = (f_f_y * l_f * jnp.cos(delta) - f_r_y * l_r) / i_com

        acceleration = jnp.array([v_x_dot, v_y_dot, w_dot])
        return acceleration

    def _ode_dyn(self, x, u, params: CarParams):
        """Compute derivative using dynamic model.
        Inputs
        -------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, velocity_r, velocity_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        x_dot: jnp.ndarray,
            shape = (6, ) -> time derivative of x

        """
        # state = [p_x, p_y, theta, v_x, v_y, w]. Velocities are in local coordinate frame.
        # Inputs: [\delta, d] -> \delta steering angle and d duty cycle of the electric motor.
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        p_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        theta_dot = w
        p_x_dot = jnp.array([p_x_dot, p_y_dot, theta_dot])

        accelerations = self._accelerations(x, u, params)

        x_dot = jnp.concatenate([p_x_dot, accelerations], axis=-1)
        return x_dot

    def _compute_dx_kin(self, x, u, params: CarParams):
        """
        Compute kinematics derivative for localized state.
        Inputs
        -----
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, v_x, v_y, w], velocities in local frame
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        dx_kin: jnp.ndarray,
            shape = (6, ) -> derivative of x

        Assumption: dot{delta} = 0.
        """
        p_x, p_y, theta, v_x, v_y, w = x[0], x[1], x[2], x[3], x[4], x[5]  # progress
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_d = params.c_d
        delta, d = u[0], u[1]
        v_r = v_x
        v_r_dot = (c_m_1 * d - (c_m_2 ** 2) * v_r - (c_d ** 2) * (v_r * jnp.abs(v_r))) / m
        beta = jnp.arctan(jnp.tan(delta) * 1 / (l_r + l_f))
        v_x_dot = v_r_dot * jnp.cos(beta)
        # Determine accelerations from the kinematic model using FD.
        v_y_dot = (v_r * jnp.sin(beta) * l_r - v_y) / self.dt_integration
        # v_x_dot = (v_r_dot + v_y * w)
        # v_y_dot = - v_x * w
        w_dot = (jnp.sin(beta) * v_r - w) / self.dt_integration
        p_g_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_g_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        dx_kin = jnp.asarray([p_g_x_dot, p_g_y_dot, w, v_x_dot, v_y_dot, w_dot])
        return dx_kin

    def _compute_dx(self, x, u, params: CarParams):
        """Calculate time derivative of state.
        Inputs:
        ------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: jnp.ndarray, derivative of x


        If params.use_blend <= 0.5 --> only kinematic model is used, else a blend between nonlinear model
        and kinematic is used.
        """
        use_kin = params.use_blend <= 0.5
        v_x = x[3]
        blend_ratio_ub = jnp.square(params.blend_ratio_ub)
        blend_ratio_lb = jnp.square(params.blend_ratio_lb)
        blend_ratio = (v_x - blend_ratio_ub) / (blend_ratio_lb + 1E-6)
        blend_ratio = blend_ratio.squeeze()
        lambda_blend = jnp.min(jnp.asarray([
            jnp.max(jnp.asarray([blend_ratio, 0])), 1])
        )
        dx_kin_full = self._compute_dx_kin(x, u, params)
        dx_dyn = self._ode_dyn(x=x, u=u, params=params)
        dx_blend = lambda_blend * dx_dyn + (1 - lambda_blend) * dx_kin_full
        dx = (1 - use_kin) * dx_blend + use_kin * dx_kin_full
        return dx

    def _ode(self, x, u, params: CarParams):
        """
        Using kinematic model with blending: https://arxiv.org/pdf/1905.05150.pdf
        Code based on: https://github.com/alexliniger/gym-racecar/

        Inputs:
        ------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: jnp.ndarray, derivative of x
        """
        delta, d = u[0], u[1]
        delta = jnp.clip(delta, a_min=-1, a_max=1) * params.steering_limit
        d = jnp.clip(d, a_min=-1., a_max=1)  # throttle
        u = u.at[0].set(delta)
        u = u.at[1].set(d)
        dx = self._compute_dx(x, u, params)
        return dx

    def ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        chex.assert_size(x, (self._x_dim,)) and chex.assert_size(u, (self.u_dim,))
        return self._ode(x, u, params)

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        return CarParams()


class RaceCarSystem(System):
    def __init__(self, dynamics: BicycleCarModel = BicycleCarModel(), reward: BicycleCarReward = BicycleCarReward()):
        super().__init__(dynamics=dynamics, reward=reward)
        # Here we have to set the axes of the system parameters which we want to vmap over

    @staticmethod
    def system_params_vmap_axes(axes: int = 0):
        return SystemParams(dynamics_params=None, reward_params=None, key=axes)

    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams[DynamicsParams, RewardParams],
             ) -> SystemState:
        """

        :param x: current state of the system
        :param u: current action of the system
        :param system_params: parameters of the system
        :return: Tuple of next state, reward, updated system parameters
        """
        x_next, _ = self.dynamics.next_state(x, u, system_params.dynamics_params)
        x_next = x_next.mean()
        reward, _ = self.reward(x=x, u=u, reward_params=system_params.reward_params, x_next=x_next)
        return SystemState(x_next=x_next, reward=reward.mean(), system_params=system_params)

    def init_params(self, key: chex.PRNGKey) -> SystemParams[DynamicsParams, RewardParams]:
        keys = jr.split(key, 3)
        return SystemParams(
            dynamics_params=self.dynamics.init_params(keys[0]),
            reward_params=self.reward.init_params(keys[1]),
            key=keys[2],
        )


class RaceCarEnv(Env):

    def __init__(self,
                 init_pos: chex.Array = jnp.array([1.42, -1.04, jnp.pi]),
                 encode_angle: bool = True
                 ):
        self.system = RaceCarSystem(dynamics=BicycleCarModel(encode_angle=encode_angle),
                                    reward=BicycleCarReward(encode_angle=encode_angle),
                                    )
        self._init_pos = init_pos
        self.encode_angle = encode_angle

    @property
    def init_pos(self):
        pos = self._init_pos
        theta = pos[self.system.reward.angle_idx]
        pos = jnp.concatenate([pos[:self.system.reward.angle_idx].reshape(-1, 1), jnp.sin(theta).reshape(-1, 1),
                               jnp.sin(theta).reshape(-1, 1),
                               jnp.zeros((3, 1))], axis=0).squeeze(-1)
        return pos

    def reset(self,
              rng: jax.Array) -> State:
        state = State(pipeline_state=self.system.init_params(rng),
                      obs=self.init_pos,
                      reward=jnp.array(0.0),
                      done=jnp.array(0.0), )
        return state

    def step(self, state: State, action: jax.Array) -> State:
        next_state = self.system.step(state.obs, action, state.pipeline_state)

        next_state = State(pipeline_state=next_state.system_params,
                           obs=next_state.x_next,
                           reward=next_state.reward,
                           done=state.done,
                           metrics=state.metrics,
                           info=state.info)
        return next_state

    @property
    def dt(self):
        return self.system.dynamics.dt

    @property
    def observation_size(self) -> int:
        return self.system.x_dim

    @property
    def action_size(self) -> int:
        return self.system.u_dim

    def backend(self) -> str:
        return 'positional'


def decode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """ Decodes the angle (theta) from sin(theta) and cos(theta)"""
    assert angle_idx < state.shape[-1] - 1
    theta = jnp.arctan2(state[..., angle_idx:angle_idx + 1],
                        state[..., angle_idx + 1:angle_idx + 2])
    state_decoded = jnp.concatenate([state[..., :angle_idx], theta, state[..., angle_idx + 2:]], axis=-1)
    assert state_decoded.shape[-1] == state.shape[-1] - 1
    return state_decoded


def rotate_coordinates(state: jnp.array, encode_angle: bool = False) -> jnp.array:
    x_pos, x_vel = state[..., 0:1], state[..., 3 + int(encode_angle): 4 + int(encode_angle)]
    y_pos, y_vel = state[..., 1:2], state[:, 4 + int(encode_angle):5 + int(encode_angle)]
    theta = state[..., 2: 3 + int(encode_angle)]
    new_state = jnp.concatenate([y_pos, -x_pos, theta, y_vel, -x_vel, state[..., 5 + int(encode_angle):]],
                                axis=-1)
    assert state.shape == new_state.shape
    return new_state


def plot_rc_trajectory(traj: jnp.array, actions: Optional[jnp.array] = None, pos_domain_size: float = 5,
                       show: bool = True, encode_angle: bool = False):
    """ Plots the trajectory of the RC car """
    if encode_angle:
        traj = decode_angles(traj, 2)

    import matplotlib.pyplot as plt
    scale_factor = 1.5
    if actions is None:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(scale_factor * 12, scale_factor * 8))
    else:
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(scale_factor * 16, scale_factor * 8))
    axes[0][0].set_xlim(-pos_domain_size, pos_domain_size)
    axes[0][0].set_ylim(-pos_domain_size, pos_domain_size)
    axes[0][0].scatter(0, 0)
    # axes[0][0].plot(traj[:, 0], traj[:, 1])
    axes[0][0].set_title('x-y')

    # chaange x -> -y and y -> x
    traj = rotate_coordinates(traj, encode_angle=False)

    # Plot the velocity of the car as vectors
    total_vel = jnp.sqrt(traj[:, 3] ** 2 + traj[:, 4] ** 2)
    axes[0][0].quiver(traj[0:-1:3, 0], traj[0:-1:3, 1], traj[0:-1:3, 3], traj[0:-1:3, 4],
                      total_vel[0:-1:3], cmap='jet', scale=20,
                      headlength=2, headaxislength=2, headwidth=2, linewidth=0.2)

    t = jnp.arange(traj.shape[0]) / 30.
    # theta
    axes[0][1].plot(t, traj[:, 2])
    axes[0][1].set_xlabel('time')
    axes[0][1].set_ylabel('theta')
    axes[0][1].set_title('theta')

    # angular velocity
    axes[0][2].plot(t, traj[:, -1])
    axes[0][2].set_xlabel('time')
    axes[0][2].set_ylabel('angular velocity')
    axes[0][2].set_title('angular velocity')

    axes[1][0].plot(t, total_vel)
    axes[1][0].set_xlabel('time')
    axes[1][0].set_ylabel('total velocity')
    axes[1][0].set_title('velocity')

    # vel x
    axes[1][1].plot(t, traj[:, 3])
    axes[1][1].set_xlabel('time')
    axes[1][1].set_ylabel('velocity x')
    axes[1][1].set_title('velocity x')

    axes[1][2].plot(t, traj[:, 4])
    axes[1][2].set_xlabel('time')
    axes[1][2].set_ylabel('velocity y')
    axes[1][2].set_title('velocity y')

    if actions is not None:
        # steering
        axes[0][3].plot(t[:actions.shape[0]], actions[:, 0])
        axes[0][3].set_xlabel('time')
        axes[0][3].set_ylabel('steer')
        axes[0][3].set_title('steering')

        # throttle
        axes[1][3].plot(t[:actions.shape[0]], actions[:, 1])
        axes[1][3].set_xlabel('time')
        axes[1][3].set_ylabel('throttle')
        axes[1][3].set_title('throttle')

    fig.tight_layout()
    if show:
        fig.show()
    return fig, axes


if __name__ == '__main__':
    import time
    from jax import jit

    ENCODE_ANGLE = True

    env = RaceCarEnv()
    t_start = time.time()
    state = env.reset(rng=jr.PRNGKey(0))

    traj = [state.obs]
    rewards = []
    actions = []

    step_fn = jit(env.step)

    for i in range(120):
        t = i / 30.
        a = jnp.array([- 0.5 * jnp.cos(1.0 * t), 0.8 / (t + 1)])
        state = step_fn(state, a)

        traj.append(state.obs)
        actions.append(a)
        rewards.append(state.reward)

    duration = time.time() - t_start
    print(f'Duration of trajectory sim {duration} sec')
    traj = jnp.stack(traj)
    actions = jnp.stack(actions)

    plot_rc_trajectory(traj, actions, encode_angle=ENCODE_ANGLE)
    print(traj[-1, :])
