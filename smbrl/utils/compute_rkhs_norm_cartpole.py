import time

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from brax.envs.base import State
from bsm.statistical_model.gp_statistical_model import GPStatisticalModel
from bsm.utils.normalization import Data
from jax import vmap
from jaxtyping import Float, Array

from smbrl.dynamics_models.gps import ARD
from smbrl.envs.cartpole_lenart import CartPoleEnv

jax.config.update("jax_enable_x64", True)

env = CartPoleEnv()

MAX_POSITION = 0.5

action_repeat = 2
state_dim = 5
action_dim = 1

input_dim = state_dim + action_dim
output_dim = state_dim

key = jr.PRNGKey(0)


def f(x):
    obs, action = x[:state_dim], x[state_dim:]
    state = State(pipeline_state=None,
                  obs=obs,
                  reward=jnp.array(0.0),
                  done=jnp.array(0.0), )
    for _ in range(action_repeat):
        state = env.step(state, action)
    return state.obs - obs


def sample_input(key: Float[Array, '2']):
    key_position, key_angle, key_lin_velocity, key_ang_velocity, key_action = jr.split(key, 5)
    # Sample position and angle
    position = jr.uniform(key=key_position, minval=-MAX_POSITION, maxval=MAX_POSITION, shape=())
    angle = jr.uniform(key=key_angle, minval=-jnp.pi, maxval=jnp.pi, shape=())
    # Sample linear velocity and angular velocity
    lin_velocity = jr.uniform(key=key_lin_velocity, minval=-4.0, maxval=4.0, shape=())
    angular_velocity = jr.uniform(key=key_ang_velocity, minval=-8, maxval=8, shape=())
    # Sample action
    action = jr.uniform(key=key_action, minval=-1, maxval=1, shape=())
    # Return concatenated input vector
    return jnp.array([position, jnp.cos(angle), jnp.sin(angle), lin_velocity, angular_velocity, action])


NUM_TRAIN_SAMPLES = 1_000
keys = jr.split(key, NUM_TRAIN_SAMPLES)
xs = vmap(sample_input)(keys)
ys = vmap(f)(xs)

# Here we need to take care of normalization
data = Data(inputs=xs, outputs=ys)

logging = True

model = GPStatisticalModel(
    kernel=ARD(input_dim=input_dim),
    input_dim=input_dim,
    output_dim=output_dim,
    output_stds=1e-3 * jnp.ones(output_dim, ),
    logging_wandb=logging,
    f_norm_bound=3,
    beta=None,
    normalize=True,
    num_training_steps=optax.constant_schedule(2000)
)

model_state = model.init(jr.PRNGKey(0))
start_time = time.time()
print('Starting with training')
if logging:
    wandb.init(
        project='Cartpole RKHS Norm Computation',
        group='test group',
    )

model_state = model.update(data=data, stats_model_state=model_state)
print(f'Training time: {time.time() - start_time:.2f} seconds')

kernel_params = model_state.model_state.params


def estimate_norm(key: Float[Array, '2'], num_samples: int):
    sample_keys = jr.split(key, num_samples)
    xs = vmap(sample_input)(sample_keys)
    ys = vmap(f)(xs)

    # We normalize the xs and ys
    xs = vmap(model.model.normalizer.normalize, in_axes=(0, None))(xs, model_state.model_state.data_stats.inputs)
    ys = vmap(model.model.normalizer.normalize, in_axes=(0, None))(ys, model_state.model_state.data_stats.outputs)

    K = model.model.m_kernel_multiple_output(xs, xs, kernel_params)
    K += jnp.eye(K.shape[-1]) * 1e-3

    def compute_norm_1d(cov_matrix, function_values):
        cholesky_tuples = jax.scipy.linalg.cho_factor(cov_matrix)
        first_step = jax.scipy.linalg.cho_solve(cholesky_tuples, function_values)
        norm_sq = function_values.T @ first_step
        return jnp.sqrt(norm_sq)

    norms = vmap(compute_norm_1d, in_axes=(0, -1))(K, ys)
    return norms


key = jr.PRNGKey(0)
NUM_TEST_SAMPLES = 1000
norms = estimate_norm(key, NUM_TEST_SAMPLES)
print(norms)

import pickle

with open('kernel_params.pickle', 'wb') as handle:
    pickle.dump(kernel_params, handle)

with open('norms.pickle', 'wb') as handle:
    pickle.dump(norms, handle)

with open('data_stats.pickle', 'wb') as handle:
    pickle.dump(model_state.model_state.data_stats, handle)

