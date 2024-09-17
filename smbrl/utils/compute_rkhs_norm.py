import time

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from brax.envs.base import State
from bsm.statistical_model.gp_statistical_model import GPStatisticalModel
from bsm.utils.normalization import Data
from jax import tree, vmap

from smbrl.dynamics_models.gps import ARD
from smbrl.envs.pendulum import PendulumEnv

jax.config.update("jax_enable_x64", True)

env = PendulumEnv()
DIMENSION = 2
action_repeat = 2


def f(x):
    obs, action = x[:3], x[3]
    state = State(pipeline_state=None,
                  obs=obs,
                  reward=jnp.array(0.0),
                  done=jnp.array(0.0), )
    for _ in range(action_repeat):
        state = env.step(state, action.reshape(1, ))
    return state.obs - obs


input_dim = 4
output_dim = 1

kernel = ARD(input_dim=input_dim)

# import jax
# jax.config.update('jax_log_compiles', True)

key = jr.PRNGKey(0)

NUM_SAMPLES = 100

theta = jr.uniform(key=jr.PRNGKey(0), minval=-jnp.pi, maxval=jnp.pi, shape=(NUM_SAMPLES,))
velocity = jr.uniform(key=jr.PRNGKey(0), minval=-6, maxval=6, shape=(NUM_SAMPLES,))
action = jr.uniform(key=jr.PRNGKey(0), minval=-1, maxval=1, shape=(NUM_SAMPLES,))

xs = jnp.stack([jnp.cos(theta), jnp.sin(theta), velocity, action], axis=1)
ys = vmap(f)(xs)[:, DIMENSION].reshape(NUM_SAMPLES, 1)

data = Data(inputs=xs, outputs=ys)

logging = False
num_particles = 10
model = GPStatisticalModel(
    kernel=ARD(input_dim=input_dim),
    input_dim=input_dim,
    output_dim=output_dim,
    output_stds=1e-3 * jnp.ones(output_dim, ),
    logging_wandb=False,
    f_norm_bound=3,
    beta=None,
    num_training_steps=optax.constant_schedule(1000)
)
model_state = model.init(jr.PRNGKey(0))
start_time = time.time()
print('Starting with training')
if logging:
    wandb.init(
        project='Pendulum',
        group='test group',
    )

# model_state = model.update(data=data, model_state=model_state)
print(f'Training time: {time.time() - start_time:.2f} seconds')
model_state = model.update(data=data, stats_model_state=model_state)

kernel_params = tree.map(lambda x: x[0], model_state.model_state.params)

for i in range(300):
    NUM_SAMPLES = i

    theta = jr.uniform(key=jr.PRNGKey(0), minval=-jnp.pi, maxval=jnp.pi, shape=(NUM_SAMPLES,))
    velocity = jr.uniform(key=jr.PRNGKey(0), minval=-6, maxval=6, shape=(NUM_SAMPLES,))
    action = jr.uniform(key=jr.PRNGKey(0), minval=-1, maxval=1, shape=(NUM_SAMPLES,))

    xs = jnp.stack([jnp.cos(theta), jnp.sin(theta), velocity, action], axis=1)
    ys = vmap(f)(xs)[:, DIMENSION].reshape(NUM_SAMPLES, 1)

    data = Data(inputs=xs, outputs=ys)

    v_k = vmap(kernel.apply, in_axes=(0, None, None))
    m_k = vmap(v_k, in_axes=(None, 0, None))

    K = m_k(xs, xs, kernel_params)

    # K_inv = inv(K)

    cholesky_tuples = jax.scipy.linalg.cho_factor(K)
    first_step = jax.scipy.linalg.cho_solve(cholesky_tuples, ys)
    norm_sq = ys.T @ first_step
    norm = jnp.sqrt(norm_sq)

    print(f'Norm: {norm}')
