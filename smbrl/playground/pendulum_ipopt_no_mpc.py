import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from brax.envs.base import State
from cyipopt import minimize_ipopt
from jax import jit, config, grad, jacfwd, jacrev
from jax.lax import scan

from smbrl.envs.pendulum import PendulumEnv

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

env = PendulumEnv()
horizon = 100
init_obs = env.reset(rng=jr.PRNGKey(0)).obs


def step_fn(obs, action):
    assert obs.shape == (env.observation_size,) and action.shape == (env.action_size,)
    state = State(pipeline_state=None,
                  obs=obs,
                  reward=jnp.array(0.0),
                  done=jnp.array(0.0), )
    state = env.step(state, action)
    return state.obs, state.reward


def objective(x):
    actions = x.reshape(horizon, env.action_size)
    final_obs, rewards = scan(step_fn, init_obs, actions)
    return -jnp.sum(rewards)


obj_jit = jit(objective)
obj_grad = jit(grad(obj_jit))  # objective gradient
# obj_hess = jit(jacrev(jacfwd(obj_jit)))
obs = env.reset(rng=jr.PRNGKey(0)).obs
x = jnp.zeros(shape=(horizon,), dtype=jnp.float64)

bnds = [(-1, 1) for _ in range(horizon)]

out = minimize_ipopt(obj_jit, jac=obj_grad, x0=x, options={'disp': 5, 'maxiter': 200}, bounds=bnds)

plt.plot(out.x)
plt.show()
print(obj_jit(out.x))
print(obj_grad(out.x))
