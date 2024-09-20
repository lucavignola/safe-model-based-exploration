from trajax.optimizers import ILQRHyperparams, ILQR
import time
import matplotlib.pyplot as plt

from smbrl.envs.pendulum import PendulumEnv
from smbrl.envs.cartpole_lenart import CartPoleEnv
from brax.envs.base import State
import jax.numpy as jnp
import jax.random as jr
from jax.lax import scan
from jax import jit, config, grad
from cyipopt import minimize_ipopt

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

env = CartPoleEnv()
action_repeat = 2
mpc_horizon = 10
total_horizon = 100


def step_fn(obs, action):
    assert obs.shape == (env.observation_size,) and action.shape == (env.action_size,)
    state = State(pipeline_state=None,
                  obs=obs,
                  reward=jnp.array(0.0),
                  done=jnp.array(0.0), )
    for _ in range(action_repeat):
        state = env.step(state, action)
    return state.obs, state.reward


def objective(x, init_obs):
    actions = x.reshape(mpc_horizon, env.action_size)
    assert init_obs.shape == (env.observation_size,)

    final_obs, rewards = scan(step_fn, init_obs, actions)
    return -jnp.sum(rewards)


obj_jit = jit(objective)
obj_grad = jit(grad(obj_jit))  # objective gradient

obs = env.reset(rng=jr.PRNGKey(0)).obs
predicted_actions = jr.uniform(key=jr.PRNGKey(0), shape=(mpc_horizon, env.action_size,), minval=-1.0, maxval=1.0)

all_obs = []
all_actions = []
all_rewards = []

times = []


def pack_optimization_vector(obs, predicted_actions):
    return predicted_actions.reshape(-1)


def unpack_optimization_vector(x):
    actions = x.reshape(mpc_horizon, env.action_size)
    return actions


bnds = [(-1, 1) for _ in range(mpc_horizon)]

for i in range(total_horizon // action_repeat):
    start_time = time.time()
    # Compute the action
    x = pack_optimization_vector(obs, predicted_actions)
    out = minimize_ipopt(obj_jit, jac=obj_grad, x0=x, options={'disp': 0, 'maxiter': 100}, args=(obs), bounds=bnds)
    predicted_actions = unpack_optimization_vector(out.x)
    action = predicted_actions[0]

    for _ in range(action_repeat):
        obs, reward = step_fn(obs, action)
    predicted_actions = jnp.concatenate([predicted_actions[1:], predicted_actions[-1].reshape(1, -1)])

    all_obs.append(obs)
    all_actions.append(action)
    all_rewards.append(reward)
    end_time = time.time()
    times.append(end_time - start_time)

fig, axs = plt.subplots(1, 4, figsize=(8, 2))
axs[0].plot(all_obs)
axs[0].set_title('Observation')
axs[1].plot(all_actions)
axs[1].set_title('Action')
axs[2].plot(all_rewards)
axs[2].set_title('Reward')
axs[3].plot(times[2:])
axs[3].set_title('Time')
plt.tight_layout()
plt.show()

print(f'Maximal velocity value: {jnp.max(jnp.stack(all_obs)[:, -1])}')
print(f'Minimal velocity value: {jnp.min(jnp.stack(all_obs)[:, -1])}')

import numpy as np

total_reward = np.sum(np.array(all_rewards))
print(f'Total reward: {total_reward}')
