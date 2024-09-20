from trajax.optimizers import ILQRHyperparams, ILQR
import time
import matplotlib.pyplot as plt


from smbrl.envs.pendulum import PendulumEnv
from smbrl.envs.cartpole_lenart import CartPoleEnv
from brax.envs.base import State
import jax.numpy as jnp
import jax.random as jr

env = CartPoleEnv()
action_repeat = 1
mpc_horizon = 20
total_horizon = 70


def dynamics_fn(x, u, t, length):
    assert x.shape == (env.observation_size,) and u.shape == (env.action_size,)
    state = State(pipeline_state=None,
                  obs=x,
                  reward=jnp.array(0.0),
                  done=jnp.array(0.0), )
    for _ in range(action_repeat):
        state = env.step(state, u)
    return state.obs


def cost_fn(x, u, t, length):
    assert x.shape == (env.observation_size,) and u.shape == (env.action_size,)
    state = State(pipeline_state=None,
                  obs=x,
                  reward=jnp.array(0.0),
                  done=jnp.array(0.0), )
    for _ in range(action_repeat):
        state = env.step(state, u)
    return -state.reward


ilqr = ILQR(cost_fn, dynamics_fn)

ts = jnp.arange(0, total_horizon)

ilqr_params = ILQRHyperparams(maxiter=100, make_psd=False)

dynamics_params = jnp.array(5.0)
cost_params = dynamics_params

obs = env.reset(rng=jr.PRNGKey(0)).obs
predicted_actions = jnp.zeros(shape=(mpc_horizon, env.action_size,))

all_obs = []
all_actions = []
all_rewards = []

times = []

for i in range(total_horizon // action_repeat):
    start_time = time.time()
    # Compute the action
    out = ilqr.solve(dynamics_params, cost_params, obs, predicted_actions, ilqr_params)
    predicted_actions = out.us
    action = predicted_actions[0]
    predicted_actions = jnp.concatenate([predicted_actions[1:], predicted_actions[-1].reshape(-1, 1)])

    for _ in range(action_repeat):
        obs = dynamics_fn(obs, action, ts[i], dynamics_params)
        reward = -cost_fn(obs, action, ts[i], dynamics_params)

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
