import time
from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt


from smbrl.envs.racecar import RaceCarSystem, RaceCarEnv
from smbrl.optimizer.icem import iCemTO, iCemParams, AbstractCost



if __name__ == '__main__':
    from jax import jit


    action_repeat = 1
    horizon = 50
    optimizer = iCemTO(
        horizon=horizon,
        action_dim=1,
        key=jr.PRNGKey(0),
        opt_params=iCemParams(exponent=1.0,
                              num_samples=500,
                              num_elites=50,
                              alpha=0.2,
                              num_steps=5,
                              num_particles=1, ),
        system=RaceCarSystem(),
        cost_fn=None,
    )

    system = RaceCarSystem()

    optimizer_state = optimizer.init(key=jr.PRNGKey(1))
    system_params = system.init_params(key=jr.PRNGKey(2))
    obs = RaceCarEnv().reset(jr.PRNGKey(0)).obs

    all_obs = []
    all_actions = []
    all_rewards = []

    times = []
    first_times = time.time()

    step_fn = jit(system.step)
    act_fn = jit(optimizer.act)

    for i in range(200 // action_repeat):
        start_time = time.time()
        action, optimizer_state = act_fn(obs, optimizer_state)
        for _ in range(action_repeat):
            sys_state = step_fn(obs, action, system_params)
            obs, reward, system_params = sys_state.x_next, sys_state.reward, sys_state.system_params
        all_obs.append(obs)
        all_actions.append(action)
        all_rewards.append(reward)
        end_time = time.time()
        times.append(end_time - start_time)

    print(f'Total time {time.time() - first_times:.2f}')

    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    all_obs = jnp.array(all_obs)
    for i in range(len(all_obs[0])):
        axs[0].plot(all_obs[:, i], label=f'State {i}')
    axs[0].set_title('Observation')
    axs[0].legend(fontsize=5)

    axs[1].plot(all_actions)
    axs[1].set_title('Action')
    axs[2].plot(all_rewards)
    axs[2].set_title('Reward')
    axs[3].plot(times[2:])
    axs[3].set_title('Time')
    plt.tight_layout()
    plt.show()

    print(f'Maximal x position: {jnp.max(jnp.stack(all_obs)[:, 0])}')
    print(f'Minimal x position: {jnp.min(jnp.stack(all_obs)[:, 0])}')

    import numpy as np

    total_reward = np.sum(np.array(all_rewards))
    print(f'Total reward: {total_reward}')