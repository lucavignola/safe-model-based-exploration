import os
from typing import NamedTuple
from jaxtyping import Float, Array
import jax.numpy as jnp

import chex
from brax.envs import State


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")


class ExplorationTrajectory(NamedTuple):
    states: State
    actions: chex.Array
    rewards: chex.Array


def decode_angles(obs: Float[Array, '3']) -> Float[Array, '2']:
    th = jnp.arctan2(obs[1], obs[0])
    omega = obs[2]
    return jnp.array([th, omega])
