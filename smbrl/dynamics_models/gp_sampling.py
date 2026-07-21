"""Pathwise sampling utilities for the Gaussian-process dynamics model.

The posterior sampler implements the decoupled/Matheron construction

    f_post(z) = g(z) + k(z, X) (K + Sigma)^{-1}
                         (y - g(X) - epsilon),

where ``g`` is approximated with random Fourier features.  A sampled state is
therefore a global function that can be evaluated consistently at arbitrary
state-action inputs during an iCEM optimization.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softplus


@chex.dataclass
class RFFPosteriorState:
    """Arrays defining a bank of approximate GP posterior paths.

    The output GPs are independent and have output-specific spectral
    frequencies.  Frequencies are shared across paths, while the cosine/sine
    weights and observation-noise draws are independent for each path.
    """

    normalized_history_inputs: chex.Array
    frequencies: chex.Array
    cosine_weights: chex.Array
    sine_weights: chex.Array
    correction_weights: chex.Array

    @property
    def num_paths(self) -> int:
        return self.cosine_weights.shape[1]

    @property
    def num_features(self) -> int:
        return self.frequencies.shape[1]


def _normalization_arrays(model_state):
    gp_state = model_state.model_state
    input_stats = gp_state.data_stats.inputs
    output_stats = gp_state.data_stats.outputs
    normalized_inputs = (
        gp_state.history.inputs - input_stats.mean
    ) / input_stats.std
    normalized_outputs = (
        gp_state.history.outputs - output_stats.mean
    ) / output_stats.std
    return normalized_inputs, normalized_outputs


def _validate_rff_compatible_model(model, model_state) -> None:
    if not hasattr(model, "model") or not hasattr(model.model, "m_kernel_multiple_output"):
        raise TypeError("RFF sampling requires a GPStatisticalModel-compatible model.")

    params = model_state.model_state.params
    if "pseudo_length_scale" not in params:
        raise TypeError(
            "RFF sampling currently supports stationary ARD kernels with "
            "'pseudo_length_scale' parameters."
        )


def sample_rff_posterior(
        model,
        model_state,
        key: chex.PRNGKey,
        num_paths: int,
        num_features: int,
        jitter: float = 1e-8,
) -> RFFPosteriorState:
    """Draws a prefix-coupled bank of approximate GP posterior paths.

    All calculations happen in the GP's normalized input/output coordinates.
    ``fold_in(path_id)`` makes path ``m`` identical in experiments with
    different values of ``num_paths``, which is useful for controlled M sweeps.
    """

    if num_paths < 1:
        raise ValueError(f"num_paths must be positive, got {num_paths}.")
    if num_features < 1:
        raise ValueError(f"num_features must be positive, got {num_features}.")
    if jitter < 0:
        raise ValueError(f"jitter must be non-negative, got {jitter}.")

    _validate_rff_compatible_model(model, model_state)

    gp_state = model_state.model_state
    normalized_inputs, normalized_outputs = _normalization_arrays(model_state)
    num_data, input_dim = normalized_inputs.shape
    output_dim = normalized_outputs.shape[1]

    pseudo_length_scales = gp_state.params["pseudo_length_scale"]
    length_scales = softplus(pseudo_length_scales)
    if length_scales.shape != (output_dim, input_dim):
        raise ValueError(
            "Expected output-specific ARD length scales with shape "
            f"{(output_dim, input_dim)}, got {length_scales.shape}."
        )

    # The paired cosine/sine basis has exact unit prior variance at every z:
    # phi(z)w = D^-1/2 sum_d (a_d cos(w_d^T z) + b_d sin(w_d^T z)).
    frequency_key = jr.fold_in(key, 0)
    base_frequencies = jr.normal(
        frequency_key, shape=(output_dim, num_features, input_dim)
    )
    frequencies = base_frequencies / length_scales[:, None, :]

    path_ids = jnp.arange(num_paths, dtype=jnp.uint32)
    path_root = jr.fold_in(key, 1)
    path_keys = jax.vmap(lambda path_id: jr.fold_in(path_root, path_id))(path_ids)

    def sample_one_path(path_key):
        cosine_key, sine_key, noise_key = jr.split(path_key, 3)
        cosine_weights = jr.normal(
            cosine_key, shape=(output_dim, num_features)
        )
        sine_weights = jr.normal(
            sine_key, shape=(output_dim, num_features)
        )
        standard_noise = jr.normal(noise_key, shape=(output_dim, num_data))
        return cosine_weights, sine_weights, standard_noise

    cosine_weights, sine_weights, standard_noise = jax.vmap(sample_one_path)(
        path_keys
    )
    # Store as [output, path, feature/data] for cheap selection by path index.
    cosine_weights = jnp.swapaxes(cosine_weights, 0, 1)
    sine_weights = jnp.swapaxes(sine_weights, 0, 1)
    standard_noise = jnp.swapaxes(standard_noise, 0, 1)

    projections = jnp.einsum(
        "odi,ni->odn", frequencies, normalized_inputs
    )
    feature_scale = jnp.sqrt(
        jnp.asarray(1.0 / num_features, dtype=normalized_inputs.dtype)
    )
    prior_at_data = feature_scale * (
        jnp.einsum("odn,omd->omn", jnp.cos(projections), cosine_weights)
        + jnp.einsum("odn,omd->omn", jnp.sin(projections), sine_weights)
    )

    output_noise_std = model.model.output_stds / gp_state.data_stats.outputs.std
    observation_noise = output_noise_std[:, None, None] * standard_noise
    residuals = (
        jnp.swapaxes(normalized_outputs, 0, 1)[:, None, :]
        - prior_at_data
        - observation_noise
    )

    kernel_matrix = model.model.m_kernel_multiple_output(
        normalized_inputs, normalized_inputs, gp_state.params
    )
    eye = jnp.eye(num_data, dtype=kernel_matrix.dtype)
    noisy_kernel_matrix = kernel_matrix + (
        jnp.square(output_noise_std) + jitter
    )[:, None, None] * eye[None, :, :]

    # Solve all M right-hand sides using one factorization per output GP.
    correction_weights = jax.vmap(
        lambda matrix, output_residuals: jnp.linalg.solve(
            matrix, output_residuals.T
        ).T
    )(noisy_kernel_matrix, residuals)

    return RFFPosteriorState(
        normalized_history_inputs=normalized_inputs,
        frequencies=frequencies,
        cosine_weights=cosine_weights,
        sine_weights=sine_weights,
        correction_weights=correction_weights,
    )


def evaluate_rff_posterior(
        model,
        model_state,
        posterior_state: RFFPosteriorState,
        input_value: chex.Array,
        path_index: chex.Array,
) -> chex.Array:
    """Evaluates one fixed posterior path at an unnormalized input."""

    gp_state = model_state.model_state
    input_stats = gp_state.data_stats.inputs
    output_stats = gp_state.data_stats.outputs
    normalized_input = (input_value - input_stats.mean) / input_stats.std

    cosine_weights = posterior_state.cosine_weights[:, path_index, :]
    sine_weights = posterior_state.sine_weights[:, path_index, :]
    projections = jnp.einsum(
        "odi,i->od", posterior_state.frequencies, normalized_input
    )
    feature_scale = jnp.sqrt(
        jnp.asarray(
            1.0 / posterior_state.num_features,
            dtype=normalized_input.dtype,
        )
    )
    prior_value = feature_scale * (
        jnp.sum(jnp.cos(projections) * cosine_weights, axis=-1)
        + jnp.sum(jnp.sin(projections) * sine_weights, axis=-1)
    )

    kernel_to_data = model.model.v_kernel_multiple_output(
        posterior_state.normalized_history_inputs,
        normalized_input,
        gp_state.params,
    )
    correction = jnp.sum(
        kernel_to_data * posterior_state.correction_weights[:, path_index, :],
        axis=-1,
    )
    normalized_value = prior_value + correction
    return normalized_value * output_stats.std + output_stats.mean
