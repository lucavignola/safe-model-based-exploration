"""Confidence widths for the theory-aligned GP experiments.

The ``bsm`` GP model has a convenient ``beta=None`` mode, but its coefficient
does not exactly match the confidence coefficient used in the SBSRL paper.  In
particular, the paper uses

    beta_n(delta) =
        B + sigma_w * sqrt(2 * (gamma_n + 1 + log(d_x / delta))).

This module keeps that distinction explicit.  It also provides a deliberately
named *finite-design* RKHS-norm estimate.  Such an estimate is useful for
calibration, but it is not a certified upper bound on the RKHS norm over the
continuous state-action domain.
"""

from __future__ import annotations

from typing import Literal

import chex
import jax.numpy as jnp
from bsm.statistical_model import GPStatisticalModel
from bsm.utils.normalization import Data
from bsm.utils.type_aliases import StatisticalModelState
from jax import vmap


InformationGainBound = Literal["diagonal", "observed"]


def _as_output_vector(
        value: float | chex.Array,
        output_dim: int,
        *,
        name: str,
) -> chex.Array:
    """Converts a scalar or output-wise value to shape ``(output_dim,)``."""

    array = jnp.asarray(value)
    if array.shape == ():
        array = jnp.full((output_dim,), array)
    if array.shape != (output_dim,):
        raise ValueError(
            f"{name} must be scalar or have shape ({output_dim},), "
            f"got {array.shape}."
        )
    return array


def theorem_confidence_beta(
        *,
        rkhs_norm_bound: float | chex.Array,
        noise_std: float | chex.Array,
        information_gain_bound: float | chex.Array,
        output_dim: int,
        delta: float,
) -> chex.Array:
    """Evaluates the confidence coefficient appearing in the SBSRL theorem.

    All quantities must be expressed in the same GP output coordinates.  For a
    normalized GP this means that ``rkhs_norm_bound`` and ``noise_std`` are for
    the normalized output, while the resulting dimensionless beta multiplies
    the (subsequently denormalized) posterior standard deviation.
    """

    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta}.")

    rkhs_norm_bound = _as_output_vector(
        rkhs_norm_bound, output_dim, name="rkhs_norm_bound"
    )
    noise_std = _as_output_vector(noise_std, output_dim, name="noise_std")
    information_gain_bound = _as_output_vector(
        information_gain_bound, output_dim, name="information_gain_bound"
    )

    if bool(jnp.any(rkhs_norm_bound < 0)):
        raise ValueError("rkhs_norm_bound must be non-negative.")
    if bool(jnp.any(noise_std <= 0)):
        raise ValueError("noise_std must be strictly positive.")
    if bool(jnp.any(information_gain_bound < 0)):
        raise ValueError("information_gain_bound must be non-negative.")

    log_union_bound = jnp.log(output_dim / delta)
    return rkhs_norm_bound + noise_std * jnp.sqrt(
        2.0 * (information_gain_bound + 1.0 + log_union_bound)
    )


def diagonal_information_gain_bound(
        *,
        num_observations: int,
        noise_std: float | chex.Array,
        output_dim: int,
        kernel_diagonal_bound: float | chex.Array = 1.0,
) -> chex.Array:
    """A generic maximum-information-gain upper bound.

    If ``k(z, z) <= kappa_squared``, Hadamard's inequality gives

        gamma_N <= N/2 log(1 + kappa_squared / sigma_w**2).

    This bound is valid for any set of ``N`` inputs, unlike the observed
    log-determinant at the particular collected inputs.  It can be very
    conservative; that conservatism is intentional in the theorem mode.
    """

    if num_observations < 0:
        raise ValueError("num_observations must be non-negative.")

    noise_std = _as_output_vector(noise_std, output_dim, name="noise_std")
    kernel_diagonal_bound = _as_output_vector(
        kernel_diagonal_bound,
        output_dim,
        name="kernel_diagonal_bound",
    )
    if bool(jnp.any(noise_std <= 0)):
        raise ValueError("noise_std must be strictly positive.")
    if bool(jnp.any(kernel_diagonal_bound < 0)):
        raise ValueError("kernel_diagonal_bound must be non-negative.")

    return 0.5 * num_observations * jnp.log1p(
        kernel_diagonal_bound / jnp.square(noise_std)
    )


def observed_information_gain(
        model: GPStatisticalModel,
        stats_model_state: StatisticalModelState,
        data: Data,
) -> chex.Array:
    """Computes the information gain at the *observed* input design.

    This is ``0.5 log det(I + K_X / sigma_w**2)`` in the fixed GP's
    normalized coordinates.  It is useful diagnostically, but it is not the
    maximum information gain over all input sets and is therefore not the
    conservative default used by :class:`TheoremGPStatisticalModel`.
    """

    gp_state = stats_model_state.model_state
    if model.normalize:
        inputs = vmap(
            model.model.normalizer.normalize, in_axes=(0, None)
        )(data.inputs, gp_state.data_stats.inputs)
        noise_std = model.model.normalizer.normalize_std(
            model.model.output_stds, gp_state.data_stats.outputs
        )
    else:
        inputs = data.inputs
        noise_std = model.model.output_stds

    covariance = model.model.m_kernel_multiple_output(
        inputs, inputs, gp_state.params
    )
    identity = jnp.eye(inputs.shape[0], dtype=covariance.dtype)
    scaled_covariance = (
        covariance / jnp.square(noise_std)[:, None, None]
        + identity[None, :, :]
    )
    signs, log_determinants = vmap(jnp.linalg.slogdet)(scaled_covariance)
    if bool(jnp.any(signs <= 0)):
        raise ValueError(
            "Information-gain matrix must be positive definite."
        )
    return 0.5 * log_determinants


def empirical_finite_design_rkhs_norm(
        model: GPStatisticalModel,
        stats_model_state: StatisticalModelState,
        calibration_data: Data,
        *,
        jitter: float = 1e-8,
        safety_factor: float = 1.0,
) -> chex.Array:
    """Estimates output-wise RKHS norms on a finite calibration design.

    The returned value is

        safety_factor * sqrt(y_X.T @ (K_X + jitter I)^-1 @ y_X)

    in the fixed GP's own normalized coordinates.  This is the minimum RKHS
    norm of a function interpolating the finite design (up to ``jitter``).
    It generally *underestimates* the norm required over the continuous
    domain, so callers must log it as an empirical calibration value rather
    than a theorem-certified bound.
    """

    if calibration_data.inputs.shape[0] == 0:
        raise ValueError(
            "Cannot estimate an RKHS norm from an empty calibration design; "
            "provide an explicit prior bound B instead."
        )
    if jitter <= 0:
        raise ValueError("jitter must be strictly positive.")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least one.")

    gp_state = stats_model_state.model_state
    if model.normalize:
        inputs = vmap(
            model.model.normalizer.normalize, in_axes=(0, None)
        )(calibration_data.inputs, gp_state.data_stats.inputs)
        outputs = vmap(
            model.model.normalizer.normalize, in_axes=(0, None)
        )(calibration_data.outputs, gp_state.data_stats.outputs)
    else:
        inputs = calibration_data.inputs
        outputs = calibration_data.outputs

    covariance = model.model.m_kernel_multiple_output(
        inputs, inputs, gp_state.params
    )
    regularized_covariance = covariance + jitter * jnp.eye(
        inputs.shape[0], dtype=covariance.dtype
    )[None, :, :]

    def one_output_norm(kernel_matrix, function_values):
        coefficients = jnp.linalg.solve(kernel_matrix, function_values)
        squared_norm = jnp.dot(function_values, coefficients)
        return jnp.sqrt(jnp.maximum(squared_norm, 0.0))

    norms = vmap(one_output_norm, in_axes=(0, 1))(
        regularized_covariance, outputs
    )
    return safety_factor * norms


class TheoremGPStatisticalModel(GPStatisticalModel):
    """A fixed-coordinate GP using the paper's confidence coefficient.

    ``information_gain_bound="diagonal"`` uses a valid maximum-information
    gain bound based only on the number of observations and a bound on the
    kernel diagonal.  ``"observed"`` is provided for diagnostics/ablations and
    must not be presented as the same maximum-information-gain certificate.

    Kernel hyperparameters and normalization statistics are required to be
    fixed.  Otherwise a fixed prior sample and the meaning of ``B`` change
    during learning.
    """

    def __init__(
            self,
            *args,
            f_norm_bound: float | chex.Array,
            delta: float = 0.05,
            information_gain_bound: InformationGainBound = "diagonal",
            kernel_diagonal_bound: float | chex.Array = 1.0,
            fixed_kernel_params: bool = True,
            **kwargs,
    ):
        if not fixed_kernel_params:
            raise ValueError(
                "The theorem beta mode requires fixed kernel parameters."
            )
        if information_gain_bound not in ("diagonal", "observed"):
            raise ValueError(
                "information_gain_bound must be 'diagonal' or 'observed'."
            )
        if kwargs.get("normalization_stats") is None:
            raise ValueError(
                "The theorem beta mode requires fixed normalization_stats "
                "(use unit statistics when normalize=False)."
            )

        output_dim = kwargs.get("output_dim")
        if output_dim is None:
            raise ValueError("output_dim must be supplied as a keyword argument.")
        self.theorem_f_norm_bound = _as_output_vector(
            f_norm_bound, output_dim, name="f_norm_bound"
        )
        self.information_gain_bound = information_gain_bound
        self.kernel_diagonal_bound = _as_output_vector(
            kernel_diagonal_bound,
            output_dim,
            name="kernel_diagonal_bound",
        )

        # beta=None delegates every post-data update to our compute_beta().
        kwargs.pop("beta", None)
        super().__init__(
            *args,
            f_norm_bound=self.theorem_f_norm_bound,
            delta=delta,
            beta=None,
            fixed_kernel_params=True,
            **kwargs,
        )

    def init(self, key: chex.PRNGKey) -> StatisticalModelState:
        state = super().init(key)
        empty_history = Data(
            inputs=jnp.zeros(
                (0, self.input_dim),
                dtype=state.model_state.history.inputs.dtype,
            ),
            outputs=jnp.zeros(
                (0, self.output_dim),
                dtype=state.model_state.history.outputs.dtype,
            ),
        )
        gp_state = state.model_state.replace(
            history=empty_history,
            data_stats=self.normalization_stats,
            alphas=jnp.zeros(
                (self.output_dim, 0),
                dtype=state.model_state.history.outputs.dtype,
            ),
        )
        # The paper defines beta_0 = B for the unconditioned prior.
        return state.replace(
            model_state=gp_state,
            beta=self.theorem_f_norm_bound,
        )

    def compute_beta(
            self,
            model_state,
            data: Data,
    ) -> chex.Array:
        if self.normalize:
            noise_std = self.model.normalizer.normalize_std(
                self.model.output_stds, model_state.data_stats.outputs
            )
        else:
            noise_std = self.model.output_stds

        temporary_state = StatisticalModelState(
            model_state=model_state,
            beta=self.theorem_f_norm_bound,
        )
        if self.information_gain_bound == "diagonal":
            gamma_bound = diagonal_information_gain_bound(
                num_observations=data.inputs.shape[0],
                noise_std=noise_std,
                output_dim=self.output_dim,
                kernel_diagonal_bound=self.kernel_diagonal_bound,
            )
        else:
            gamma_bound = observed_information_gain(
                self, temporary_state, data
            )

        return theorem_confidence_beta(
            rkhs_norm_bound=self.theorem_f_norm_bound,
            noise_std=noise_std,
            information_gain_bound=gamma_bound,
            output_dim=self.output_dim,
            delta=self.delta,
        )
