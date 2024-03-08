"""
Image formation models simulated from gaussian noise distributions.
"""

from typing import Optional
from typing_extensions import override
from equinox import field

import numpy as np
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from ._distribution import AbstractDistribution
from ...image.operators import FourierOperatorLike, Constant
from ...simulator import AbstractPipeline
from ...typing import Real_, Image, ComplexImage


class IndependentFourierGaussian(AbstractDistribution, strict=True):
    r"""A gaussian noise model, where each fourier mode is independent.

    This computes the likelihood in Fourier space,
    so that the variance to be an arbitrary noise power spectrum.

    **Attributes:**

    - `pipeline`: The image formation model.

    - `variance`: The variance of each fourier mode.

    - `contrast_scale`: The standard deviation of an image simulated
                        from `pipeline`, excluding the noise.
    """

    pipeline: AbstractPipeline
    variance: FourierOperatorLike
    contrast_scale: Real_ = field(converter=jnp.asarray)

    def __init__(
        self,
        pipeline: AbstractPipeline,
        variance: Optional[FourierOperatorLike] = None,
        contrast_scale: Optional[Real_] = None,
    ):
        self.pipeline = pipeline
        self.variance = variance or Constant(1.0)
        self.contrast_scale = contrast_scale or jnp.asarray(1.0)

    @override
    def render(self, *, get_real: bool = True) -> Image:
        """Render the image formation model."""
        return self.contrast_scale * self.pipeline.render(
            normalize=True, get_real=get_real
        )

    @override
    def sample(self, key: PRNGKeyArray, *, get_real: bool = True) -> Image:
        """Sample from the gaussian noise model."""
        N_pix = np.prod(self.pipeline.integrator.config.padded_shape)
        freqs = self.pipeline.integrator.config.padded_frequency_grid_in_angstroms.get()
        # Compute the zero mean variance and scale up to be independent of the number of pixels
        std = jnp.sqrt(N_pix * self.variance(freqs))
        noise = self.pipeline.crop_and_apply_operators(
            std * jr.normal(key, shape=freqs.shape[0:-1]).at[0, 0].set(0.0),
            get_real=get_real,
        )
        image = self.render(get_real=get_real)
        return image + noise

    @override
    def log_likelihood(self, observed: ComplexImage) -> Real_:
        """Evaluate the log-likelihood of the gaussian noise model.

        **Arguments:**

        `observed` : The observed data in fourier space. `observed.shape`
                     must match `ImageConfig.padded_shape`.
        """
        N_pix = np.prod(self.pipeline.integrator.config.padded_shape)
        freqs = self.pipeline.integrator.config.frequency_grid_in_angstroms.get()
        # Compute the variance and scale up to be independent of the number of pixels
        variance = N_pix * self.variance(freqs)
        # Create simulated data
        simulated = self.render(get_real=False)
        # Compute residuals
        residuals = simulated - observed
        # Compute standard normal random variables
        squared_standard_normal_per_mode = jnp.abs(residuals) ** 2 / (2 * variance)
        # Compute the log-likelihood for each fourier mode. Divide by the
        # number of pixels so that the likelihood is a sum over pixels in
        # real space (parseval's theorem)
        log_likelihood_per_mode = (
            squared_standard_normal_per_mode - jnp.log(2 * jnp.pi * variance) / 2
        ) / N_pix
        # Compute log-likelihood, throwing away the zero mode. Need to take care
        # to compute the loss function in fourier space for a real-valued function.
        log_likelihood = -1.0 * (
            jnp.sum(log_likelihood_per_mode[1:, 0])
            + 2 * jnp.sum(log_likelihood_per_mode[:, 1:])
        )

        return log_likelihood