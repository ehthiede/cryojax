"""Pytrees that represent particle stacks."""

from equinox import Module, AbstractVar, field
from jaxtyping import Shaped

import jax.numpy as jnp

from ..inference.distributions import AbstractDistribution
from ..typing import Image


class AbstractParticleStack(Module, strict=True):
    """An abstraction of a particle stack, represented by a pytree
    that stores an image stack.

    Subclasses should also include parameters included in the
    image formation model, typically represented with `cryojax` objects.
    """

    image_stack: AbstractVar[Shaped[Image, "batch_dim"]]


class CryojaxParticleStack(AbstractParticleStack, strict=True):
    """The standard particle stack supported by `cryojax`."""

    image_stack: Shaped[Image, "batch_dim"] = field(converter=jnp.asarray)
    distribution: AbstractDistribution


CryojaxParticleStack.__init__.__doc__ = """**Arguments:**

- `image_stack`: The stack of images. The shape of this array
                 is a leading batch dimension followed by the shape
                 of an image in the stack.
- `distribution`: The statistical model from which the data is generated.
                  Any subset of pytree leaves may have a batch dimension.
"""
