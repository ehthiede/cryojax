"""
Transformations used for reparameterizing cryojax models. These classes are
to be used with equinox pytree manipulation utilities, like eqx.partition and
eqx.combine.
The purpose of these classes are to do inference in better geometries or to enforce
parameter constraints.
"""

from abc import abstractmethod
from equinox import Module, field
from jaxtyping import PyTree

import jax.numpy as jnp
import jax.tree_util as jtu

from ..typing import Real_


def _is_transformed(x):
    return isinstance(x, AbstractParameterTransform)


def _is_none(x):
    return x is None


def _apply_inverse_transform(x):
    if isinstance(x, AbstractParameterTransform):
        return x.get()
    else:
        return x


def _apply_transform(t, x):
    if x is None:
        return x
    else:
        # t is a method that returns an AbstractParameterTransform
        # instance
        return t(x)


def apply_inverse_transforms(pytree: PyTree):
    """Transforms a pytree whose parameters have entries
    that are `AbstractParameterTransform`s back to its
    original parameter space.
    """
    return jtu.tree_map(_apply_inverse_transform, pytree, is_leaf=_is_transformed)


def apply_transforms(
    pytree: PyTree,
    transforms: PyTree,
):
    """Applies a set of transformations to the jax arrays in
    a pytree. The pytree should have `None` values.
    """
    # Assumes that transforms is a prefix of pytree
    return jtu.tree_map(_apply_transform, transforms, pytree, is_leaf=_is_none)


class AbstractParameterTransform(Module, strict=True):
    """Base class for a parameter transformation.

    This is a very simple class. When the class is initialized,
    a parameter is taken to a transformed parameter space. When
    `transform.get()` is called, the parameter is taken back to
    the original parameter space.
    """

    @abstractmethod
    def get(self):
        """Get the parameter in the original parameter space."""
        raise NotImplementedError


class ExpTransform(AbstractParameterTransform, strict=True):
    """This class transforms a parameter to its logarithm.

    **Attributes:**

    - `log_parameter`: The parameter on a logarithmic scale.
    """

    log_parameter: Real_

    def __init__(self, parameter: Real_):
        """**Arguments:**

        - `parameter`: The parameter to be transformed to a logarithmic
                       scale.
        """
        self.log_parameter = jnp.log(parameter)

    def get(self):
        """The `log_parameter` transformed back with an exponential."""
        return jnp.exp(self.log_parameter)


class RescalingTransform(AbstractParameterTransform, strict=True):
    """This class transforms a parameter by a scale factor.

    **Attributes:**

    - `rescaled_parameter`: The rescaled parameter.
    """

    rescaled_parameter: Real_
    rescaling: float = field(static=True)

    def __init__(self, parameter: Real_, rescaling: float):
        """**Arguments:**

        - `parameter`: The parameter to be rescaled.

        - `rescaling`: The scale factor.
        """
        self.rescaled_parameter = rescaling * parameter
        self.rescaling = rescaling

    def get(self):
        """The rescaled `rescaled_parameter` transformed back to the original scale."""
        return self.rescaled_parameter / self.rescaling