"""
Masks to apply to images in real space.
"""

from typing import overload

import jax
import jax.numpy as jnp
from equinox import field
from jaxtyping import Array, Float

from ._operator import AbstractImageMultiplier


class AbstractMask(AbstractImageMultiplier, strict=True):
    """Base class for computing and applying an image mask."""

    @overload
    def __call__(
        self, image: Float[Array, "y_dim x_dim"]
    ) -> Float[Array, "y_dim x_dim"]: ...

    @overload
    def __call__(
        self, image: Float[Array, "z_dim y_dim x_dim"]
    ) -> Float[Array, "z_dim y_dim x_dim"]: ...

    def __call__(
        self, image: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        return image * jax.lax.stop_gradient(self.buffer)


class CustomMask(AbstractMask, strict=True):
    """Pass a custom mask as an array."""

    buffer: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]

    def __init__(
        self, mask: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ):
        self.buffer = mask


class CircularCosineMask(AbstractMask, strict=True):
    """Apply a circular mask to an image with a cosine
    soft-edge.
    """

    buffer: Float[Array, "y_dim x_dim"]

    radius_in_angstroms_or_pixels: float = field(static=True)
    rolloff_width_in_angstroms_or_pixels: float = field(static=True)

    def __init__(
        self,
        coordinate_grid_in_angstroms_or_pixels: Float[Array, "y_dim x_dim 2"],
        radius_in_angstroms_or_pixels: float,
        rolloff_width_in_angstroms_or_pixels: float,
    ):
        """**Arguments:**

        - `coordinate_grid_in_angstroms_or_pixels`:
            The image coordinates.
        - `grid_spacing`:
            The pixel or voxel size of `coordinate_grid_in_angstroms_or_pixels`.
        - `radius_in_angstroms_or_pixels`:
            The radius of the circular mask.
        - `rolloff_width_in_angstroms_or_pixels`:
            The rolloff width of the soft edge.
        """
        self.radius_in_angstroms_or_pixels = radius_in_angstroms_or_pixels
        self.rolloff_width_in_angstroms_or_pixels = rolloff_width_in_angstroms_or_pixels
        self.buffer = _compute_circular_or_spherical_mask(
            coordinate_grid_in_angstroms_or_pixels,
            self.radius_in_angstroms_or_pixels,
            self.rolloff_width_in_angstroms_or_pixels,
        )


class SquareCosineMask(AbstractMask, strict=True):
    """Apply a square mask to an image with a cosine
    soft-edge.
    """

    buffer: Float[Array, "y_dim x_dim"]

    side_length_in_angstroms_or_pixels: float = field(static=True)
    rolloff_width_in_angstroms_or_pixels: float = field(static=True)

    def __init__(
        self,
        coordinate_grid_in_angstroms_or_pixels: Float[Array, "y_dim x_dim 2"],
        side_length_in_angstroms_or_pixels: float,
        rolloff_width_in_angstroms_or_pixels: float,
    ):
        """**Arguments:**

        - `coordinate_grid_in_angstroms_or_pixels`:
            The image coordinates.
        - `grid_spacing`:
            The pixel or voxel size of `coordinate_grid_in_angstroms_or_pixels`.
        - `side_length_in_angstroms_or_pixels`:
            The side length of the square.
        - `rolloff_width_in_angstroms_or_pixels`:
            The rolloff width of the soft edge.
        """
        self.side_length_in_angstroms_or_pixels = side_length_in_angstroms_or_pixels
        self.rolloff_width_in_angstroms_or_pixels = rolloff_width_in_angstroms_or_pixels
        self.buffer = _compute_square_mask(
            coordinate_grid_in_angstroms_or_pixels,
            self.side_length_in_angstroms_or_pixels,
            self.rolloff_width_in_angstroms_or_pixels,
        )


class SphericalCosineMask(AbstractMask, strict=True):
    """Apply a spherical mask to a volume with a cosine
    soft-edge.
    """

    buffer: Float[Array, "z_dim y_dim x_dim"]

    radius_in_angstroms_or_voxels: float = field(static=True)
    rolloff_width_in_angstroms_or_pixels: float = field(static=True)

    def __init__(
        self,
        coordinate_grid_in_angstroms_or_voxels: Float[Array, "z_dim y_dim x_dim 3"],
        radius_in_angstroms_or_voxels: float,
        rolloff_width_in_angstroms_or_pixels: float,
    ):
        """**Arguments:**

        - `coordinate_grid_in_angstroms_or_voxels`:
            The volume coordinates.
        - `grid_spacing`:
            The pixel or voxel size of `coordinate_grid_in_angstroms_or_voxels`.
        - `radius_in_angstroms_or_voxels`:
            The radius of the spherical mask.
        - `rolloff_width_in_angstroms_or_pixels`:
            The rolloff width of the soft edge.
        """
        self.radius_in_angstroms_or_voxels = radius_in_angstroms_or_voxels
        self.rolloff_width_in_angstroms_or_pixels = rolloff_width_in_angstroms_or_pixels
        self.buffer = _compute_circular_or_spherical_mask(
            coordinate_grid_in_angstroms_or_voxels,
            self.radius_in_angstroms_or_voxels,
            self.rolloff_width_in_angstroms_or_pixels,
        )


@overload
def _compute_circular_or_spherical_mask(
    coordinate_grid: Float[Array, "y_dim x_dim 2"],
    radius: float,
    rolloff_width: float,
) -> Float[Array, "y_dim x_dim"]: ...


@overload
def _compute_circular_or_spherical_mask(
    coordinate_grid: Float[Array, "z_dim y_dim x_dim 3"],
    radius: float,
    rolloff_width: float,
) -> Float[Array, "z_dim y_dim x_dim"]: ...


def _compute_circular_or_spherical_mask(
    coordinate_grid: Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"],
    radius: float,
    rolloff_width: float,
) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
    radial_coordinate_grid = jnp.linalg.norm(coordinate_grid, axis=-1)

    def compute_mask_at_coordinate(radial_coordinate):
        return jnp.where(
            radial_coordinate <= radius,
            1.0,
            jnp.where(
                radial_coordinate > radius + rolloff_width,
                0.0,
                0.5
                * (
                    1
                    + jnp.cos(jnp.pi * (radial_coordinate_grid - radius) / rolloff_width)
                ),
            ),
        )

    compute_mask = (
        jax.vmap(jax.vmap(compute_mask_at_coordinate))
        if radial_coordinate_grid.ndim == 2
        else jax.vmap(jax.vmap(jax.vmap(compute_mask_at_coordinate)))
    )

    return compute_mask(radial_coordinate_grid)


def _compute_square_mask(
    coordinate_grid: Float[Array, "y_dim x_dim 2"],
    side_length: float,
    rolloff_width: float,
) -> Float[Array, "y_dim x_dim"]:
    is_in_square_fn = lambda x, y, s: jnp.logical_and(
        jnp.abs(x) <= s / 2, jnp.abs(y) <= s / 2
    )
    is_in_edge_fn = lambda x_or_y, s, w: jnp.logical_and(
        jnp.abs(x_or_y) > s / 2, jnp.abs(x_or_y) < s / 2 + w
    )
    compute_edge_fn = lambda x_or_y, s, w: 0.5 * (
        1 + jnp.cos(jnp.pi * (x_or_y - s / 2) / w)
    )

    def compute_mask_at_coordinate(coordinate):
        x, y = coordinate
        # Check coordinate is in either the square of the unmasked region
        is_in_unmasked_square = is_in_square_fn(x, y, side_length)
        # ... or the square of the unmasked region, plus the rolloff width
        # of the soft edge
        is_in_unmasked_plus_soft_edge_square = is_in_square_fn(
            x, y, side_length + 2 * rolloff_width
        )
        # Next, compute where (if anywhere) the coordinate is in the soft edge
        # region
        is_in_edge_x = is_in_edge_fn(x, side_length, rolloff_width)
        is_in_edge_y = is_in_edge_fn(y, side_length, rolloff_width)
        # Compute the soft edges
        edge_x, edge_y = (
            compute_edge_fn(x, side_length, rolloff_width),
            compute_edge_fn(y, side_length, rolloff_width),
        )

        return jnp.where(
            is_in_unmasked_square,
            1.0,
            jnp.where(
                is_in_unmasked_plus_soft_edge_square,
                jnp.where(
                    jnp.logical_and(is_in_edge_x, is_in_edge_y),
                    edge_x * edge_y,
                    jnp.where(is_in_edge_x, edge_x, edge_y),
                ),
                0.0,
            ),
        )

    compute_mask = jax.vmap(jax.vmap(compute_mask_at_coordinate))

    return compute_mask(coordinate_grid)
