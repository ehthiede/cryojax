import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.simulator.density import FourierVoxelGrid, RealVoxelGrid
from cryojax.simulator.density._voxel_density import (
    _build_real_space_voxels_from_atoms,
)
from cryojax.image import ifftn, make_coordinates
from jax import config

config.update("jax_enable_x64", True)


def test_VoxelGrid_agreement(sample_pdb_path):
    """
    Integration test ensuring that the VoxelGrid classes
    produce comparable electron densities when loaded from PDB.
    """
    n_voxels_per_side = (128, 128, 128)
    voxel_size = 0.5

    # Load the PDB file into a VoxelGrid
    vg = FourierVoxelGrid.from_pdb(
        sample_pdb_path,
        n_voxels_per_side=n_voxels_per_side,
        voxel_size=voxel_size,
    )
    # Since Voxelgrid is in Frequency space by default, we have to first
    # transform back into real space.
    vg_density = ifftn(jnp.fft.ifftshift(vg.weights)).real
    # The constructors each transpose in a unique way in order for
    # jax-finufft and the fourier slice theorem to match each other
    # and cisTEM. This operation undos the difference between both transposes
    vg_density = jnp.transpose(vg_density, axes=[1, 0, 2])
    # Ravel the grid
    vg_density = vg_density.ravel()

    vc = RealVoxelGrid.from_pdb(
        sample_pdb_path,
        n_voxels_per_side=n_voxels_per_side,
        voxel_size=voxel_size,
    )

    np.testing.assert_allclose(vg_density, vc.weights.ravel(), atol=1e-12)


class TestBuildRealSpaceVoxelsFromAtoms:
    @pytest.mark.parametrize("largest_atom", range(0, 3))
    def test_maxima_are_in_right_positions(
        self, toy_gaussian_cloud, largest_atom
    ):
        """
        Test that the maxima of the density are in the correct positions.
        """
        (
            atom_positions,
            ff_a,
            ff_b,
            n_voxels_per_side,
            voxel_size,
        ) = toy_gaussian_cloud
        ff_a[largest_atom] += 1.0
        coordinate_grid = make_coordinates(n_voxels_per_side, voxel_size)

        # Build the density
        density = _build_real_space_voxels_from_atoms(
            atom_positions, ff_a, ff_b, coordinate_grid
        )

        # Find the maximum
        maximum_index = jnp.argmax(density)
        maximum_position = coordinate_grid.reshape(-1, 3)[maximum_index]

        # Check that the maximum is in the correct position
        assert jnp.allclose(maximum_position, atom_positions[largest_atom])

    def test_integral_is_correct(self, toy_gaussian_cloud):
        """
        Test that the maxima of the density are in the correct positions.
        """
        (
            atom_positions,
            ff_a,
            ff_b,
            n_voxels_per_side,
            voxel_size,
        ) = toy_gaussian_cloud
        coordinate_grid = make_coordinates(n_voxels_per_side, voxel_size)

        # Build the density
        density = _build_real_space_voxels_from_atoms(
            atom_positions, ff_a, ff_b, coordinate_grid
        )

        integral = jnp.sum(density) * voxel_size**3
        assert jnp.isclose(integral, jnp.sum(ff_a))


class TestBuildVoxelsFromTrajectories:
    def test_indexing_matches_individual_calls(self, toy_gaussian_cloud):
        (
            atom_positions,
            ff_a,
            ff_b,
            n_voxels_per_side,
            voxel_size,
        ) = toy_gaussian_cloud
        second_set_of_positions = atom_positions + 1.0
        traj = np.stack([atom_positions, second_set_of_positions], axis=0)

        coordinate_grid = make_coordinates(n_voxels_per_side, voxel_size)

        # Build the trajectory $density
        elements = np.array([1, 1, 2, 6])

        traj_voxels = RealVoxelGrid.from_trajectory(
            traj, elements, voxel_size, coordinate_grid
        )

        voxel1 = RealVoxelGrid.from_atoms(
            atom_positions, elements, voxel_size, coordinate_grid
        )

        voxel2 = RealVoxelGrid.from_atoms(
            second_set_of_positions, elements, voxel_size, coordinate_grid
        )

        np.testing.assert_allclose(
            traj_voxels.weights[0], voxel1.weights, atol=1e-12
        )
        np.testing.assert_allclose(
            traj_voxels.weights[1], voxel2.weights, atol=1e-12
        )
