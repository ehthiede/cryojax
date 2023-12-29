"""
Routines for interfacing with mdtraj
"""

import numpy as np
from cryojax.io.load_atoms import get_form_factor_params

__all__ = [
    "get_scattering_info_from_mdtraj",
    "mdtraj_load_from_file",
]


def get_scattering_info_from_mdtraj(traj):
    """
    Gets the scattering information from an mdtraj trajectory.

    Parameters
    ----------
    traj : mdtraj trajectory

    Returns
    -------
    atom_positions : np.array
        Atomic positions.
    a_vals : np.array
        Atomic form factor parameters a.
    b_vals : np.array
        Atomic form factor parameters b.
    """
    atom_element_names = [a.element.symbol for a in traj.top.atoms]
    atom_positions = traj.xyz

    a_vals, b_vals = get_form_factor_params(atom_element_names)
    return np.array(atom_positions), np.array(a_vals), np.array(b_vals)


def mdtraj_load_from_file(path: str, top=None):
    """
    Loads
    """
    import mdtraj as md

    traj = md.load(path, top=top)
    return get_scattering_info_from_mdtraj(traj)
