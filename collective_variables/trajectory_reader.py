"""
GROMACS Trajectory Reader

Provides interface to read GROMACS trajectory files (XTC, TRR) and topology files
using MDAnalysis library. Handles periodic boundary conditions and atom selection.
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCReader
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class TrajectoryReader:
    """
    Interface to GROMACS trajectory and topology files.

    Parameters:
        topology_file (str): Path to topology file (TPR, GRO, or PSF)
        trajectory_file (str): Path to trajectory file (XTC or TRR)
    """

    def __init__(self, topology_file: str, trajectory_file: str):
        """Initialize trajectory reader."""
        try:
            self.universe = mda.Universe(topology_file, trajectory_file)
            logger.info(f"Loaded universe: {self.universe.trajectory.n_frames} frames")
        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}")
            raise

        self.topology_file = topology_file
        self.trajectory_file = trajectory_file
        self.n_atoms = self.universe.atoms.n_atoms
        self.n_frames = self.universe.trajectory.n_frames

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get atomic coordinates for a specific frame.

        Parameters:
            frame_idx (int): Frame index (0-based)

        Returns:
            coordinates (np.ndarray): Atomic positions (N, 3) in Ångströms
        """
        self.universe.trajectory[frame_idx]
        return self.universe.atoms.positions.copy()

    def get_velocities(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get atomic velocities for a specific frame.

        Parameters:
            frame_idx (int): Frame index (0-based)

        Returns:
            velocities (np.ndarray): Atomic velocities (N, 3) or None if not available
        """
        self.universe.trajectory[frame_idx]

        try:
            return self.universe.atoms.velocities.copy()
        except (AttributeError, ValueError):
            return None

    def get_box(self, frame_idx: int) -> np.ndarray:
        """
        Get periodic box vectors for a specific frame.

        Parameters:
            frame_idx (int): Frame index (0-based)

        Returns:
            box (np.ndarray): Box vectors (3, 3) in Ångströms
        """
        self.universe.trajectory[frame_idx]
        return self.universe.dimensions[:3].copy()

    def get_time(self, frame_idx: int) -> float:
        """
        Get simulation time for a specific frame.

        Parameters:
            frame_idx (int): Frame index (0-based)

        Returns:
            time (float): Simulation time in picoseconds
        """
        self.universe.trajectory[frame_idx]
        return self.universe.trajectory.time

    def select_atoms(self, selection: str) -> np.ndarray:
        """
        Select atoms based on MDAnalysis selection string.

        Common selections:
          "name CL": Chloride ions
          "name O": Oxygen atoms (water oxygens)
          "residue SOL": Water molecules
          "residue SOL and name O": Water oxygens only

        Parameters:
            selection (str): MDAnalysis selection string

        Returns:
            indices (np.ndarray): Array of selected atom indices (0-based)
        """
        try:
            atoms = self.universe.select_atoms(selection)
            return atoms.indices
        except Exception as e:
            logger.error(f"Invalid selection string '{selection}': {e}")
            raise

    def get_atom_names(self) -> np.ndarray:
        """Get atom names for all atoms."""
        return self.universe.atoms.names.copy()

    def get_atom_residues(self) -> np.ndarray:
        """Get residue names for all atoms."""
        return self.universe.atoms.resnames.copy()

    def get_atom_indices_by_name(self, atom_name: str) -> np.ndarray:
        """Get indices of all atoms with a specific name."""
        return np.where(self.universe.atoms.names == atom_name)[0]

    def get_atom_indices_by_residue(self, residue_name: str) -> np.ndarray:
        """Get indices of all atoms in a specific residue type."""
        return np.where(self.universe.atoms.resnames == residue_name)[0]

    def iterate_frames(
        self, start: int = 0, stop: Optional[int] = None, step: int = 1
    ) -> Tuple[int, float, np.ndarray, np.ndarray]:
        """
        Iterate over trajectory frames.

        Yields:
            frame_idx (int): Frame index
            time (float): Simulation time (ps)
            coordinates (np.ndarray): Atomic positions (N, 3)
            box (np.ndarray): Box vectors (3, 3)
        """
        if stop is None:
            stop = self.n_frames

        for frame_idx in range(start, min(stop, self.n_frames), step):
            self.universe.trajectory[frame_idx]
            yield (
                frame_idx,
                self.universe.trajectory.time,
                self.universe.atoms.positions.copy(),
                self.universe.dimensions[:3].copy()
            )

    def get_atom_masses(self) -> np.ndarray:
        """Get atomic masses for all atoms."""
        return self.universe.atoms.masses.copy()

    def get_atom_charges(self) -> Optional[np.ndarray]:
        """
        Get atomic charges if available in topology.

        Returns:
            charges (np.ndarray) or None if not available
        """
        try:
            return self.universe.atoms.charges.copy()
        except (AttributeError, ValueError):
            logger.warning("Charges not available in topology")
            return None

    def close(self):
        """Close the trajectory."""
        self.universe.trajectory.close()


class MultiFileTrajectoryReader:
    """
    Handle multiple trajectory files for a single system.

    Useful when trajectory is split across multiple XTC files.
    """

    def __init__(self, topology_file: str, trajectory_files: List[str]):
        """
        Initialize reader for multiple trajectory files.

        Parameters:
            topology_file (str): Path to topology file
            trajectory_files (List[str]): List of trajectory file paths in order
        """
        self.topology_file = topology_file
        self.trajectory_files = trajectory_files

        # Create universe with all trajectories
        self.universe = mda.Universe(topology_file, *trajectory_files)
        self.n_atoms = self.universe.atoms.n_atoms
        self.n_frames = self.universe.trajectory.n_frames

        logger.info(f"Loaded {len(trajectory_files)} trajectory files, "
                    f"{self.n_frames} total frames")

    def iterate_frames(
        self, start: int = 0, stop: Optional[int] = None, step: int = 1
    ) -> Tuple[int, float, np.ndarray, np.ndarray]:
        """Iterate over all frames in all trajectories."""
        if stop is None:
            stop = self.n_frames

        for frame_idx in range(start, min(stop, self.n_frames), step):
            self.universe.trajectory[frame_idx]
            yield (
                frame_idx,
                self.universe.trajectory.time,
                self.universe.atoms.positions.copy(),
                self.universe.dimensions[:3].copy()
            )

    def close(self):
        """Close all trajectories."""
        self.universe.trajectory.close()
