"""
Bulk Solvation Energy Extraction

Calculate bulk ion-solvent interaction energies (U_bulk) by sampling
the bulk regions of an interface MD trajectory.

Strategy:
1. Define bulk regions (Z < z_cutoff for water, Z > z_cutoff for DCM)
2. Identify frames where ion is in bulk region
3. Calculate interaction energy (LJ + Coulomb) for those frames
4. Time-average to get <U_bulk>
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class BulkEnergyCalculator:
    """
    Extract bulk solvation energies from interface MD trajectory.

    Parameters:
        water_region_z (float): Z coordinate threshold for water bulk (typically < -10 Å)
        dcm_region_z (float): Z coordinate threshold for DCM bulk (typically > +10 Å)
        min_frames_fraction (float): Minimum fraction of frames with ion in bulk
        k_coul (float): Coulomb constant (kcal·Å/mol·e²)
    """

    def __init__(
        self,
        water_region_z: float = -10.0,
        dcm_region_z: float = 10.0,
        min_frames_fraction: float = 0.1,
        k_coul: float = 332.0636
    ):
        """Initialize bulk energy calculator."""
        self.water_region_z = water_region_z
        self.dcm_region_z = dcm_region_z
        self.min_frames_fraction = min_frames_fraction
        self.k_coul = k_coul

    def calculate_bulk_energy_water(
        self,
        ion_pos_trajectory: np.ndarray,
        water_positions_trajectory: np.ndarray,
        water_charges: np.ndarray,
        ion_charge: float,
        ion_lj_params: Tuple[float, float],
        water_lj_params: Tuple[np.ndarray, np.ndarray],
        box_trajectory: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Tuple[float, int, float]:
        """
        Calculate bulk ion-water solvation energy.

        Parameters:
            ion_pos_trajectory (np.ndarray): Ion positions over time (n_frames, 3)
            water_positions_trajectory (np.ndarray): Water positions (n_frames, n_waters, 3)
            water_charges (np.ndarray): Charges of water atoms (n_waters,)
            ion_charge (float): Charge of ion
            ion_lj_params (Tuple): (sigma, epsilon) for ion
            water_lj_params (Tuple): (sigma, epsilon) arrays for water atoms
            box_trajectory (np.ndarray): Box vectors (n_frames, 3, 3)
            verbose (bool): Print statistics

        Returns:
            U_bulk_water (float): Average bulk solvation energy (kcal/mol)
            n_bulk_frames (int): Number of frames sampled
            std_error (float): Standard error of mean
        """

        n_frames = ion_pos_trajectory.shape[0]
        energies = []

        for frame in range(n_frames):
            ion_z = ion_pos_trajectory[frame, 2]

            # Check if ion is in water bulk region
            if ion_z < self.water_region_z:
                energy = self._calculate_frame_energy(
                    ion_pos_trajectory[frame],
                    water_positions_trajectory[frame],
                    water_charges,
                    ion_charge,
                    ion_lj_params,
                    water_lj_params,
                    box_trajectory[frame] if box_trajectory is not None else None
                )
                energies.append(energy)

        if len(energies) == 0:
            logger.warning("No frames with ion in water bulk region!")
            return 0.0, 0, 0.0

        if len(energies) < self.min_frames_fraction * n_frames:
            frac = len(energies) / n_frames
            logger.warning(
                f"Low sampling of water bulk: {len(energies)} frames "
                f"({frac:.1%}) < {self.min_frames_fraction:.0%}"
            )

        energies = np.array(energies)
        U_bulk = np.mean(energies)
        std_error = np.std(energies) / np.sqrt(len(energies))

        if verbose:
            logger.info(f"Water bulk energy: {U_bulk:.2f} ± {std_error:.2f} kcal/mol "
                        f"({len(energies)} frames)")

        return U_bulk, len(energies), std_error

    def calculate_bulk_energy_dcm(
        self,
        ion_pos_trajectory: np.ndarray,
        dcm_positions_trajectory: np.ndarray,
        dcm_charges: np.ndarray,
        ion_charge: float,
        ion_lj_params: Tuple[float, float],
        dcm_lj_params: Tuple[np.ndarray, np.ndarray],
        box_trajectory: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Tuple[float, int, float]:
        """
        Calculate bulk ion-DCM solvation energy.

        Parameters:
            ion_pos_trajectory (np.ndarray): Ion positions over time (n_frames, 3)
            dcm_positions_trajectory (np.ndarray): DCM positions (n_frames, n_dcm_atoms, 3)
            dcm_charges (np.ndarray): Charges of DCM atoms
            ion_charge (float): Charge of ion
            ion_lj_params (Tuple): (sigma, epsilon) for ion
            dcm_lj_params (Tuple): (sigma, epsilon) arrays for DCM atoms
            box_trajectory (np.ndarray): Box vectors (n_frames, 3, 3)
            verbose (bool): Print statistics

        Returns:
            U_bulk_dcm (float): Average bulk solvation energy (kcal/mol)
            n_bulk_frames (int): Number of frames sampled
            std_error (float): Standard error of mean
        """

        n_frames = ion_pos_trajectory.shape[0]
        energies = []

        for frame in range(n_frames):
            ion_z = ion_pos_trajectory[frame, 2]

            # Check if ion is in DCM bulk region
            if ion_z > self.dcm_region_z:
                energy = self._calculate_frame_energy(
                    ion_pos_trajectory[frame],
                    dcm_positions_trajectory[frame],
                    dcm_charges,
                    ion_charge,
                    ion_lj_params,
                    dcm_lj_params,
                    box_trajectory[frame] if box_trajectory is not None else None
                )
                energies.append(energy)

        if len(energies) == 0:
            logger.warning("No frames with ion in DCM bulk region!")
            return 0.0, 0, 0.0

        if len(energies) < self.min_frames_fraction * n_frames:
            frac = len(energies) / n_frames
            logger.warning(
                f"Low sampling of DCM bulk: {len(energies)} frames "
                f"({frac:.1%}) < {self.min_frames_fraction:.0%}"
            )

        energies = np.array(energies)
        U_bulk = np.mean(energies)
        std_error = np.std(energies) / np.sqrt(len(energies))

        if verbose:
            logger.info(f"DCM bulk energy: {U_bulk:.2f} ± {std_error:.2f} kcal/mol "
                        f"({len(energies)} frames)")

        return U_bulk, len(energies), std_error

    def _calculate_frame_energy(
        self,
        ion_pos: np.ndarray,
        solvent_positions: np.ndarray,
        solvent_charges: np.ndarray,
        ion_charge: float,
        ion_lj_params: Tuple[float, float],
        solvent_lj_params: Tuple[np.ndarray, np.ndarray],
        box: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate total ion-solvent interaction energy for one frame.

        Returns:
            energy (float): Total interaction energy in kcal/mol
        """

        sigma_ion, epsilon_ion = ion_lj_params
        sigma_solvent, epsilon_solvent = solvent_lj_params

        energy_lj = 0.0
        energy_coul = 0.0

        n_solvent = solvent_positions.shape[0]

        for i in range(n_solvent):
            delta = solvent_positions[i] - ion_pos

            # Apply PBC
            if box is not None:
                for dim in range(3):
                    box_length = np.linalg.norm(box[dim])
                    if box_length > 0:
                        delta[dim] = delta[dim] - box_length * np.round(delta[dim] / box_length)

            r = np.linalg.norm(delta)

            if r < 0.1:  # Skip very close atoms
                continue

            # Lennard-Jones interaction
            # U_LJ = 4ε[(σ/r)^12 - (σ/r)^6]
            sigma = (sigma_ion + sigma_solvent[i]) / 2.0
            epsilon = np.sqrt(epsilon_ion * epsilon_solvent[i])

            sigma_r = sigma / r
            sigma_r6 = sigma_r ** 6
            sigma_r12 = sigma_r6 * sigma_r6

            u_lj = 4 * epsilon * (sigma_r12 - sigma_r6)
            energy_lj += u_lj

            # Coulomb interaction
            # U_C = k * q1 * q2 / r
            u_coul = self.k_coul * ion_charge * solvent_charges[i] / r
            energy_coul += u_coul

        total_energy = energy_lj + energy_coul

        return total_energy
