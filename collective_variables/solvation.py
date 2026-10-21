"""
Solvation Coordinates (s_W, s_NB) Calculation

Based on Karnes & Benjamin (2016) - normalized ion-solvent interaction energies
that track the solvation state of the transferring ion.

  s_W  = U_I-W  / U_bulk_W   (water solvation coordinate)
  s_NB = U_I-NB / U_bulk_NB  (non-aqueous solvation coordinate)

Where:
  U_I-W  = instantaneous ion-water interaction energy (LJ + Coulomb)
  U_bulk_W = average U_I-W when ion is in bulk water
  U_I-NB = instantaneous ion-DCM interaction energy
  U_bulk_NB = average U_I-NB when ion is in bulk DCM

s_W ≈ 1 when ion is fully solvated by water, ≈ 0 when desolvated.
s_NB ≈ 1 when ion is fully solvated by DCM, ≈ 0 when desolvated.

Reference:
  Karnes, J. J. & Benjamin, I. (2016).
  J. Chem. Phys., 145, 014701.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SolvationCoordinate:
    """
    Compute solvation coordinates s_W and s_NB from molecular positions.

    Parameters:
        k_coul (float): Coulomb constant (kcal·Å/mol·e²)
        min_distance (float): Minimum distance cutoff to avoid singularities (Å)
    """

    def __init__(
        self,
        k_coul: float = 332.0636,
        min_distance: float = 0.1
    ):
        self.k_coul = k_coul
        self.min_distance = min_distance

    def calculate_interaction_energy(
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

        Computes sum of LJ + Coulomb interactions between the ion and
        all solvent atoms. Uses Lorentz-Berthelot combining rules.

        Parameters:
            ion_pos: Ion position [x, y, z]
            solvent_positions: Solvent atom positions (N, 3)
            solvent_charges: Charges of solvent atoms (N,)
            ion_charge: Charge of ion
            ion_lj_params: (sigma, epsilon) for ion
            solvent_lj_params: (sigma_array, epsilon_array) for solvent atoms
            box: Periodic box vectors (3, 3)

        Returns:
            U: Total interaction energy (kcal/mol)
        """
        sigma_ion, epsilon_ion = ion_lj_params
        sigma_solvent, epsilon_solvent = solvent_lj_params
        n_solvent = solvent_positions.shape[0]

        # Vectorized distance calculation
        delta = solvent_positions - ion_pos[np.newaxis, :]

        # Apply PBC
        if box is not None:
            for dim in range(3):
                box_length = np.linalg.norm(box[dim])
                if box_length > 0:
                    delta[:, dim] -= box_length * np.round(delta[:, dim] / box_length)

        r = np.linalg.norm(delta, axis=1)

        # Mask very close atoms
        valid = r > self.min_distance
        r_valid = r[valid]

        if len(r_valid) == 0:
            return 0.0

        # Lorentz-Berthelot combining rules
        sigma = (sigma_ion + sigma_solvent[valid]) / 2.0
        epsilon = np.sqrt(epsilon_ion * epsilon_solvent[valid])

        # Lennard-Jones: U_LJ = 4ε[(σ/r)^12 - (σ/r)^6]
        sigma_r = sigma / r_valid
        sigma_r6 = sigma_r ** 6
        u_lj = np.sum(4.0 * epsilon * (sigma_r6 * sigma_r6 - sigma_r6))

        # Coulomb: U_C = k * q_ion * q_i / r
        u_coul = np.sum(
            self.k_coul * ion_charge * solvent_charges[valid] / r_valid
        )

        return u_lj + u_coul

    def calculate_solvation_coordinate(
        self,
        ion_pos: np.ndarray,
        solvent_positions: np.ndarray,
        solvent_charges: np.ndarray,
        ion_charge: float,
        ion_lj_params: Tuple[float, float],
        solvent_lj_params: Tuple[np.ndarray, np.ndarray],
        U_bulk: float,
        box: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate solvation coordinate s = U_interaction / U_bulk.

        Parameters:
            ion_pos: Ion position [x, y, z]
            solvent_positions: Solvent atom positions (N, 3)
            solvent_charges: Charges of solvent atoms (N,)
            ion_charge: Charge of ion
            ion_lj_params: (sigma, epsilon) for ion
            solvent_lj_params: (sigma_array, epsilon_array) for solvent
            U_bulk: Bulk solvation energy (kcal/mol), must be non-zero
            box: Periodic box vectors (3, 3)

        Returns:
            s: Solvation coordinate (dimensionless, ≈1 in bulk, ≈0 desolvated)
        """
        if abs(U_bulk) < 1e-10:
            raise ValueError("U_bulk is too close to zero for normalization")

        U_interaction = self.calculate_interaction_energy(
            ion_pos, solvent_positions, solvent_charges,
            ion_charge, ion_lj_params, solvent_lj_params, box
        )

        return U_interaction / U_bulk

    def calculate_solvation_derivatives(
        self,
        ion_pos: np.ndarray,
        solvent_positions: np.ndarray,
        solvent_charges: np.ndarray,
        ion_charge: float,
        ion_lj_params: Tuple[float, float],
        solvent_lj_params: Tuple[np.ndarray, np.ndarray],
        U_bulk: float,
        box: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate analytical derivatives of solvation coordinate s.

        ds/dX_i = (1/U_bulk) * dU_interaction/dX_i

        Parameters:
            (same as calculate_solvation_coordinate)

        Returns:
            derivatives: Shape (N+1, 3) where N = num solvent atoms
                         derivatives[0] = ds/d(ion pos)
                         derivatives[i+1] = ds/d(solvent_i pos)
        """
        if abs(U_bulk) < 1e-10:
            raise ValueError("U_bulk is too close to zero for normalization")

        sigma_ion, epsilon_ion = ion_lj_params
        sigma_solvent, epsilon_solvent = solvent_lj_params
        n_solvent = solvent_positions.shape[0]
        derivatives = np.zeros((n_solvent + 1, 3))

        for i in range(n_solvent):
            delta = solvent_positions[i] - ion_pos

            # Apply PBC
            if box is not None:
                for dim in range(3):
                    box_length = np.linalg.norm(box[dim])
                    if box_length > 0:
                        delta[dim] -= box_length * np.round(delta[dim] / box_length)

            r = np.linalg.norm(delta)

            if r < self.min_distance:
                continue

            r_hat = delta / r

            # LJ derivative: dU_LJ/dr = 4ε[-12σ^12/r^13 + 6σ^6/r^7]
            sigma = (sigma_ion + sigma_solvent[i]) / 2.0
            epsilon = np.sqrt(epsilon_ion * epsilon_solvent[i])

            sigma_r = sigma / r
            sigma_r6 = sigma_r ** 6
            sigma_r12 = sigma_r6 * sigma_r6

            dU_LJ_dr = 4.0 * epsilon * (-12.0 * sigma_r12 / r + 6.0 * sigma_r6 / r)

            # Coulomb derivative: dU_C/dr = -k * q1 * q2 / r^2
            dU_Coul_dr = -self.k_coul * ion_charge * solvent_charges[i] / (r * r)

            # Total radial force
            dU_dr = dU_LJ_dr + dU_Coul_dr

            # Convert to Cartesian gradient: dU/dX = (dU/dr) * r_hat
            grad_U = dU_dr * r_hat

            # Chain rule for s = U / U_bulk
            grad_s = grad_U / U_bulk

            # Ion moves opposite to solvent
            derivatives[0] -= grad_s
            derivatives[i + 1] += grad_s

        return derivatives
