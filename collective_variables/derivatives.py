"""
Analytical Derivatives for Collective Variables

Computes dCV/dX_i for all atoms - required for PLUMED biasing forces.

For water finger coordinate w:
  The gradient is computed through the bottleneck pair that defines w.
  dw/dX_i depends on which molecule i is part of the bottleneck.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DerivativeCalculator:
    """
    Compute analytical derivatives of collective variables.
    """

    @staticmethod
    def water_finger_derivatives(
        ion_pos: np.ndarray,
        water_positions: np.ndarray,
        bottleneck_pair: Tuple[int, int],
        box: Optional[np.ndarray] = None,
        eps: float = 1e-5
    ) -> np.ndarray:
        """
        Calculate derivatives of water finger coordinate w.

        The water finger coordinate is determined by two molecules that form
        the "bottleneck" - the maximum distance edge in the path connecting
        the ion to bulk water.

        Parameters:
            ion_pos (np.ndarray): Position of Cl⁻ ion [x, y, z]
            water_positions (np.ndarray): Positions of all water oxygens (N, 3)
            bottleneck_pair (Tuple[int, int]): Indices of bottleneck molecules
                                               (0 = ion, 1..N = waters)
            box (np.ndarray): Periodic box vectors (3, 3)
            eps (float): Numerical differentiation step (fallback)

        Returns:
            derivatives (np.ndarray): Shape (N+1, 3) where N = num waters
                                      derivatives[0] = dw/d(ion pos)
                                      derivatives[i+1] = dw/d(water_i pos)
        """

        n_waters = water_positions.shape[0]
        n_atoms = n_waters + 1
        derivatives = np.zeros((n_atoms, 3))

        if bottleneck_pair is None:
            logger.warning("No bottleneck pair; returning zero derivatives")
            return derivatives

        idx1, idx2 = bottleneck_pair

        # Get positions of the two molecules forming the bottleneck
        if idx1 == 0:
            pos1 = ion_pos
            pos2 = water_positions[idx2 - 1]
            atom1_is_ion = True
            atom2_is_ion = False
            atom2_idx = idx2 - 1
        elif idx2 == 0:
            pos1 = water_positions[idx1 - 1]
            pos2 = ion_pos
            atom1_is_ion = False
            atom2_is_ion = True
            atom1_idx = idx1 - 1
        else:
            pos1 = water_positions[idx1 - 1]
            pos2 = water_positions[idx2 - 1]
            atom1_is_ion = False
            atom2_is_ion = False
            atom1_idx = idx1 - 1
            atom2_idx = idx2 - 1

        # Calculate distance and its gradient
        delta = DerivativeCalculator._compute_delta_pbc(pos2, pos1, box)
        distance = np.linalg.norm(delta)

        if distance < 1e-10:
            logger.warning("Bottleneck pair very close; returning zero derivatives")
            return derivatives

        # Gradient of distance between two points
        grad_distance = delta / distance

        # Apply chain rule: dw/dX = grad_distance * (dX influence)
        # The water finger coordinate equals the distance of the bottleneck pair

        if atom1_is_ion:
            # idx1 is ion (0), idx2 is water
            derivatives[0] = -grad_distance  # dw/d(ion) = -grad_distance
            derivatives[idx2] = grad_distance  # dw/d(water) = +grad_distance

        elif atom2_is_ion:
            # idx1 is water, idx2 is ion (0)
            derivatives[idx1] = -grad_distance
            derivatives[0] = grad_distance

        else:
            # Both are waters
            derivatives[idx1] = -grad_distance
            derivatives[idx2] = grad_distance

        return derivatives

    @staticmethod
    def _compute_delta_pbc(
        pos_to: np.ndarray,
        pos_from: np.ndarray,
        box: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Compute displacement vector with periodic boundary conditions.

        Returns: pos_to - pos_from with PBC applied
        """
        delta = pos_to - pos_from

        if box is not None:
            for dim in range(3):
                box_length = np.linalg.norm(box[dim])
                if box_length > 0:
                    delta[dim] = delta[dim] - box_length * np.round(delta[dim] / box_length)

        return delta

    @staticmethod
    def solvation_derivatives(
        ion_pos: np.ndarray,
        solvent_positions: np.ndarray,
        solvent_charges: np.ndarray,
        solvent_lj_params: Tuple[np.ndarray, np.ndarray],
        ion_charge: float,
        ion_lj_params: Tuple[float, float],
        box: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate derivatives of solvation coordinate s.

        s = U_interaction / U_bulk where U_interaction is the sum of
        Lennard-Jones and Coulomb interactions.

        Parameters:
            ion_pos: Position of ion
            solvent_positions: Positions of solvent molecules (N, 3)
            solvent_charges: Charges of solvent atoms (N,)
            solvent_lj_params: (sigma, epsilon) arrays for solvent
            ion_charge: Charge of ion
            ion_lj_params: (sigma, epsilon) for ion
            box: Periodic box vectors

        Returns:
            derivatives: Shape (N+1, 3) - gradients w.r.t. all atomic positions
        """

        n_solvent = solvent_positions.shape[0]
        n_atoms = n_solvent + 1
        derivatives = np.zeros((n_atoms, 3))

        sigma_ion, epsilon_ion = ion_lj_params
        sigma_solvent, epsilon_solvent = solvent_lj_params

        # Coulomb constant (in kcal/mol Ų⁻² for atomic units)
        k_coul = 332.0636  # kcal·Å/mol·e²

        for i in range(n_solvent):
            delta = DerivativeCalculator._compute_delta_pbc(
                solvent_positions[i], ion_pos, box
            )
            r = np.linalg.norm(delta)

            if r < 1e-8:
                continue

            r_hat = delta / r

            # Lennard-Jones contribution
            # U_LJ = 4ε[(σ/r)^12 - (σ/r)^6]
            # dU/dr = 4ε[-12(σ/r)^12/r + 6(σ/r)^6/r]

            sigma = (sigma_ion + sigma_solvent[i]) / 2.0
            epsilon = np.sqrt(epsilon_ion * epsilon_solvent[i])

            r6 = r ** 6
            r12 = r ** 12

            dU_LJ_dr = 4 * epsilon * (-12 * (sigma ** 12) / r12 / r +
                                       6 * (sigma ** 6) / r6 / r)

            # Coulomb contribution
            # U_C = k * q1 * q2 / r
            # dU/dr = -k * q1 * q2 / r^2

            dU_Coul_dr = -k_coul * ion_charge * solvent_charges[i] / (r * r)

            # Total radial derivative
            dU_dr = dU_LJ_dr + dU_Coul_dr

            # Convert to Cartesian: dU/dX = (dU/dr) * (dX/dr) = (dU/dr) * r_hat
            grad_U = dU_dr * r_hat

            # dU/d(ion) = -grad_U (negative because ion position appears as -delta)
            derivatives[0] -= grad_U
            # dU/d(solvent_i) = grad_U
            derivatives[i + 1] += grad_U

        return derivatives

    @staticmethod
    def numerical_derivative(
        cv_func,
        positions: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """
        Compute numerical derivatives using finite differences.

        For validation only - should match analytical derivatives.

        Parameters:
            cv_func: Function that takes positions and returns CV value
            positions: All atomic positions (N, 3)
            eps: Finite difference step

        Returns:
            derivatives: Shape (N, 3)
        """

        n_atoms = positions.shape[0]
        derivatives = np.zeros((n_atoms, 3))

        cv_ref = cv_func(positions)

        for i in range(n_atoms):
            for j in range(3):
                positions_pert = positions.copy()
                positions_pert[i, j] += eps

                cv_pert = cv_func(positions_pert)
                derivatives[i, j] = (cv_pert - cv_ref) / eps

        return derivatives
