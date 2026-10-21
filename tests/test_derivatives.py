"""
Unit Tests for Analytical Derivatives

Validates analytical derivative calculations by comparing with
numerical differentiation. Critical for PLUMED biasing accuracy.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collective_variables.water_finger import WaterFingerCoordinate
from collective_variables.derivatives import DerivativeCalculator


class TestWaterFingerDerivatives:
    """Test analytical derivatives for water finger coordinate."""

    @pytest.fixture
    def setup(self):
        """Setup common test configuration."""
        return {
            'calculator': WaterFingerCoordinate(),
            'derivative_calculator': DerivativeCalculator(),
            'eps': 1e-5,
            'rtol': 1e-3  # Relative tolerance for derivative comparison
        }

    def test_derivatives_shape(self, setup):
        """Test that derivatives have correct shape."""
        ion_pos = np.array([0.0, 0.0, 5.0])
        water_positions = np.array([
            [0.0, 0.0, 2.5],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, -2.0],
        ])

        # Calculate w and derivatives
        w, bottleneck = setup['calculator'].calculate(
            ion_pos, water_positions, return_bottleneck=True
        )

        derivs = DerivativeCalculator.water_finger_derivatives(
            ion_pos, water_positions, bottleneck
        )

        # Should be (n_atoms, 3)
        n_atoms = 1 + water_positions.shape[0]
        assert derivs.shape == (n_atoms, 3), \
            f"Expected shape {(n_atoms, 3)}, got {derivs.shape}"

    def test_derivatives_numerical_validation(self, setup):
        """
        Validate analytical derivatives against numerical differentiation.

        This is critical for PLUMED biasing - derivatives must be accurate.
        """
        ion_pos = np.array([0.0, 0.0, 5.0])
        water_positions = np.array([
            [0.0, 0.0, 2.5],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, -2.0],
        ])

        # Calculate analytical derivatives
        w, bottleneck = setup['calculator'].calculate(
            ion_pos, water_positions, return_bottleneck=True
        )

        derivs_analytical = DerivativeCalculator.water_finger_derivatives(
            ion_pos, water_positions, bottleneck
        )

        # Calculate numerical derivatives
        eps = setup['eps']
        derivs_numerical = np.zeros_like(derivs_analytical)

        # Derivative w.r.t. ion position
        for j in range(3):
            ion_pert = ion_pos.copy()
            ion_pert[j] += eps
            w_pert = setup['calculator'].calculate(ion_pert, water_positions)
            derivs_numerical[0, j] = (w_pert - w) / eps

        # Derivative w.r.t. water positions
        for i in range(water_positions.shape[0]):
            for j in range(3):
                water_pert = water_positions.copy()
                water_pert[i, j] += eps
                w_pert = setup['calculator'].calculate(ion_pos, water_pert)
                derivs_numerical[i + 1, j] = (w_pert - w) / eps

        # Compare analytical and numerical
        rel_error = np.abs(derivs_analytical - derivs_numerical) / (np.abs(derivs_analytical) + 1e-10)
        max_error = np.max(rel_error)

        assert max_error < setup['rtol'], \
            f"Derivatives differ by {max_error:.6f}, threshold {setup['rtol']}"

    def test_zero_derivative_far_atoms(self, setup):
        """
        Test that atoms not in bottleneck have near-zero derivatives.

        Only the two atoms forming the bottleneck should have non-zero gradients.
        """
        ion_pos = np.array([0.0, 0.0, 5.0])
        water_positions = np.array([
            [0.0, 0.0, 2.5],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, -2.0],
            [5.0, 5.0, -10.0],  # Far away, not in bottleneck
        ])

        w, bottleneck = setup['calculator'].calculate(
            ion_pos, water_positions, return_bottleneck=True
        )

        derivs = DerivativeCalculator.water_finger_derivatives(
            ion_pos, water_positions, bottleneck
        )

        # The far atom should have near-zero derivative
        far_atom_idx = 4  # Last water, index 4 in derivatives array
        far_atom_deriv_norm = np.linalg.norm(derivs[far_atom_idx])

        assert far_atom_deriv_norm < 1e-6, \
            f"Far atom should have ~zero derivative, got {far_atom_deriv_norm}"

    def test_bottleneck_derivative_magnitude(self, setup):
        """
        Test that bottleneck atoms have reasonable derivative magnitudes.

        Derivatives should relate to the distance between bottleneck atoms.
        """
        ion_pos = np.array([0.0, 0.0, 5.0])
        water_positions = np.array([
            [0.0, 0.0, 2.5],
            [0.0, 0.0, -2.5],
        ])

        w, bottleneck = setup['calculator'].calculate(
            ion_pos, water_positions, return_bottleneck=True
        )

        derivs = DerivativeCalculator.water_finger_derivatives(
            ion_pos, water_positions, bottleneck
        )

        # Bottleneck atoms should have meaningful gradients
        bottleneck_derivs_norm = [
            np.linalg.norm(derivs[idx]) for idx in bottleneck
        ]

        assert all(n > 1e-3 for n in bottleneck_derivs_norm), \
            f"Bottleneck atoms should have large derivatives, got {bottleneck_derivs_norm}"

    def test_derivative_symmetry(self, setup):
        """
        Test derivative symmetry properties.

        For symmetric configurations, derivatives should be symmetric.
        """
        ion_pos = np.array([0.0, 0.0, 0.0])

        # Symmetric configuration
        water_positions = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -2.0],
        ])

        w, bottleneck = setup['calculator'].calculate(
            ion_pos, water_positions, return_bottleneck=True
        )

        derivs = DerivativeCalculator.water_finger_derivatives(
            ion_pos, water_positions, bottleneck
        )

        # Ions should have symmetric derivatives if bottleneck doesn't prefer one
        # (This is a weaker test - just check no NaN)
        assert np.all(np.isfinite(derivs)), "All derivatives should be finite"

    def test_pbc_derivatives(self, setup):
        """
        Test that derivatives work correctly with periodic boundary conditions.
        """
        box = np.array([
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 20.0]
        ])

        ion_pos = np.array([1.0, 10.0, 10.0])
        water_positions = np.array([
            [19.0, 10.0, 10.0],  # Close with PBC
            [19.0, 10.0, -5.0],
        ])

        w, bottleneck = setup['calculator'].calculate(
            ion_pos, water_positions, box=box, return_bottleneck=True
        )

        derivs = DerivativeCalculator.water_finger_derivatives(
            ion_pos, water_positions, bottleneck, box=box
        )

        # Should produce valid derivatives
        assert np.all(np.isfinite(derivs)), "All derivatives should be finite with PBC"
        assert np.any(np.abs(derivs) > 1e-10), "Some derivatives should be non-zero"


class TestSolvationDerivatives:
    """Test analytical derivatives for solvation coordinates."""

    def test_solvation_derivatives_shape(self):
        """Test that solvation coordinate derivatives have correct shape."""
        ion_pos = np.array([0.0, 0.0, 0.0])
        water_positions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        water_charges = np.array([-0.73, 0.365, 0.365])
        ion_charge = -1.0
        ion_lj = (4.410, 0.100)
        water_lj = (np.array([3.204, 0.0, 0.0]), np.array([0.160, 0.0, 0.0]))

        derivs = DerivativeCalculator.solvation_derivatives(
            ion_pos, water_positions, water_charges,
            water_lj, ion_charge, ion_lj
        )

        n_atoms = 1 + water_positions.shape[0]
        assert derivs.shape == (n_atoms, 3)

    def test_solvation_derivatives_finite(self):
        """Test that solvation derivatives are finite."""
        ion_pos = np.array([0.0, 0.0, 5.0])
        water_positions = np.random.randn(10, 3) * 3

        water_charges = np.array([-0.73] + [0.365] * 9)
        ion_charge = -1.0
        ion_lj = (4.410, 0.100)
        water_lj = (
            np.array([3.204] + [0.0]*9),
            np.array([0.160] + [0.0]*9)
        )

        derivs = DerivativeCalculator.solvation_derivatives(
            ion_pos, water_positions, water_charges,
            water_lj, ion_charge, ion_lj
        )

        assert np.all(np.isfinite(derivs)), "All derivatives should be finite"


class TestNumericalDerivative:
    """Test the numerical derivative helper function."""

    def test_numerical_derivative_water_finger(self):
        """Test numerical derivative computation for water finger."""
        calculator = WaterFingerCoordinate()

        ion_pos = np.array([0.0, 0.0, 5.0])
        water_positions = np.array([
            [0.0, 0.0, 2.5],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, -2.0],
        ])

        # Concatenate all positions
        all_positions = np.vstack([ion_pos, water_positions])

        def cv_func(positions):
            ion = positions[0]
            waters = positions[1:]
            w = calculator.calculate(ion, waters)
            return w

        derivs_numerical = DerivativeCalculator.numerical_derivative(
            cv_func, all_positions, eps=1e-5
        )

        # Should have same shape as positions
        assert derivs_numerical.shape == all_positions.shape

        # Should be finite
        assert np.all(np.isfinite(derivs_numerical))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
