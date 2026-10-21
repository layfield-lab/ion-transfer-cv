"""
Unit Tests for Solvation Coordinates (s_W, s_NB)

Validates the solvation coordinate calculation and its analytical
derivatives against numerical differentiation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collective_variables.solvation import SolvationCoordinate


# Shared test parameters for Cl⁻ / TIP3P-like water
ION_CHARGE = -1.0
ION_LJ = (4.410, 0.100)  # sigma, epsilon for Cl⁻

# 3-site water: O + 2H per molecule. For simplicity, tests use
# per-atom arrays (one entry per atom, not per molecule).
WATER_CHARGES_3 = np.array([-0.73, 0.365, 0.365])
WATER_LJ_SIGMA_3 = np.array([3.204, 0.0, 0.0])
WATER_LJ_EPS_3 = np.array([0.160, 0.0, 0.0])


class TestInteractionEnergy:
    """Test the raw ion-solvent interaction energy calculation."""

    @pytest.fixture
    def calc(self):
        return SolvationCoordinate()

    def test_single_atom_coulomb(self, calc):
        """Coulomb energy between ion and one charged atom at known distance."""
        ion_pos = np.array([0.0, 0.0, 0.0])
        solvent_pos = np.array([[3.0, 0.0, 0.0]])
        charges = np.array([-0.73])
        lj_sigma = np.array([3.204])
        lj_eps = np.array([0.160])

        U = calc.calculate_interaction_energy(
            ion_pos, solvent_pos, charges,
            ION_CHARGE, ION_LJ, (lj_sigma, lj_eps)
        )

        # Expected Coulomb: k * q1 * q2 / r = 332.0636 * (-1) * (-0.73) / 3.0
        U_coul_expected = 332.0636 * 1.0 * 0.73 / 3.0
        # LJ will also contribute, so just check finite and positive
        # (both charges negative -> Coulomb repulsive -> positive)
        assert np.isfinite(U)
        # Coulomb alone would be ~80.8 kcal/mol; LJ may shift it
        assert U > 0, "Should be repulsive (like charges)"

    def test_energy_decreases_with_distance(self, calc):
        """Energy magnitude decreases as solvent moves further away."""
        ion_pos = np.array([0.0, 0.0, 0.0])

        energies = []
        for dist in [3.0, 5.0, 8.0, 12.0]:
            solvent_pos = np.array([[dist, 0.0, 0.0]])
            U = calc.calculate_interaction_energy(
                ion_pos, solvent_pos, WATER_CHARGES_3[:1],
                ION_CHARGE, ION_LJ,
                (WATER_LJ_SIGMA_3[:1], WATER_LJ_EPS_3[:1])
            )
            energies.append(U)

        # At large distances Coulomb dominates; magnitude should decrease
        abs_energies = [abs(e) for e in energies]
        for i in range(len(abs_energies) - 1):
            assert abs_energies[i] > abs_energies[i + 1], \
                f"|U| should decrease with distance: {abs_energies}"

    def test_pbc_energy(self, calc):
        """PBC should give different energy when atom wraps around box."""
        ion_pos = np.array([1.0, 0.0, 0.0])
        solvent_pos = np.array([[19.0, 0.0, 0.0]])
        charges = np.array([-0.73])
        lj_sigma = np.array([3.204])
        lj_eps = np.array([0.160])

        box = np.array([
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 20.0]
        ])

        U_pbc = calc.calculate_interaction_energy(
            ion_pos, solvent_pos, charges,
            ION_CHARGE, ION_LJ, (lj_sigma, lj_eps), box=box
        )
        U_nopbc = calc.calculate_interaction_energy(
            ion_pos, solvent_pos, charges,
            ION_CHARGE, ION_LJ, (lj_sigma, lj_eps), box=None
        )

        # With PBC distance is 2 Å; without PBC distance is 18 Å
        assert abs(U_pbc) > abs(U_nopbc), \
            "PBC should give stronger interaction (closer distance)"

    def test_multi_atom_energy(self, calc):
        """Energy with multiple solvent atoms is sum of pairwise terms."""
        ion_pos = np.array([0.0, 0.0, 0.0])
        solvent_pos = np.array([
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ])

        U_all = calc.calculate_interaction_energy(
            ion_pos, solvent_pos, WATER_CHARGES_3,
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3, WATER_LJ_EPS_3)
        )

        # Sum of individual
        U_sum = 0.0
        for i in range(3):
            U_i = calc.calculate_interaction_energy(
                ion_pos, solvent_pos[i:i+1],
                WATER_CHARGES_3[i:i+1],
                ION_CHARGE, ION_LJ,
                (WATER_LJ_SIGMA_3[i:i+1], WATER_LJ_EPS_3[i:i+1])
            )
            U_sum += U_i

        assert np.isclose(U_all, U_sum, rtol=1e-10), \
            f"Multi-atom energy should equal sum of pairwise: {U_all} vs {U_sum}"


class TestSolvationCoordinate:
    """Test the normalized solvation coordinate s = U / U_bulk."""

    @pytest.fixture
    def calc(self):
        return SolvationCoordinate()

    def test_bulk_gives_s_near_one(self, calc):
        """When configuration matches bulk, s should be close to 1."""
        ion_pos = np.array([0.0, 0.0, 0.0])
        solvent_pos = np.array([
            [3.0, 0.0, 0.0],
            [0.0, 3.5, 0.0],
            [0.0, 0.0, 3.2],
        ])

        # First compute the energy for this config as "bulk"
        U_bulk = calc.calculate_interaction_energy(
            ion_pos, solvent_pos, WATER_CHARGES_3,
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3, WATER_LJ_EPS_3)
        )

        # Same config should give s = 1
        s = calc.calculate_solvation_coordinate(
            ion_pos, solvent_pos, WATER_CHARGES_3,
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3, WATER_LJ_EPS_3),
            U_bulk
        )
        assert np.isclose(s, 1.0, rtol=1e-10)

    def test_desolvated_gives_small_s(self, calc):
        """When solvent is far away, s should be small."""
        ion_pos = np.array([0.0, 0.0, 0.0])

        # "Bulk" config: solvent close
        close_pos = np.array([[3.0, 0.0, 0.0]])
        U_bulk = calc.calculate_interaction_energy(
            ion_pos, close_pos, WATER_CHARGES_3[:1],
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3[:1], WATER_LJ_EPS_3[:1])
        )

        # Desolvated config: solvent far
        far_pos = np.array([[30.0, 0.0, 0.0]])
        s = calc.calculate_solvation_coordinate(
            ion_pos, far_pos, WATER_CHARGES_3[:1],
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3[:1], WATER_LJ_EPS_3[:1]),
            U_bulk
        )
        assert abs(s) < 0.2, f"Desolvated s should be small, got {s}"

    def test_zero_U_bulk_raises(self, calc):
        """Passing U_bulk=0 should raise ValueError."""
        ion_pos = np.array([0.0, 0.0, 0.0])
        solvent_pos = np.array([[3.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="U_bulk"):
            calc.calculate_solvation_coordinate(
                ion_pos, solvent_pos, WATER_CHARGES_3[:1],
                ION_CHARGE, ION_LJ,
                (WATER_LJ_SIGMA_3[:1], WATER_LJ_EPS_3[:1]),
                U_bulk=0.0
            )

    def test_s_scales_linearly(self, calc):
        """s should scale linearly with interaction energy."""
        ion_pos = np.array([0.0, 0.0, 0.0])
        solvent_pos = np.array([[4.0, 0.0, 0.0]])

        U = calc.calculate_interaction_energy(
            ion_pos, solvent_pos, WATER_CHARGES_3[:1],
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3[:1], WATER_LJ_EPS_3[:1])
        )

        # s = U / U_bulk, so if U_bulk = 2*U, s should be 0.5
        s = calc.calculate_solvation_coordinate(
            ion_pos, solvent_pos, WATER_CHARGES_3[:1],
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3[:1], WATER_LJ_EPS_3[:1]),
            U_bulk=2.0 * U
        )
        assert np.isclose(s, 0.5, rtol=1e-10)


class TestSolvationDerivatives:
    """Test analytical derivatives against numerical differentiation."""

    @pytest.fixture
    def setup(self):
        calc = SolvationCoordinate()
        ion_pos = np.array([0.0, 0.0, 0.0])
        solvent_pos = np.array([
            [3.5, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 3.8],
        ])
        # Use an arbitrary but realistic U_bulk
        U_bulk = calc.calculate_interaction_energy(
            ion_pos, solvent_pos, WATER_CHARGES_3,
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3, WATER_LJ_EPS_3)
        ) * 1.5  # scale so s != 1 exactly
        return {
            'calc': calc,
            'ion_pos': ion_pos,
            'solvent_pos': solvent_pos,
            'U_bulk': U_bulk,
            'eps': 1e-5,
            'rtol': 1e-3,
        }

    def test_derivatives_shape(self, setup):
        derivs = setup['calc'].calculate_solvation_derivatives(
            setup['ion_pos'], setup['solvent_pos'], WATER_CHARGES_3,
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3, WATER_LJ_EPS_3),
            setup['U_bulk']
        )
        n_atoms = 1 + setup['solvent_pos'].shape[0]
        assert derivs.shape == (n_atoms, 3)

    def test_derivatives_finite(self, setup):
        derivs = setup['calc'].calculate_solvation_derivatives(
            setup['ion_pos'], setup['solvent_pos'], WATER_CHARGES_3,
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3, WATER_LJ_EPS_3),
            setup['U_bulk']
        )
        assert np.all(np.isfinite(derivs))

    def test_derivatives_vs_numerical(self, setup):
        """
        Critical test: analytical derivatives must match numerical
        differentiation for PLUMED biasing accuracy.
        """
        calc = setup['calc']
        ion_pos = setup['ion_pos']
        solvent_pos = setup['solvent_pos']
        U_bulk = setup['U_bulk']
        eps = setup['eps']

        # Analytical
        derivs_analytical = calc.calculate_solvation_derivatives(
            ion_pos, solvent_pos, WATER_CHARGES_3,
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3, WATER_LJ_EPS_3),
            U_bulk
        )

        # Reference s value
        s_ref = calc.calculate_solvation_coordinate(
            ion_pos, solvent_pos, WATER_CHARGES_3,
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3, WATER_LJ_EPS_3),
            U_bulk
        )

        # Numerical derivatives
        derivs_numerical = np.zeros_like(derivs_analytical)

        # w.r.t. ion position
        for j in range(3):
            ion_pert = ion_pos.copy()
            ion_pert[j] += eps
            s_pert = calc.calculate_solvation_coordinate(
                ion_pert, solvent_pos, WATER_CHARGES_3,
                ION_CHARGE, ION_LJ,
                (WATER_LJ_SIGMA_3, WATER_LJ_EPS_3),
                U_bulk
            )
            derivs_numerical[0, j] = (s_pert - s_ref) / eps

        # w.r.t. solvent positions
        for i in range(solvent_pos.shape[0]):
            for j in range(3):
                sp = solvent_pos.copy()
                sp[i, j] += eps
                s_pert = calc.calculate_solvation_coordinate(
                    ion_pos, sp, WATER_CHARGES_3,
                    ION_CHARGE, ION_LJ,
                    (WATER_LJ_SIGMA_3, WATER_LJ_EPS_3),
                    U_bulk
                )
                derivs_numerical[i + 1, j] = (s_pert - s_ref) / eps

        # Compare using np.allclose (combined absolute + relative tolerance)
        # Absolute tolerance handles near-zero components where relative
        # error is meaningless (e.g., y-derivative of atom on x-axis)
        atol = 1e-4
        rtol = setup['rtol']
        assert np.allclose(derivs_analytical, derivs_numerical, atol=atol, rtol=rtol), \
            f"Solvation derivatives mismatch:\n" \
            f"  Analytical:\n{derivs_analytical}\n" \
            f"  Numerical:\n{derivs_numerical}\n" \
            f"  Max abs diff: {np.max(np.abs(derivs_analytical - derivs_numerical)):.2e}"

    def test_ion_derivative_opposes_solvent(self, setup):
        """
        Newton's third law: force on ion should oppose force on solvent
        for pairwise interaction.
        """
        calc = setup['calc']
        ion_pos = np.array([0.0, 0.0, 0.0])
        # Single solvent atom
        solvent_pos = np.array([[4.0, 0.0, 0.0]])

        derivs = calc.calculate_solvation_derivatives(
            ion_pos, solvent_pos, WATER_CHARGES_3[:1],
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3[:1], WATER_LJ_EPS_3[:1]),
            U_bulk=-100.0  # arbitrary nonzero
        )

        # ds/d(ion) should be opposite to ds/d(solvent)
        np.testing.assert_allclose(
            derivs[0], -derivs[1], rtol=1e-10,
            err_msg="Ion and solvent derivatives should be equal and opposite"
        )

    def test_pbc_derivatives(self, setup):
        """Derivatives should be finite and non-zero with PBC."""
        calc = setup['calc']
        box = np.diag([20.0, 20.0, 20.0])
        ion_pos = np.array([1.0, 10.0, 10.0])
        solvent_pos = np.array([[19.0, 10.0, 10.0]])

        derivs = calc.calculate_solvation_derivatives(
            ion_pos, solvent_pos, WATER_CHARGES_3[:1],
            ION_CHARGE, ION_LJ,
            (WATER_LJ_SIGMA_3[:1], WATER_LJ_EPS_3[:1]),
            U_bulk=-100.0,
            box=box
        )

        assert np.all(np.isfinite(derivs))
        assert np.any(np.abs(derivs) > 1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
