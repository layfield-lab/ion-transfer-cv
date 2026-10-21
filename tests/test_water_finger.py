"""
Unit Tests for Water Finger Coordinate

Tests the graph-based water finger coordinate calculation against
known configurations and reference values from literature.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collective_variables.water_finger import WaterFingerCoordinate


class TestWaterFingerCoordinate:
    """Test suite for water finger coordinate calculation."""

    @pytest.fixture
    def calculator(self):
        """Create a water finger coordinate calculator."""
        return WaterFingerCoordinate(
            ion_water_offset=-0.4,
            rOO_max=3.4,
            bulk_z_cutoff=0.0
        )

    def test_initialization(self, calculator):
        """Test that calculator initializes correctly."""
        assert calculator.ion_water_offset == -0.4
        assert calculator.rOO_max == 3.4
        assert calculator.bulk_z_cutoff == 0.0

    def test_simple_connected_path(self, calculator):
        """
        Test a simple case where ion is connected to bulk water.

        Setup:
          - Ion at origin
          - Water molecules forming continuous chain to bulk
          - w should be approximately the distance of the bottleneck
        """
        ion_pos = np.array([0.0, 0.0, 10.0])

        # Create linear chain of waters from ion to bulk
        water_positions = np.array([
            [0.0, 0.0, 6.0],   # Close to ion
            [0.0, 0.0, 3.0],   # Middle
            [0.0, 0.0, 0.0],   # At interface
            [0.0, 0.0, -2.0],  # In bulk
            [0.0, 0.0, -6.0],  # Deep in bulk
        ])

        w = calculator.calculate(ion_pos, water_positions)

        # Should be relatively small (waters close together)
        assert w > 0, "w should be positive"
        assert w < 10.0, "w should be less than maximum distance"
        assert np.isfinite(w), "w should be finite"

    def test_disconnected_configuration(self, calculator):
        """
        Test a disconnected configuration.

        Setup:
          - Ion in organic phase
          - Waters only in aqueous phase
          - w should be large (ion not connected to waters)
        """
        ion_pos = np.array([0.0, 0.0, 15.0])  # In DCM bulk

        # All waters far away in water bulk
        water_positions = np.array([
            [5.0, 0.0, -5.0],
            [0.0, 5.0, -6.0],
            [-5.0, 0.0, -4.0],
            [0.0, -5.0, -7.0],
            [0.0, 0.0, -8.0],
        ])

        w, bottleneck = calculator.calculate(ion_pos, water_positions, return_bottleneck=True)

        # Should be large
        assert w > 10.0, "Disconnected configuration should have large w"

    def test_periodic_boundary_conditions(self, calculator):
        """
        Test that PBC are handled correctly.

        Setup:
          - Ion wrapped across boundary
          - Waters positioned to appear close with PBC
        """
        # Box of size 20x20x20
        box = np.array([
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 20.0]
        ])

        # Ion near one edge
        ion_pos = np.array([1.0, 10.0, 10.0])

        # Water on opposite edge (appears close with PBC)
        water_positions = np.array([
            [19.0, 10.0, 10.0],  # Distance should be 2 Å with PBC
            [19.0, 10.0, -5.0],  # In bulk with Z
        ])

        w = calculator.calculate(ion_pos, water_positions, box=box)

        # Should work with PBC
        assert w < 5.0, "PBC should make distances smaller"

    def test_derivative_cache(self, calculator):
        """Test that bottleneck pair is cached for derivatives."""
        ion_pos = np.array([0.0, 0.0, 5.0])
        water_positions = np.array([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -3.0],
        ])

        w, bottleneck = calculator.calculate(
            ion_pos, water_positions, return_bottleneck=True
        )

        # Check that bottleneck is cached
        cached_bottleneck = calculator.get_bottleneck_pair()
        assert cached_bottleneck == bottleneck
        assert cached_bottleneck is not None

    def test_symmetry(self, calculator):
        """
        Test that w is independent of water molecule ordering.

        Same configuration, different ordering should give same w.
        """
        ion_pos = np.array([0.0, 0.0, 5.0])

        waters_order1 = np.array([
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 1.0],
            [0.0, 0.5, -2.0],
        ])

        waters_order2 = np.array([
            [0.0, 0.5, -2.0],
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 1.0],
        ])

        w1 = calculator.calculate(ion_pos, waters_order1)
        w2 = calculator.calculate(ion_pos, waters_order2)

        assert np.isclose(w1, w2, rtol=1e-6), \
            f"w should be independent of water ordering: {w1} vs {w2}"

    def test_hydrogen_bond_criteria(self, calculator):
        """
        Test that H-bond detection uses correct distance threshold.

        Waters separated by > rOO_max should not be directly connected
        in a proper configuration, forcing longer path.
        """
        ion_pos = np.array([0.0, 0.0, 10.0])

        # Two disconnected clusters
        water_positions = np.array([
            # Cluster 1: near ion
            [0.0, 0.0, 7.0],
            [0.2, 0.0, 6.8],
            # Cluster 2: far from cluster 1, in bulk
            [0.0, 0.0, -5.0],
            [0.2, 0.0, -5.2],
        ])

        w = calculator.calculate(ion_pos, water_positions)

        # Should reflect that clusters are disconnected
        assert w > 4.0, "Disconnected clusters should have larger w"

    def test_numerical_stability(self, calculator):
        """
        Test numerical stability with various configurations.

        Should not produce NaN or Inf values.
        """
        ion_pos = np.array([0.0, 0.0, 10.0])

        # Random but reasonable configuration
        np.random.seed(42)
        water_positions = np.random.randn(20, 3) * 5 + np.array([0, 0, 0])

        # Add some waters in bulk region
        water_positions[-3:, 2] = -5.0

        w = calculator.calculate(ion_pos, water_positions)

        assert np.isfinite(w), f"w should be finite, got {w}"
        assert w > 0, "w should be positive"


class TestWaterFingerIntegration:
    """Integration tests for water finger coordinate."""

    def test_sequential_frames(self):
        """
        Test calculating w over a sequence of frames.

        Simulates evolving from intact to broken finger configuration.
        """
        calculator = WaterFingerCoordinate()

        # Frame 1: Ion in water, connected to bulk
        ion_pos_1 = np.array([0.0, 0.0, 5.0])
        water_pos_1 = np.array([
            [0.0, 0.0, 2.5],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, -2.0],
        ])
        w_1 = calculator.calculate(ion_pos_1, water_pos_1)

        # Frame 2: Ion moves to organic phase, water finger stretches
        ion_pos_2 = np.array([0.0, 0.0, 8.0])  # Moved away
        water_pos_2 = np.array([
            [0.0, 0.0, 2.5],  # Still same position (didn't follow)
            [0.1, 0.0, 1.0],
            [0.0, 0.1, -2.0],
        ])
        w_2 = calculator.calculate(ion_pos_2, water_pos_2)

        # w should increase as ion moves away
        assert w_2 > w_1, f"w should increase as ion moves away: {w_1} -> {w_2}"

    def test_literature_values(self):
        """
        Test against known values from Kikkawa et al. (2015).

        Expected approximate values from the paper:
          - Intact finger: w ≈ 2.5-3.0 Å
          - Broken finger: w > 5.0 Å
        """
        calculator = WaterFingerCoordinate(
            ion_water_offset=-0.4,
            rOO_max=3.4,
            bulk_z_cutoff=0.0
        )

        # Approximate "intact" configuration
        ion_pos = np.array([0.0, 0.0, 5.0])
        waters = np.array([
            [0.0, 0.0, 2.75],  # O-O ≈ 2.75 Å (typical H-bond)
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -2.75],
            [0.0, 0.0, -5.5],
        ])

        w = calculator.calculate(ion_pos, waters)

        # Should be in reasonable range for intact finger
        assert 1.0 < w < 5.0, f"Intact configuration w should be 1-5 Å, got {w}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
