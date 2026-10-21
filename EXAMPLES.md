# Usage Examples

Detailed examples of how to use the ion transfer collective variables package.

## Table of Contents

1. [Basic Water Finger Calculation](#basic-water-finger-calculation)
2. [Multiple Collective Variables](#multiple-collective-variables)
3. [Working with Configuration Files](#working-with-configuration-files)
4. [PLUMED Integration](#plumed-integration)
5. [Python API Usage](#python-api-usage)
6. [Advanced Analysis](#advanced-analysis)

## Basic Water Finger Calculation

Calculate the water finger coordinate from a GROMACS trajectory:

```bash
python scripts/calculate_cv.py \
  --topology system.tpr \
  --trajectory traj.xtc \
  --config config/systems/cl_water_dcm.yaml \
  --output results/w.colvar
```

This produces a PLUMED-compatible COLVAR file with:
- Time values
- Water finger coordinate (w) for each frame
- Analytical derivatives with respect to all atomic coordinates

### Output Format

```
#! FIELDS time w dw/dx_0 dw/dy_0 dw/dz_0 ... dw/dx_N dw/dy_N dw/dz_N
0.000000 5.234821 -0.00123456 0.00234567 -0.00045678 ...
0.002000 5.241783 -0.00125000 0.00236500 -0.00046250 ...
```

## Multiple Collective Variables

Calculate water finger and both solvation coordinates in a single run:

```bash
python scripts/calculate_cv.py \
  --topology system.tpr \
  --trajectory traj.xtc \
  --config config/systems/cl_water_dcm.yaml \
  --cvs all \
  --U-bulk-water -59.2 \
  --U-bulk-dcm -15.8 \
  --output results/all_cvs.colvar
```

The `all` option calculates:
- **w**: Water finger coordinate
- **s_W**: Solvation coordinate in water
- **s_NB**: Solvation coordinate in organic solvent (DCM)

### With Pre-calculated Bulk Energies

If you've already calculated bulk solvation energies from a separate simulation:

```bash
python scripts/calculate_cv.py \
  --topology system.tpr \
  --trajectory traj.xtc \
  --config config/default_config.yaml \
  --cvs s_W s_NB \
  --U-bulk-water -59.2 \
  --U-bulk-dcm -15.8
```

This skips the first-pass analysis of the trajectory and uses provided values directly.

## Working with Configuration Files

### Using Default Configuration

The default configuration works for many common systems:

```bash
python scripts/calculate_cv.py \
  --topology system.tpr \
  --trajectory traj.xtc \
  --config config/default_config.yaml
```

### Creating a Custom Configuration

Create `config/systems/my_system.yaml`:

```yaml
system:
  ion:
    name: "K"                  # Your ion atom name in GROMACS
    charge: 1.0
    lj_sigma: 3.740            # From force field
    lj_epsilon: 0.100

  solvent1:
    name: "SOL"                # Water
    oxygen_name: "O"
    charges: {O: -0.73, H: 0.365}
    lj_sigma: {O: 3.204, H: 0.0}
    lj_epsilon: {O: 0.160, H: 0.0}

  solvent2:
    name: "NB"                 # Your organic solvent
    atom_names: [C, O, O, H]   # Atoms in periodic order
    charges: {C: 0.6, O: -0.3, H: 0.0}
    lj_sigma: {C: 3.550, O: 2.960, H: 2.500}
    lj_epsilon: {C: 0.200, O: 0.120, H: 0.030}

water_finger:
  ion_water_offset: -0.4
  rOO_max: 3.4
  bulk_z_cutoff: 0.0

solvation:
  energy_cutoff: 20.0

numerics:
  pbc: true
  k_coulomb: 332.0636

output:
  print_interval: 100
  verbose: true
```

Then use it:

```bash
python scripts/calculate_cv.py \
  --topology system.tpr \
  --trajectory traj.xtc \
  --config config/systems/my_system.yaml
```

## PLUMED Integration

### Generate PLUMED Input File

```bash
python scripts/calculate_cv.py \
  --topology system.tpr \
  --trajectory traj.xtc \
  --config config/systems/cl_water_dcm.yaml \
  --cvs w \
  --generate-plumed-input
```

This creates `cv_output_plumed.dat` with example biasing configurations.

### Running Biased MD with PLUMED

1. Generate CV values for your trajectory:
```bash
python scripts/calculate_cv.py \
  --topology system.tpr \
  --trajectory traj.xtc \
  --config config/systems/cl_water_dcm.yaml \
  --output cv_values.colvar
```

2. Create a PLUMED input file (see generated template or create manually)

3. Add to your GROMACS `.mdp` file:
```
plumed           = yes
plumed-file      = plumed.dat
```

4. Run GROMACS with PLUMED:
```bash
gmx mdrun -s system.tpr -c system.gro -plumed plumed.dat ...
```

### Example PLUMED Script

```
# plumed.dat
MOLINFO STRUCTURE=system.tpr

# Load collective variable from file
w: CUSTOM FILE=cv_values.colvar VALUE=w

# Restraint to maintain specific w value
RESTRAINT ARG=w AT=5.0 KAPPA=200.0

# Print CV values
PRINT ARG=w FILE=cv_trajectory.txt STRIDE=100
```

## Python API Usage

### Import and Use Directly

```python
import numpy as np
from collective_variables.water_finger import WaterFingerCoordinate
from collective_variables.solvation import SolvationCoordinate

# Initialize water finger calculator
wf = WaterFingerCoordinate(
    ion_water_offset=-0.4,
    rOO_max=3.4,
    bulk_z_cutoff=0.0
)

# Calculate water finger for a frame
ion_pos = np.array([15.0, 15.0, 0.0])  # Ion position in Ångströms
water_positions = np.array([
    [14.0, 15.0, -2.0],
    [16.0, 15.0, -2.5],
    [15.0, 14.0, -3.0],
    # ... more water oxygen positions
])
box = np.array([30.0, 30.0, 60.0])  # Box dimensions (with PBC)

w, bottleneck = wf.calculate(
    ion_pos, water_positions, box,
    return_bottleneck=True
)
print(f"Water finger coordinate: {w:.4f} Å")
```

### Calculate Derivatives

```python
from collective_variables.derivatives import DerivativeCalculator

# Get analytical derivatives
derivs_w = DerivativeCalculator.water_finger_derivatives(
    ion_pos, water_positions, bottleneck, box
)
print(f"Derivative shape: {derivs_w.shape}")  # (N_atoms, 3)
```

### Solvation Coordinates

```python
# Initialize solvation calculator
solv = SolvationCoordinate(k_coul=332.0636)

# Define charges and LJ parameters
ion_charge = -1.0
ion_lj = (4.410, 0.100)  # (sigma, epsilon)

water_charges = np.full(len(water_positions), -0.73)  # Oxygen charges
water_lj = (
    np.full(len(water_positions), 3.204),  # sigma
    np.full(len(water_positions), 0.160)   # epsilon
)

# Calculate solvation coordinate
U_bulk_water = -59.2  # Pre-calculated bulk energy
s_W = solv.calculate_solvation_coordinate(
    ion_pos, water_positions, water_charges,
    ion_charge, ion_lj, water_lj, U_bulk_water, box
)
print(f"Water solvation coordinate: {s_W:.4f}")
```

## Advanced Analysis

### Processing Trajectory Subset

Calculate CVs for every 10th frame between frames 100-500:

```bash
python scripts/calculate_cv.py \
  --topology system.tpr \
  --trajectory traj.xtc \
  --config config/systems/cl_water_dcm.yaml \
  --start 100 --stop 500 --stride 10 \
  --output cv_subset.colvar
```

### Verbose Output for Debugging

```bash
python scripts/calculate_cv.py \
  --topology system.tpr \
  --trajectory traj.xtc \
  --config config/systems/cl_water_dcm.yaml \
  --verbose
```

Shows detailed logging:
```
2026-04-04 12:00:00,123 - __main__ - INFO - ======================================================================
2026-04-04 12:00:00,123 - __main__ - INFO - Collective Variable Calculator
2026-04-04 12:00:00,123 - __main__ - INFO - Computing: w
2026-04-04 12:00:00,123 - __main__ - INFO - ======================================================================
2026-04-04 12:00:00,234 - __main__ - INFO - Loaded configuration from config/systems/cl_water_dcm.yaml
2026-04-04 12:00:00,456 - __main__ - INFO - Loading trajectory: system.tpr, traj.xtc
...
```

### Calculating Bulk Energies Automatically

If not providing `--U-bulk-water`:

```bash
python scripts/calculate_cv.py \
  --topology system.tpr \
  --trajectory traj.xtc \
  --config config/systems/cl_water_dcm.yaml \
  --cvs s_W
```

The script automatically:
1. Scans the trajectory for bulk water frames (z < -10 Å)
2. Calculates average solvation energy in bulk
3. Uses this for normalizing s_W values
4. Prints statistics: `Calculated U_bulk_water = -59.24 ± 0.45 kcal/mol`

### Analyzing Results

```python
import numpy as np
import matplotlib.pyplot as plt

# Load COLVAR file
data = np.loadtxt('cv_output.colvar', skiprows=2)
time = data[:, 0]
w_values = data[:, 1]

# Plot water finger vs time
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, w_values, linewidth=1.5)
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Water Finger Coordinate (Å)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('w_trajectory.png', dpi=150)

# Print statistics
print(f"Mean w: {np.mean(w_values):.4f} Å")
print(f"Std w: {np.std(w_values):.4f} Å")
print(f"Min w: {np.min(w_values):.4f} Å")
print(f"Max w: {np.max(w_values):.4f} Å")
```

## Testing

Run the test suite to verify installation:

```bash
# Quick test
pytest tests/ -v

# With coverage report
pytest tests/ --cov=collective_variables --cov-report=html

# Run specific test
pytest tests/test_water_finger.py::TestWaterFingerCoordinate::test_simple_connected_path -v
```
