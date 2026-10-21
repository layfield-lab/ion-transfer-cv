#!/usr/bin/env python3
"""
Calculate Water Finger Coordinate from GROMACS Trajectory

Main script for computing the water finger coordinate (w) and its analytical
derivatives from GROMACS MD trajectories. Outputs in PLUMED-compatible format
for use in biased MD simulations.

Usage:
    python calculate_cv.py --topology system.tpr --trajectory traj.xtc \\
                          --config config/systems/cl_water_dcm.yaml \\
                          --output cv_output.colvar

Author: Generated for ion transfer PLUMED CV
"""

import argparse
import logging
import yaml
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from collective_variables.water_finger import WaterFingerCoordinate
from collective_variables.solvation import SolvationCoordinate
from collective_variables.derivatives import DerivativeCalculator
from collective_variables.bulk_energy import BulkEnergyCalculator
from collective_variables.trajectory_reader import TrajectoryReader
from plumed.plumed_writer import PlumedColvWriter, PlumedConfigWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_file: str) -> dict:
    """Load YAML configuration file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_file}")
    return config


def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description='Calculate water finger coordinate from GROMACS trajectory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python calculate_cv.py --topology system.tpr --trajectory traj.xtc \\
                        --config config/systems/cl_water_dcm.yaml

  # Process subset of frames
  python calculate_cv.py --topology system.tpr --trajectory traj.xtc \\
                        --config config/systems/cl_water_dcm.yaml \\
                        --start 100 --stop 1000 --stride 10

  # Save to custom output
  python calculate_cv.py --topology system.tpr --trajectory traj.xtc \\
                        --config config/systems/cl_water_dcm.yaml \\
                        --output my_cv.colvar
        """
    )

    parser.add_argument(
        '--topology', '-t',
        required=True,
        help='GROMACS topology file (TPR format)'
    )
    parser.add_argument(
        '--trajectory', '-f',
        required=True,
        help='GROMACS trajectory file (XTC or TRR format)'
    )
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='YAML configuration file'
    )
    parser.add_argument(
        '--output', '-o',
        default='cv_output.colvar',
        help='Output COLVAR file (default: cv_output.colvar)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start frame index (default: 0)'
    )
    parser.add_argument(
        '--stop',
        type=int,
        default=None,
        help='Stop frame index (default: end of trajectory)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Read every Nth frame (default: 1)'
    )
    parser.add_argument(
        '--cvs',
        nargs='+',
        default=['w'],
        choices=['w', 's_W', 's_NB', 'all'],
        help='Which CVs to calculate (default: w). Use "all" for w, s_W, s_NB.'
    )
    parser.add_argument(
        '--U-bulk-water',
        type=float,
        default=None,
        help='Pre-calculated bulk water solvation energy (kcal/mol). '
             'If not provided, will be calculated from trajectory.'
    )
    parser.add_argument(
        '--U-bulk-dcm',
        type=float,
        default=None,
        help='Pre-calculated bulk DCM solvation energy (kcal/mol). '
             'If not provided, will be calculated from trajectory.'
    )
    parser.add_argument(
        '--generate-plumed-input',
        action='store_true',
        help='Generate example PLUMED input file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve CV list
    cv_list = args.cvs
    if 'all' in cv_list:
        cv_list = ['w', 's_W', 's_NB']

    logger.info("="*70)
    logger.info("Collective Variable Calculator")
    logger.info(f"Computing: {', '.join(cv_list)}")
    logger.info("="*70)

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Load trajectory
    try:
        logger.info(f"Loading trajectory: {args.topology}, {args.trajectory}")
        traj_reader = TrajectoryReader(args.topology, args.trajectory)
        logger.info(f"Trajectory has {traj_reader.n_frames} frames, {traj_reader.n_atoms} atoms")
    except Exception as e:
        logger.error(f"Failed to load trajectory: {e}")
        return 1

    # Select ion and water atoms
    try:
        ion_config = config['system']['ion']
        solvent1_config = config['system']['solvent1']

        ion_indices = traj_reader.select_atoms(f"name {ion_config['name']}")
        water_indices = traj_reader.select_atoms(
            f"residue {solvent1_config['name']} and name {solvent1_config['oxygen_name']}"
        )

        if len(ion_indices) == 0:
            logger.error(f"No ions found with name {ion_config['name']}")
            return 1
        if len(water_indices) == 0:
            logger.error(f"No water molecules found")
            return 1

        logger.info(f"Selected {len(ion_indices)} ion(s), {len(water_indices)} water molecules")

        if len(ion_indices) > 1:
            logger.warning(f"Multiple ions found; using first one")
            ion_indices = ion_indices[:1]

        # Select DCM atoms if s_NB is requested
        dcm_indices = None
        if 's_NB' in cv_list:
            solvent2_config = config['system']['solvent2']
            dcm_indices = traj_reader.select_atoms(
                f"residue {solvent2_config['name']}"
            )
            if len(dcm_indices) == 0:
                logger.error(f"No DCM molecules found with residue {solvent2_config['name']}")
                return 1
            logger.info(f"Selected {len(dcm_indices)} DCM atoms")

    except Exception as e:
        logger.error(f"Failed to select atoms: {e}")
        return 1

    # Build force field parameter arrays for solvation coordinates
    ion_charge = ion_config['charge']
    ion_lj = (ion_config['lj_sigma'], ion_config['lj_epsilon'])

    water_charges = None
    water_lj = None
    dcm_charges = None
    dcm_lj = None

    if 's_W' in cv_list:
        # Build per-atom arrays for water oxygens
        n_water = len(water_indices)
        water_charges = np.full(n_water, solvent1_config['charges']['O'])
        water_lj = (
            np.full(n_water, solvent1_config['lj_sigma']['O']),
            np.full(n_water, solvent1_config['lj_epsilon']['O'])
        )

    if 's_NB' in cv_list and dcm_indices is not None:
        solvent2_config = config['system']['solvent2']
        n_dcm = len(dcm_indices)
        # DCM atoms cycle through atom_names pattern
        atom_names = solvent2_config['atom_names']
        n_sites = len(atom_names)
        dcm_charges = np.array([
            solvent2_config['charges'][atom_names[i % n_sites]]
            for i in range(n_dcm)
        ])
        dcm_lj = (
            np.array([
                solvent2_config['lj_sigma'][atom_names[i % n_sites]]
                for i in range(n_dcm)
            ]),
            np.array([
                solvent2_config['lj_epsilon'][atom_names[i % n_sites]]
                for i in range(n_dcm)
            ])
        )

    # Initialize CV calculators
    water_finger = None
    solvation_calc = None

    if 'w' in cv_list:
        water_finger = WaterFingerCoordinate(
            ion_water_offset=config['water_finger']['ion_water_offset'],
            rOO_max=config['water_finger']['rOO_max'],
            bulk_z_cutoff=config['water_finger']['bulk_z_cutoff']
        )

    if 's_W' in cv_list or 's_NB' in cv_list:
        solvation_calc = SolvationCoordinate(
            k_coul=config.get('numerics', {}).get('k_coulomb', 332.0636)
        )

    # Get or calculate bulk energies
    U_bulk_water = args.U_bulk_water
    U_bulk_dcm = args.U_bulk_dcm

    if 's_W' in cv_list and U_bulk_water is None:
        solv_config = config.get('solvation', {})
        U_bulk_water = solv_config.get('U_bulk_water')
        if U_bulk_water is None:
            logger.info("U_bulk_water not provided; will calculate from trajectory bulk region")
            logger.info("(First pass: scanning for bulk frames...)")
            bulk_calc = BulkEnergyCalculator(
                water_region_z=float(config['bulk_solvation']['water_region'].split('<')[1]),
            )
            # Collect bulk frames
            ion_pos_all = []
            water_pos_all = []
            box_all = []
            for frame_idx, time, coords, box in traj_reader.iterate_frames(
                start=args.start, stop=args.stop, step=args.stride
            ):
                ion_pos_all.append(coords[ion_indices[0]])
                water_pos_all.append(coords[water_indices])
                if box is not None:
                    box_all.append(box)
            U_bulk_water, n_bulk, std_err = bulk_calc.calculate_bulk_energy_water(
                np.array(ion_pos_all),
                np.array(water_pos_all),
                water_charges,
                ion_charge, ion_lj, water_lj,
                np.array(box_all) if box_all else None
            )
            if n_bulk == 0:
                logger.error("No bulk water frames found; cannot compute s_W")
                return 1
            logger.info(f"Calculated U_bulk_water = {U_bulk_water:.2f} ± {std_err:.2f} kcal/mol")

    if 's_NB' in cv_list and U_bulk_dcm is None:
        solv_config = config.get('solvation', {})
        U_bulk_dcm = solv_config.get('U_bulk_dcm')
        if U_bulk_dcm is None:
            logger.info("U_bulk_dcm not provided; will calculate from trajectory bulk region")
            bulk_calc = BulkEnergyCalculator(
                dcm_region_z=float(config['bulk_solvation']['dcm_region'].split('>')[1]),
            )
            ion_pos_all = []
            dcm_pos_all = []
            box_all = []
            for frame_idx, time, coords, box in traj_reader.iterate_frames(
                start=args.start, stop=args.stop, step=args.stride
            ):
                ion_pos_all.append(coords[ion_indices[0]])
                dcm_pos_all.append(coords[dcm_indices])
                if box is not None:
                    box_all.append(box)
            U_bulk_dcm, n_bulk, std_err = bulk_calc.calculate_bulk_energy_dcm(
                np.array(ion_pos_all),
                np.array(dcm_pos_all),
                dcm_charges,
                ion_charge, ion_lj, dcm_lj,
                np.array(box_all) if box_all else None
            )
            if n_bulk == 0:
                logger.error("No bulk DCM frames found; cannot compute s_NB")
                return 1
            logger.info(f"Calculated U_bulk_dcm = {U_bulk_dcm:.2f} ± {std_err:.2f} kcal/mol")

    # Initialize COLVAR writer
    try:
        colvar_writer = PlumedColvWriter(args.output, traj_reader.n_atoms)
        colvar_writer.write_header(cv_list)
        logger.info(f"Output file: {args.output}")
    except Exception as e:
        logger.error(f"Failed to initialize output: {e}")
        return 1

    # Process trajectory
    n_processed = 0
    cv_values = {cv: [] for cv in cv_list}

    try:
        for frame_idx, time, coords, box in traj_reader.iterate_frames(
            start=args.start,
            stop=args.stop,
            step=args.stride
        ):
            ion_pos = coords[ion_indices[0]]
            frame_cvs = {}
            frame_derivs = {}

            # Water finger coordinate
            if 'w' in cv_list:
                water_positions = coords[water_indices]
                w, bottleneck = water_finger.calculate(
                    ion_pos, water_positions, box, return_bottleneck=True
                )
                frame_cvs['w'] = w
                cv_values['w'].append(w)

                derivs_w = DerivativeCalculator.water_finger_derivatives(
                    ion_pos, water_positions, bottleneck, box
                )
                frame_derivs['w'] = derivs_w

            # Water solvation coordinate
            if 's_W' in cv_list:
                water_positions = coords[water_indices]
                s_W = solvation_calc.calculate_solvation_coordinate(
                    ion_pos, water_positions, water_charges,
                    ion_charge, ion_lj, water_lj, U_bulk_water, box
                )
                frame_cvs['s_W'] = s_W
                cv_values['s_W'].append(s_W)

                derivs_sW = solvation_calc.calculate_solvation_derivatives(
                    ion_pos, water_positions, water_charges,
                    ion_charge, ion_lj, water_lj, U_bulk_water, box
                )
                frame_derivs['s_W'] = derivs_sW

            # DCM solvation coordinate
            if 's_NB' in cv_list:
                dcm_positions = coords[dcm_indices]
                s_NB = solvation_calc.calculate_solvation_coordinate(
                    ion_pos, dcm_positions, dcm_charges,
                    ion_charge, ion_lj, dcm_lj, U_bulk_dcm, box
                )
                frame_cvs['s_NB'] = s_NB
                cv_values['s_NB'].append(s_NB)

                derivs_sNB = solvation_calc.calculate_solvation_derivatives(
                    ion_pos, dcm_positions, dcm_charges,
                    ion_charge, ion_lj, dcm_lj, U_bulk_dcm, box
                )
                frame_derivs['s_NB'] = derivs_sNB

            # Write to file
            colvar_writer.write_frame(time, frame_cvs, frame_derivs)

            n_processed += 1

            if n_processed % config['output']['print_interval'] == 0:
                status = ", ".join(f"{cv}={frame_cvs[cv]:.4f}" for cv in cv_list)
                logger.info(f"Frame {frame_idx}: {status} ({n_processed} total)")

    except Exception as e:
        logger.error(f"Error processing trajectory: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        colvar_writer.close()
        traj_reader.close()

    # Print summary statistics
    logger.info("="*70)
    logger.info(f"Completed processing {n_processed} frames")
    for cv in cv_list:
        vals = np.array(cv_values[cv])
        unit = " Å" if cv == 'w' else ""
        logger.info(f"{cv} statistics:")
        logger.info(f"  Mean:   {np.mean(vals):.4f}{unit}")
        logger.info(f"  Std:    {np.std(vals):.4f}{unit}")
        logger.info(f"  Min:    {np.min(vals):.4f}{unit}")
        logger.info(f"  Max:    {np.max(vals):.4f}{unit}")
    logger.info("="*70)

    # Generate PLUMED input if requested
    if args.generate_plumed_input:
        try:
            plumed_input = args.output.replace('.colvar', '_plumed.dat')
            PlumedConfigWriter.write_plumed_input(plumed_input)
            logger.info(f"Generated PLUMED input template: {plumed_input}")
        except Exception as e:
            logger.warning(f"Failed to generate PLUMED input: {e}")

    logger.info("Collective variable calculation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
