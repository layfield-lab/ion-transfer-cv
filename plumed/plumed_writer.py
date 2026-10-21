"""
PLUMED COLVAR File Writer

Generates PLUMED-compatible COLVAR files with collective variable values
and analytical derivatives.

PLUMED COLVAR Format:
  Lines starting with '#!' contain metadata (field names, periodicity, etc.)
  Each data line contains: time cv_value der_x_1 der_y_1 der_z_1 ... (derivatives)
"""

import numpy as np
from typing import List, Dict, Optional, TextIO, Tuple
import logging

logger = logging.getLogger(__name__)


class PlumedColvWriter:
    """
    Write collective variables in PLUMED COLVAR format.

    Parameters:
        filename (str): Output file path
        n_atoms (int): Total number of atoms (for derivative dimensionality)
    """

    def __init__(self, filename: str, n_atoms: int):
        """Initialize COLVAR writer."""
        self.filename = filename
        self.n_atoms = n_atoms
        self.file = None
        self.header_written = False

    def write_header(
        self,
        cv_names: List[str],
        periodic: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Write PLUMED COLVAR header.

        Parameters:
            cv_names (List[str]): Names of collective variables
            periodic (Dict): Dictionary of {cv_name: (min, max)} for periodic CVs
        """

        self.file = open(self.filename, 'w')

        # Build field names
        fields = ['time']
        for cv_name in cv_names:
            fields.append(cv_name)
            # Add derivative fields for each atom dimension
            for i in range(self.n_atoms):
                fields.append(f"d{cv_name}/dx_{i}")
                fields.append(f"d{cv_name}/dy_{i}")
                fields.append(f"d{cv_name}/dz_{i}")

        # Write field header
        self.file.write('#! FIELDS ')
        self.file.write(' '.join(fields))
        self.file.write('\n')

        # Write set header (optional, for PLUMED metadata)
        self.file.write('#! SET num_expansions 1\n')

        # Write periodicity if specified
        if periodic:
            for cv_name, (cv_min, cv_max) in periodic.items():
                self.file.write(f'#! PERIODIC {cv_name} {cv_min:.6f} {cv_max:.6f}\n')

        self.file.flush()
        self.header_written = True
        logger.info(f"COLVAR header written: {len(fields)} fields")

    def write_frame(
        self,
        time: float,
        cv_values: Dict[str, float],
        cv_derivatives: Dict[str, np.ndarray]
    ):
        """
        Write one frame of CV data.

        Parameters:
            time (float): Simulation time (ps)
            cv_values (Dict): {cv_name: value}
            cv_derivatives (Dict): {cv_name: (n_atoms, 3) array of derivatives}
        """

        if not self.header_written:
            raise ValueError("Header not written; call write_header() first")

        # Start with time
        data = [f"{time:.6f}"]

        # Get ordered list of CV names (should match header order)
        cv_names = list(cv_values.keys())

        for cv_name in cv_names:
            # CV value
            data.append(f"{cv_values[cv_name]:.8f}")

            # Derivatives for all atoms
            derivs = cv_derivatives[cv_name]  # (n_atoms, 3)

            for i in range(self.n_atoms):
                for j in range(3):
                    data.append(f"{derivs[i, j]:.8f}")

        # Write line
        self.file.write(' '.join(data))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        """Close the output file."""
        if self.file is not None:
            self.file.close()
            logger.info(f"COLVAR file closed: {self.filename}")

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.close()


class PlumedConfigWriter:
    """
    Generate PLUMED input file configuration for water finger CV.

    This helps users set up biasing with the water finger coordinate
    in their PLUMED scripts.
    """

    @staticmethod
    def write_plumed_input(
        filename: str,
        cv_name: str = "w",
        biasing_kappa: float = 100.0,
        biasing_center: float = 5.0,
        use_well_tempered: bool = True,
        biasfactor: Optional[float] = None,
        temperature: float = 300.0
    ):
        """
        Write example PLUMED input file.

        This generates a PLUMED script that can be used to bias the
        water finger coordinate during MD simulations.

        Parameters:
            filename (str): Output PLUMED input file
            cv_name (str): Name of the CV
            biasing_kappa (float): Spring constant for biasing potential (kcal/mol/Ų)
            biasing_center (float): Center of harmonic bias (Ångströms)
            use_well_tempered (bool): Use well-tempered metadynamics
            biasfactor (float): Bias factor for well-tempered MD (default: 6.0)
            temperature (float): Simulation temperature (K)
        """

        if biasfactor is None:
            biasfactor = 6.0

        with open(filename, 'w') as f:
            f.write("# PLUMED Input File - Water Finger CV Biasing\n")
            f.write("# This file should be read via: 'plumed_load ./plumed.dat'\n")
            f.write("# in your GROMACS input file\n\n")

            f.write("# =====================================================\n")
            f.write("# Collective Variables\n")
            f.write("# =====================================================\n\n")

            f.write(f"# Water finger coordinate\n")
            f.write(f"{cv_name}: CUSTOM ...\n")
            f.write(f"  FILE=cv_output.colvar\n")
            f.write(f"  VALUE=w\n")
            f.write(f"  PERIODIC=NO\n")
            f.write(f"... CUSTOM\n\n")

            f.write("# Print CV for inspection\n")
            f.write(f"PRINT ARG={cv_name} FILE=cv_trajectory.txt STRIDE=1\n\n")

            f.write("# =====================================================\n")
            f.write("# Biasing Potential\n")
            f.write("# =====================================================\n\n")

            f.write("# Harmonic restraint centered at fixed point\n")
            f.write(f"RESTRAINT ARG={cv_name} ")
            f.write(f"AT={biasing_center:.2f} ")
            f.write(f"KAPPA={biasing_kappa:.1f}\n\n")

            if use_well_tempered:
                f.write("# Alternative: Well-tempered metadynamics\n")
                f.write("# (uncomment to use instead of RESTRAINT)\n")
                f.write(f"# METAD ARG={cv_name} PACE=500 HEIGHT=1.2 ")
                f.write(f"SIGMA=0.5 BIASFACTOR={biasfactor:.1f} ")
                f.write(f"TEMP={temperature:.0f}\n")

        logger.info(f"PLUMED input template written to {filename}")

    @staticmethod
    def write_bash_driver_script(filename: str, topology: str, trajectory: str):
        """
        Write bash script for running GROMACS with PLUMED biasing.

        Parameters:
            filename (str): Output bash script filename
            topology (str): GROMACS topology file (TPR)
            trajectory (str): Initial trajectory/configuration file (GRO)
        """

        with open(filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Script to run GROMACS with PLUMED biasing\n\n")

            f.write("TOPOLOGY=system.tpr\n")
            f.write("CONFIG=system.gro\n")
            f.write("PLUMED_INPUT=plumed.dat\n")
            f.write("OUTNAME=simulation_biased\n")
            f.write("TIME=10000  # ps\n")
            f.write("TEMP=300    # K\n\n")

            f.write("# Run GROMACS with PLUMED\n")
            f.write("gmx_mpi mdrun \\\n")
            f.write("  -s ${TOPOLOGY} \\\n")
            f.write("  -c ${CONFIG} \\\n")
            f.write("  -plumed ${PLUMED_INPUT} \\\n")
            f.write("  -o ${OUTNAME}.trr \\\n")
            f.write("  -x ${OUTNAME}.xtc \\\n")
            f.write("  -cpo ${OUTNAME}.cpt \\\n")
            f.write("  -e ${OUTNAME}.edr \\\n")
            f.write("  -g ${OUTNAME}.log \\\n")
            f.write("  -maxh 24.0 \\\n")
            f.write("  -nt 0 \\\n")
            f.write("  -ntmpi 4\n\n")

            f.write("echo 'GROMACS+PLUMED simulation completed'\n")

        logger.info(f"Bash driver script written to {filename}")
