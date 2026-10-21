#!/usr/bin/env python
"""Setup configuration for ion transfer collective variables package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = [line.strip() for line in requirements_file.read_text(encoding="utf-8").split('\n')
                if line.strip() and not line.startswith('#')]

setup(
    name="ion-transfer-cv",
    version="0.1.0",
    description="Collective variables for ion transfer across liquid-liquid interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Joshua Layfield",
    url="https://github.com/layfield-lab/ion-transfer-cv",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "calculate-cv=scripts.calculate_cv:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="molecular-dynamics collective-variables ion-transfer PLUMED GROMACS",
)
