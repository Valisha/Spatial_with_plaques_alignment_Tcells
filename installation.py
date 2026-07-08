#!/usr/bin/env python3

"""
Setup instructions for the spatial transcriptomics Python environment.

Run these commands in your terminal before using this project:

# Activate the pipenv environment
pipenv shell

# Install required packages
pipenv install numpy scipy pandas scikit-learn matplotlib seaborn scanpy squidpy anndata umap-learn louvain leidenalg scvelo statsmodels h5py zarr dask numba scikit-image python-igraph networkx

Alternatively, if you already have a virtual environment activated, use:

python -m pip install numpy scipy pandas scikit-learn matplotlib seaborn scanpy squidpy anndata umap-learn louvain leidenalg scvelo statsmodels h5py zarr dask numba scikit-image python-igraph networkx

This script does not execute installation itself; it documents the environment setup steps.
"""

import importlib

REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
    "scanpy",
    "squidpy",
    "anndata",
    "umap",
    "louvain",
    "leidenalg",
    "scvelo",
    "statsmodels",
    "h5py",
    "zarr",
    "dask",
    "numba",
    "skimage",
    "igraph",
    "networkx",
]


def check_packages():
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    return missing


if __name__ == "__main__":
    missing_packages = check_packages()
    if missing_packages:
        print("The following packages are not installed:")
        for pkg in missing_packages:
            print(f" - {pkg}")
        print("\nInstall them using the commands in the script header.")
    else:
        print("All required spatial transcriptomics packages are installed.")
