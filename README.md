# Fast multi-compartment microstructure fingerprinting

Welcome to the official repository for the paper "Fast Multi-Compartment Microstructure Fingerprinting Using Deep Neural Networks". This repository includes a pre-calculated dictionary of microstructural features, PyTorch model weights, and Python scripts for voxel simulations, model training, and data analysis.

## Usage

### Setting Up the Environment

To utilize the scripts and resources provided in this repository, you must first set up the appropriate environment by installing the associated Python package. This package includes all the code for the Hybrid and Fully-Learned methods discussed in our paper.

````bash
pip install fastmf==1.0
````

### Generating Synthetic Data

Slurm scripts used for the generation of synthetic datasets and the training of both the Hybrid and Fully-Learned models are organized within the `preprocessing` directory:

- **scheme-HCP Subdirectory**: Contains Slurm scripts for generating synthetic data based on the Human Connectome Project (HCP) scheme and for training both primary models.
- **scheme-Clinical Subdirectory**: Contains Slurm scripts for generating synthetic data based on the Clinical scheme and for training the Hybrid Clinical model, as detailed in Experiment 2 of our study.

### Model Weights

The `model_weights` folder houses the pre-trained weights for both the Fully-Learned and Hybrid methods utilized in the paper.

# Advanced Usage

For those interested in exploring or modifying the source code of the Fully-Learned and Hybrid methods, the complete codebase is accessible on GitHub: 
[FastMF Python Source Code](https://github.com/Hyedryn/FastMF_public) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10643497.svg)](https://doi.org/10.5281/zenodo.10643497)



