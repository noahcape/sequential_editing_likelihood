# APC 523 Final Project

This is the repository of the final project for APC 523 of Noah Cape. The project is title: *Reconstructing Phylogenies using DNA Typewriter Data*.

The key files are as follows:

- **./py/model.py** contains the main data abstraction of DNA Typewriter tapes

- **./py/n_model.py** contains the computation of the likelihood over a fixed tree as well as the parameter optimization code (note that ./py/model.py contains stale code *unoptimized* which I compared against n_model.py)

- **./py/tree_search.py** contains code for searching tree space

- **./py/myo.py** contains io code

- Other files contain checks and plotting functions for evaluating and visualizing results

The following executables contain code for submitting jobs to the cluster:

- **test_reconstruction.sh** will reconstruct a small tree
- **process_reconstruction.sh** will evaluate the reconstruction accuracy against the true tree of a larger pre-reconstructed instance

Note that these are small test instances as larger instances take on the order of 10s of hours to complete.

**./src** contains an old implementation of the likelihood computation using Rust as well as some simulation code that was used to generate data for testing.

This codebase requires the following python packages:

| Package           | Version |
|-------------------|---------|
| diffrax           | 0.7.2   |
| equinox           | 0.13.6  |
| jax               | 0.9.2   |
| jaxlib            | 0.9.2   |
| matplotlib        | 3.10.9  |
| networkx          | 3.6.1   |
| numpy             | 2.4.3   |
| optax             | 0.2.8   |
| pandas            | 3.0.2   |
| scipy             | 1.17.1  |
