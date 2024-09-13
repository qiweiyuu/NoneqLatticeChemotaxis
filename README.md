# Nonequilibrium lattice model of the chemosensory array

This repository contains code and data for a nonequilibrium lattice model of the chemosensory array, which is described in the paper "Time-reversal symmetry breaking in the chemosensory array reveals mechanisms for asymmetric switching and dissipation-enhanced cooperative sensing" by David Hathcock, Qiwei Yu, and Yuhai Tu, which is available on the arXiv at [https://arxiv.org/abs/2312.17424](https://arxiv.org/abs/2312.17424).

The dynamics of the nonequilibrium model is simulated using the Gillespie algorithm, which is implemented in `ising_neq_v6_scan_functions_lax.py`. Its usage is demonstrated in a Jupyter notebook `ising_neq_v6_demo.ipynb`. The code requires standard python libraries, including `numpy`, `scipy`, and `matplotlib`, as well as [`Jax`](https://jax.readthedocs.io/en/latest/index.html). 

Additionally, each folder contains the data and code snippets for reproducing the main figures in the paper (Figure 2-5).