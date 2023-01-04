# Simulations of active polymers with correlated noise

This package can be used to perform Brownian dynamics simulations of a Rouse polymer
driven by correlated, active forces which can vary in magnitude along the chain.

The act_pol.bdsim module contains the code to simulate a Rouse polymer with correlated noise
along with other deterministic forces, such as additional harmonic bonds coupling distinct monomers and a soft elliptical confinement. The act_pol.analysis module contains various scripts to analyze the output of the simulation.
In particular, the act_pol.analysis.contacts module allows one to compute the mean distance
between all pairs of monomers, as well as contact probabilities from ensemble and time-averaged
simulation data. 

A. Goychuk, D. Kannan, A. K. Chakraborty, and M. Kardar. Polymer folding through active processes recreates features of genome organization. *bioRxiv* (2022) https://doi.org/10.1101/2022.12.24.521789


