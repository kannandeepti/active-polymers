r"""
Simulations of active polymers with correlated noise
----------------------------------------------------
This package can be used to perform Brownian dynamics simulations of a Rouse polymer
driven by correlated, active forces which can vary in magnitude along the chain [Goychuk22]_.
At equilibrium, all monomers of the chain experience the same magnitude
of stochastic kicks, i.e. are at uniform temperature. Out of equilibrium,
activity along the polymer (i.e. due to forces exerted by motor proteins or energy
dissipated by nonequilibrium reactions) can induce stronger kicks along certain regions
of the chain, which can be modeled as a local increase in the effective
temperature of the monomer. More generally, active forces at different regions
of the polymer could be correlated, for example, due to the coordinated transcription of an
enhancer and its cognate promoter, or the coordinated activation of multiple genes by common
transcription factors.

The act_pol.bdsim module contains the code to simulate a Rouse polymer with correlated noise
along with other deterministic forces, such as additional harmonic bonds coupling distinct monomers,
a soft elliptical confinement, and self-avoidance (via a soft core repulsive
potential). Distance calculations are simplified through the use of neighbor lists.
The act_pol.analysis module contains various scripts to analyze the output of the simulation.
The act_pol.analysis.contacts module allows one to compute the mean distance
between all pairs of monomers, as well as contact probabilities from ensemble and time-averaged
simulation data. The act_pol.analysis.msd module
can be used to study dynamics of correlated active polymers, such as the relaxation dynamics of
the polymer center of mass or the mean squared displacement of individual monomers.

.. rubric:: References

.. [Goychuk22] A. Goychuk, D. Kannan, A. K. Chakraborty, and M. Kardar. *Polymer folding through active processes recreates features of genome organization* bioRxiv (2022) https://doi.org/10.1101/2022.12.24.521789

"""
