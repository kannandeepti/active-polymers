r"""
Simulations of active polymers with correlated noise
----------------------------------------------------
This package can be used to perform Brownian dynamics simulations of an active Rouse polymer.
At equilibrium, all monomers of the chain experience the same magnitude
of stochastic kicks, i.e. are at uniform temperature. Out of equilibrium,
activity along the polymer (i.e. action of motor proteins and enzymes
that walk along the DNA) can induce stronger kicks along certain regions
of the chain, which can be modeled as a local increase in the effective
temperature of the monomer. More generally, activity at different regions
of the polymer could be correlated, for example, due to an enhancer regulating
two promoters, or a master transcription factor turning on multiple genes.

Here we explore a minimal model in which activity is encoded via correlations
in athermal flucutations experienced by monomers of a Rouse chain. The act_pol.bdsim
module contains the code to simulate a Rouse polymer with correlated noise subject to
forces due to confinement and self-avoidance (via a WCA potential or a soft core repulsive
potential). Distance calculations are simplified through the use of neighbor lists.
The act_pol.analysis module contains various scripts to analyze the output of the simulation.
The act_pol.analysis.analyze module allows one to compute the mean distance
between all pairs of monomers, as well as contact probabilities. The act_pol.analysis.msd module
can be used to study dynamics of correlated active polymers, such as the relaxation dynamics of
the polymer center of mass or the mean squared displacement of individual monomers.

"""
__version__ = "0.1.0"

import numpy as np
import numpy as np
import scipy
import scipy.special
from numba import jit, njit
import mpmath
