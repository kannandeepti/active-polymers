r"""
Various ways of creating neighbor lists in python
-------------------------------------------------
Existing packages:
- lammps python neighbor list
- GriSPy (grid search algorithm): https://grispy.readthedocs.io/en/latest/index.html
    - builds a grid and queries it for neighbors
- ASE neighbor list: https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#
    - collision detector, does not use jit or any fancy algorithms, but takes a radial cutoff and
    skin as parameters (Exactly what we want. could take this code and make it jit compatible)
- PyNNDescent: https://pynndescent.readthedocs.io/en/latest/index.html
    - uses k nearest neighbor graph construction and jit under the hood
    - But we don't want k nearest neighbors, we want neighbors within a cutoff...

Problem is these functions need to be @jit-able; otherwise, cannot integrate with BD code.
Instead, start by implementing cell-linked Verlet list. Let's time this
and compare to existing packages.
"""
import numpy as np
from numba import njit, types, typed, int64, float64
from numba.experimental import jitclass

#dictionary specifying types of all jitclass attributes
spec = [
    ('skinsq' , float64),
    ('rl' , float64),
    ('rlsq' , float64),
    ('box_size' , float64),
    ('num_cells' , int64),
    ('grid_size' , float64),
    ('neighbors' , int64[:, :]),
    ('xl', float64[:, :])
]

@jitclass(spec)
class NeighborList:
    """ This class utilizes the cell-linked list and Verlet list algorithms to construct
    a neighbor list containing the connectivity of the N monomers of the polymer. We first
    partition the simulation domain into a 3-dimensional grid and assign all monomers to
    their associated cells, storing this information in a dictionary. We next create a 
    cell linked list `cll`, where cll[i] points to the neighbor of i or points to -1 if i
    has no other neighbors within the same cell.

    To update the neighbor list, we then loop through all N particles and append all neighbors of i
    in the same cell and in the 13 neighboring cells (technically 26 neighboring cells, but only
    need to loop through half since we are computing pairwise interactions).
    """

    def __init__(self, rcut, rskin, box_size):
        self.skinsq = rskin ** 2
        self.rl = rcut + rskin
        self.rlsq = self.rl ** 2 #precalculated for future use
        self.box_size = box_size
        self.num_cells = int(box_size // self.rl)
        self.grid_size = box_size / self.num_cells
        #indices of 13 neighboring cells (relative to current cell)
        #neighbors in current layer
        neighbors = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0]]
        for dispx in range(-1, 2):
            for dispy in range(-1, 2):
                index = [dispx, dispy, 1]
                neighbors.append(index)
        self.neighbors = np.array(neighbors)

    def checkNL(self, X):
        """ Returns true if maximum displacement of a bead exceeds the size of the skin."""
        disp = X - self.xl
        sqdisp = np.sum(disp * disp, axis=-1)
        return (np.max(sqdisp) > self.skinsq)

    def updateNL(self, X):
        N, ndim = X.shape
        #data structures
        cll = np.ones(N, dtype='int') #list of length number of particles
        grid_dict = typed.Dict() #look up table containing first particle ID in each cell

        #assign N monomers to their respective grid cells
        #account for periodic BCs
        X = X - np.rint(X / self.box_size) * self.box_size
        inds = np.ones(X.shape, dtype='int')
        inds[:, :] = (X + self.box_size/2) // self.grid_size
        for i in range(N):
            key = (inds[i,0], inds[i,1], inds[i,2])
            if key in grid_dict:
                cll[i] = grid_dict[key]
            else:
                cll[i] = -1
            grid_dict[key] = i

        #cl[i+1] points to index in nl of last neighbor of bead i
        cl = np.zeros(N + 1, )
        nl = []
        k = 0

        for i in range(N):
            cl[i] = k
            j = cll[i]

            # get all neighbors of i in current cell
            while (j > -1):
                diff = X[i] - X[j]
                rijsq = diff @ diff
                if rijsq <= self.rlsq:
                    nl.append(j)
                    k += 1
                j = cll[j]

            # loop over all particles in neighboring cells (13)
            # modding accounts for periodic BCs
            ninds = (inds[i] + self.neighbors) % self.num_cells
            for jC in range(ninds.shape[0]):
                key = (ninds[jC, 0], ninds[jC, 1], ninds[jC, 2])
                if key in grid_dict:
                    j = grid_dict[key]
                while (j > -1):
                    diff = X[i] - X[j]
                    rijsq = diff @ diff
                    if rijsq <= self.rlsq:
                        nl.append(j)
                        k += 1
                    j = cll[j]
        cl[N] = len(nl) - 1
        #save positions of particles at the time the neighbor list was built
        self.xl = X
        return cl, nl



