import numpy as np
#import networkx as nx
from collections import defaultdict 
#import MDAnalysis as mda
from MDAnalysis.lib.mdamath import triclinic_vectors

def make_whole(atomgroup, reference_atom=None, inplace=True):
    """Move all atoms in a single molecule so that bonds don't split over
    images.

    This function is most useful when atoms have been packed into the primary
    unit cell, causing breaks mid molecule, with the molecule then appearing
    on either side of the unit cell. This is problematic for operations
    such as calculating the center of mass of the molecule. ::

       +-----------+     +-----------+
       |           |     |           |
       | 6       3 |     |         3 | 6
       | !       ! |     |         ! | !
       |-5-8   1-2-| ->  |       1-2-|-5-8
       | !       ! |     |         ! | !
       | 7       4 |     |         4 | 7
       |           |     |           |
       +-----------+     +-----------+


    Parameters
    ----------
    atomgroup : AtomGroup
        The :class:`MDAnalysis.core.groups.AtomGroup` to work with.
        The positions of this are modified in place.  All these atoms
        must belong to the same molecule or fragment.
    reference_atom : :class:`~MDAnalysis.core.groups.Atom`
        The atom around which all other atoms will be moved.
        Defaults to atom 0 in the atomgroup.
    inplace : bool, optional
        If ``True``, coordinates are modified in place.

    Returns
    -------
    coords : numpy.ndarray
        The unwrapped atom coordinates.

    Raises
    ------
    NoDataError
        There are no bonds present.
        (See :func:`~MDAnalysis.topology.core.guess_bonds`)

    ValueError
        The algorithm fails to work.  This is usually
        caused by the atomgroup not being a single fragment.
        (ie the molecule can't be traversed by following bonds)


    Example
    -------
    Make fragments whole::

        from MDAnalysis.lib.mdamath import make_whole

        # This algorithm requires bonds, these can be guessed!
        u = mda.Universe(......, guess_bonds=True)

        # MDAnalysis can split AtomGroups into their fragments
        # based on bonding information.
        # Note that this function will only handle a single fragment
        # at a time, necessitating a loop.
        for frag in u.atoms.fragments:
            make_whole(frag)

    Alternatively, to keep a single atom in place as the anchor::

        # This will mean that atomgroup[10] will NOT get moved,
        # and all other atoms will move (if necessary).
        make_whole(atomgroup, reference_atom=atomgroup[10])


    See Also
    --------
    :meth:`MDAnalysis.core.groups.AtomGroup.unwrap`


    .. versionadded:: 0.11.0
    .. versionchanged:: 0.20.0
        Inplace-modification of atom positions is now optional, and positions
        are returned as a numpy array.
    """
       
    n_atoms = atomgroup.n_atoms
    positions = atomgroup.atoms.positions
    bonds = atomgroup.bonds.indices
    
    # Make a graph of the structure (better if this were part of a core class)
    G = bond_dict(bonds)
    
    # This is an expensive check!
    #t = G.bonded_components()
    #if len(t)!=1:
    #    raise ValueError("AtomGroup was not contiguous from bonds, process failed")
    
    if reference_atom is None:
        ref = 0
    else:
        ref = reference_atom.index
        if ref not in G.keys():
            raise ValueError("Reference atom not in atomgroup")           

    
    # Calculate the smallest minimum cell vector length 0.5 * lmin
    #boxlengths2 = np.sum(boxvects ** 2, axis=1) 
    #minbl2 = 0.25 * boxlengths2.min() # (0.5*sqrt(lmin**2))**2 = 0.25*lmin**2 
    minbl2 = ((0.5 * atomgroup.dimensions) ** 2)[:3].min()
    # Instead of using the function minimum_image_triclinic for each atom we 
    # create an array of the 27 vectors required to move to the periodic images.
    # The minimum_image_triclinic perfomrs a reduced matrix multiplication for 
    # every atom to get a set of 27 constant shift vectors 
    # (cell vectors * shifts), this code performs it once.
    s = [-1,0,1]
    x, y, z = np.meshgrid(s, s, s)
    # array of cell shifts to find images
    shiftlist = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    # array of cartesian vectors to find images
    boxvects = triclinic_vectors(atomgroup.dimensions)
    shiftvects = np.matmul(boxvects.T, shiftlist.T).T

    # Calculate the squared bond lengths
    p1 = positions[bonds[:,0],:]
    p2 = positions[bonds[:,1],:]
    v = p1 - p2
    d2 = np.sum(v ** 2, axis=1)
    
    # find bonds that cross a cell boundary
    islonger = d2>minbl2
    if not np.any(islonger):
        newpos = atomgroup.positions
        #print('Atoms in correct cell, no work required!')
    else:
        #print(f'Total broken bonds = {np.sum(islonger)}')
        # For a bond (a,b), determine what shift is needed to get the closest image
        # of atom b from atom a, where (a,b) is known to cross a cell boundary
        d2min = d2[islonger]
        shifts = np.zeros((np.sum(islonger), 3))
        for sv in shiftvects:
            d2s = np.sum((v[islonger] - sv) ** 2, axis=1)
            isshorter = d2s<d2min
            shifts[isshorter] = sv
            d2min[isshorter] = d2s[isshorter]
        # Create an array to record cell adressess of closest image
        atom_shifts = np.zeros((n_atoms, 3))
        for i, bond in enumerate(bonds[islonger]):
            G.remove_bond(bond)
            connected = G.bfs(bond[1])
            G.add_bond(bond)
            atom_shifts[list(connected)] += shifts[i]

        newpos = positions + atom_shifts - atom_shifts[reference_atom]
        
        if inplace:
            atomgroup.positions = newpos
    
    return newpos

class bond_dict:
    G = defaultdict(list)
    def __init__(self, bonds=None):
        if bonds is not None:
            self.add_bonds(bonds)
            
    def add_bond(self, bond):
        a, b = bond
        self.G[a].append(b)
        self.G[b].append(a)
    
    def add_bonds(self, bonds):
        for bond in bonds:
            self.add_bond(bond)

    def remove_bond(self, bond):
        a, b = bond
        self.G[a].remove(b)
        self.G[b].remove(a)
        #self.G[a] = [z for z in self.G[a] if z != b]
        #self.G[b] = [z for z in self.G[b] if z != a]
        
    def remove_bonds(self, bonds):
        for bond in bonds:
            self.remove_bond(bond)
    
    def bonded_components(self):
        seen = set()
        components = []
        for v in self.G:
            if v not in seen:
                c = self.bfs(v)
                seen.update(c)
                components.append(c)
        return components
    
    def bfs(self,atom_indx):
        """A fast BFS node generator"""
        seen = set()
        nextlevel = {atom_indx}
        while nextlevel:
            thislevel = nextlevel
            nextlevel = set()
            for v in thislevel:
                if v not in seen:
                    seen.add(v)
                    nextlevel.update(self.G[v])
        return seen




    