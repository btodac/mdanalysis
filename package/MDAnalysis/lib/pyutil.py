# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
#

#import cython
import numpy as np
#from collections import defaultdict 
#cimport numpy as np
#from libc.math cimport sqrt, fabs

#from MDAnalysis import NoDataError
from MDAnalysis.lib.mdamath import triclinic_vectors
from molgraph import mol_graph
#from libcpp.set cimport set as cset
#from libcpp.map cimport map as cmap
#from libcpp.vector cimport vector
#from libcpp.utility cimport pair
#from cython.operator cimport dereference as deref


__all__ = ['make_whole', 'find_fragments']
           #'_sarrus_det_single', '_sarrus_det_multiple']
'''
cdef extern from "calc_distances.h":
    ctypedef float coordinate[3]
    void minimum_image(double* x, float* box, float* inverse_box)
    void minimum_image_triclinic(double* dx, float* box)

ctypedef cset[int] intset
ctypedef cmap[int, intset] intmap
'''
def _in2d(arr1,arr2):
    """Similar to np.in1d except works on 2d arrays 

    Parameters
    ----------
    arr1, arr2 : numpy.ndarray, shape (n,p) and (m, p)
       arrays of integers

    Returns
    -------
    in1d : bool array
      if an element of arr1 was in arr2

    .. versionadded:: 1.1.
    Perform columnwise sort on both arrays then test for the
    """
    
    n_cols = arr1.shape[1]
    arr1 = np.sort(arr1, axis=1)
    arr2 = np.sort(arr2, axis=1)
    
    baseval = np.max([arr1.max(), arr2.max()]) + 1
    arr1 = arr1.astype(object)
    arr2 = arr2.astype(object)
    arr1 = arr1 * baseval ** np.array(range(n_cols),dtype=object)
    arr2 = arr2 * baseval ** np.array(range(n_cols),dtype=object)
    return np.isin(np.sum(arr1, axis=1), np.sum(arr2, axis=1))
    
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
    G = mol_graph(bonds)
    
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

        newpos = positions + atom_shifts - atom_shifts[ref]
        
        if inplace:
            atomgroup.positions = newpos
    
    return newpos
'''
class bond_dict:
    G = defaultdict(list)
    atoms = []
    def __init__(self, atoms=None, bonds=None):
        if atoms is not None:
            self.atoms = atoms
        
        if bonds is not None:
            self.add_bonds(bonds)
            testatoms = np.unique(bonds)
            if len(self.atoms) is not 0:
                if not np.all(np.isin(testatoms, self.atoms)):
                    self.atoms.append(np.setdiff1d(testatoms, self.atoms))
            else:
                self.atoms = testatoms
    
    def keys(self):
        return self.G.keys()
    
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
    
    def add_atom(self, atom):
        if atom is not in self.G.keys():
            self.G[atom] = []
            
    def add_atoms(self, atoms):
        for atom in atoms:
            self.add_atom(atom)
            
    def remove_atom(self, atom):
        self.atoms = list(set(self.atoms) - set([atom]))
        del self.G[atom]
        for key in self.G.keys():
            self.G[key] = list(set(self.G[key]) - atom)
            
    def remove_atoms(self, atoms):
        atoms = set(atoms)
        self.atoms = list(set(self.atoms) - atoms)
        for a in atoms:
            del self.G[a]
        for key in self.G.keys():
            self.G[key] = list(set(self.G[key]) - atoms)
    
    def bonded_components(self):
        seen = set()
        components = []
        for v in self.atoms:
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
'''
'''
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float _dot(float * a, float * b):
    """Return dot product of two 3d vectors"""
    cdef ssize_t n
    cdef float sum1

    sum1 = 0.0
    for n in range(3):
        sum1 += a[n] * b[n]
    return sum1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _cross(float * a, float * b, float * result):
    """
    Calculates the cross product between 3d vectors

    Note
    ----
    Modifies the result array
    """

    result[0] = a[1]*b[2] - a[2]*b[1]
    result[1] = - a[0]*b[2] + a[2]*b[0]
    result[2] = a[0]*b[1] - a[1]*b[0]

cdef float _norm(float * a):
    """
    Calculates the magnitude of the vector
    """
    cdef float result
    cdef ssize_t n
    result = 0.0
    for n in range(3):
        result += a[n]*a[n]
    return sqrt(result)
'''
'''
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.float64_t _sarrus_det_single(np.float64_t[:, ::1] m):
    """Computes the determinant of a 3x3 matrix."""
    cdef np.float64_t det
    det = m[0, 0] * m[1, 1] * m[2, 2]
    det -= m[0, 0] * m[1, 2] * m[2, 1]
    det += m[0, 1] * m[1, 2] * m[2, 0]
    det -= m[0, 1] * m[1, 0] * m[2, 2]
    det += m[0, 2] * m[1, 0] * m[2, 1]
    det -= m[0, 2] * m[1, 1] * m[2, 0]
    return det

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray _sarrus_det_multiple(np.float64_t[:, :, ::1] m):
    """Computes all determinants of an array of 3x3 matrices."""
    cdef np.intp_t n
    cdef np.intp_t i
    cdef np.float64_t[:] det
    n = m.shape[0]
    det = np.empty(n, dtype=np.float64)
    for i in range(n):
        det[i] = m[i, 0, 0] * m[i, 1, 1] * m[i, 2, 2]
        det[i] -= m[i, 0, 0] * m[i, 1, 2] * m[i, 2, 1]
        det[i] += m[i, 0, 1] * m[i, 1, 2] * m[i, 2, 0]
        det[i] -= m[i, 0, 1] * m[i, 1, 0] * m[i, 2, 2]
        det[i] += m[i, 0, 2] * m[i, 1, 0] * m[i, 2, 1]
        det[i] -= m[i, 0, 2] * m[i, 1, 1] * m[i, 2, 0]
    return np.array(det)
'''
    
def find_fragments(atoms, bondlist):
    """Calculate distinct fragments from nodes (atom indices) and edges (pairs
    of atom indices).

    Parameters
    ----------
    atoms : array_like
       1-D Array of atom indices (dtype will be converted to ``numpy.int64``
       internally)
    bonds : array_like
       2-D array of bonds (dtype will be converted to ``numpy.int32``
       internally), where ``bonds[i, 0]`` and ``bonds[i, 1]`` are the
       indices of atoms connected by the ``i``-th bond. Any bonds referring to
       atom indices not in `atoms` will be ignored.

    Returns
    -------
    fragments : list
       List of arrays, each containing the atom indices of a fragment.

    .. versionaddded:: 0.19.0
    """
        
    G = mol_graph(atoms, bondlist)
    return G.bonded_components()
    