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
# NOTE: This file authored by Mat Tolladay 26/11/2021 as part of de-cython

#import cython
import numpy as np
from .mdamath import triclinic_vectors
#cimport numpy as np
#cimport MDAnalysis.lib._cutil
#from MDAnalysis.lib._cutil cimport _dot ,_norm, _cross

#from libcpp.vector cimport vector


__all__ = ['augment_coordinates', 'undo_augment']


def augment_coordinates(coordinates, box, r):
    r"""Calculates the periodic images of particles which are within a distance
    `r` from the box walls.

    The algorithm works by generating explicit periodic images of atoms residing
    close to any of the six box walls. The steps involved in generating images
    involves the evaluation of reciprocal box vectors followed by the
    calculation of distances of atoms from the walls by means of projection onto
    the reciprocal vectors. If the distance is less than a specified cutoff
    distance, relevant periodic images are generated using box translation
    vectors :math:`\vec{t}` with

    .. math:: \vec{t}=l\cdot\vec{a}+m\cdot\vec{b}+n\cdot \vec{c}\,,

    where :math:`l,\,m,\,n \in \{-1,\,0,\,1\}` are the neighboring cell indices
    in :math:`x`-, :math:`y`-, and :math:`z`-direction relative to the central
    cell with box vectors :math:`\vec{a},\,\vec{b},\,\vec{c}`.

    For instance, an atom close to the :math:`xy`-plane containing the origin
    will generate a periodic image outside the central cell and close to the
    opposite :math:`xy`-plane of the box, i.e., shifted by
    :math:`\vec{t} = 0\cdot\vec{a}+0\cdot\vec{b}+1\cdot\vec{c}=\vec{c}`.

    Likewise, if the particle is close to more than one box walls, images along
    the diagonals are also generated::

                                    x            x
        +------------+                +------------+
        |            |   augment      |            |
        |            |   ------->     |            |
        |          o |              x |          o |
        +------------+                +------------+

    Parameters
    ----------
    coordinates : numpy.ndarray
      Input coordinate array of shape ``(n, 3)`` and dtype ``numpy.float32``
      used to generate duplicate images in the vicinity of the central cell. All
      coordinates must be within the primary unit cell.
    box : numpy.ndarray
      Box dimensions of shape ``(6,)`` and dtype ``numpy.float32``. The
      dimensions must be provided in the same format as returned
      by :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
      ``[lx, ly, lz, alpha, beta, gamma]``
    r : float
      Thickness of cutoff region for duplicate image generation.
      THIS IS ASSUMED TO BE LESS THAN 0.5*SHORTEST TRICLINIC VECTOR LENGTH

    Returns
    -------
    output : numpy.ndarray
      Coordinates of duplicate (augmented) particles (dtype ``numpy.float32``).
    indices : numpy.ndarray
      Original indices of the augmented coordinates (dtype ``numpy.int64``).
      Maps the indices of augmented particles to their original particle index
      such that ``indices[augmented_index] = original_index``.

    Note
    ----
    Output does not return coordinates from the initial array.
    To merge the particles with their respective images, the following operation
    is necessary when generating the images:

    .. code-block:: python

        images, mapping = augment_coordinates(coordinates, box, max_cutoff)
        all_coords = numpy.concatenate([coordinates, images])


    See Also
    --------
    :meth:`undo_augment`


    .. versionadded:: Testversion
    """


    # REQUIRED NUMPY CODE
    dm = triclinic_vectors(box)
    reciprocal = np.cross(dm[[1,2,0]], dm[[2,0,1]], axis=1)
    reciprocal /= np.linalg.norm(reciprocal, axis=1)[:,np.newaxis]
    other = np.sum(dm, axis=0) - coordinates

    lo = np.einsum('ij,ikj->ik', coordinates, reciprocal[np.newaxis])
    hi = np.einsum('ij,ikj->ik', other, reciprocal[np.newaxis])
    shifts = (lo<=r).astype(int) - (hi<=r).astype(int) # array of 1, 0 and -1
    # atom images can be at a face ([1 0 0]) and edge ([1 1 0]) and a corner 
    # ([1 1 1]) (or with negative 1). That is: there can be up to six images 
    # to include for each atom
    def make_images(shift_type):
        is_shifted = np.all(shifts[:,shift_type]!=0, axis=1)
        x = coordinates[is_shifted,:]
        for s in shift_type:
            x += shifts[is_shifted,np.newaxis,s] * dm[np.newaxis,s,:]
        return x, np.where(is_shifted)
    
    components = [[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]
    image_coordinates = []
    image_indices = []
    for c in components:
        image_coordinates, image_indices = make_images(c)
        
    return np.concatenate(image_coordinates, dtype=np.float32), \
        np.concatenate(image_indices,dtype=np.intp)
        
    
    
    # REPLACE BELOW HERE
    '''
    cdef bint lo_x, hi_x, lo_y, hi_y, lo_z, hi_z
    cdef int i, j, N
    cdef float norm
    cdef float shiftX[3] # The vectors (Z along x axis)
    cdef float shiftY[3] # Y in x-y plane
    cdef float shiftZ[3] # Z
    cdef float coord[3]
    cdef float end[3] # Seriously? end? Sum of furthest cartesian points from (0,0,0)
    cdef float other[3]
    cdef float dm[3][3]
    cdef float reciprocal[3][3]
    
    for i in range(3):
        shiftX[i] = dm[0][i]
        shiftY[i] = dm[1][i]
        shiftZ[i] = dm[2][i]
        end[i] = dm[0][i] + dm[1][i] + dm[2][i]
    # Calculate reciprocal vectors
    _cross(&dm[1][0], &dm[2][0], &reciprocal[0][0])
    _cross(&dm[2][0], &dm[0][0], &reciprocal[1][0])
    _cross(&dm[0][0], &dm[1][0], &reciprocal[2][0])
    # Normalize
    for i in range(3):
        norm = _norm(&reciprocal[i][0])
        for j in range(3):
            reciprocal[i][j] = reciprocal[i][j]/norm

    N = coordinates.shape[0]

# May be slow without initialisation giving number of elements
    cdef vector[float] output 
    cdef vector[int] indices

    for i in range(N): # Looping over atoms is always bad...
        for j in range(3):
            coord[j] = coordinates[i, j]
            other[j] = end[j] - coordinates[i, j]
        # identify the condition
        lo_x = _dot(&coord[0], &reciprocal[0][0]) <= r
        hi_x = _dot(&other[0], &reciprocal[0][0]) <= r
        lo_y = _dot(&coord[0], &reciprocal[1][0]) <= r
        hi_y = _dot(&other[0], &reciprocal[1][0]) <= r
        lo_z = _dot(&coord[0], &reciprocal[2][0]) <= r
        hi_z = _dot(&other[0], &reciprocal[2][0]) <= r

        if lo_x:
            # if X, face piece
            for j in range(3):
                # add to output
                # push_back will results in vectors resizing i.e. lots of 
                # copying to find sufficiently large contiguous memory!
                output.push_back(coord[j] + shiftX[j]) 
            # keep record of which index this augmented
            # position was created from
            indices.push_back(i)

            if lo_y:
                # if X&Y, edge piece
                for j in range(3):
                    output.push_back(coord[j] + shiftX[j] + shiftY[j])
                indices.push_back(i)

                if lo_z:
                    # if X&Y&Z, corner piece
                    for j in range(3):
                        output.push_back(coord[j] + shiftX[j] + shiftY[j] + shiftZ[j])
                    indices.push_back(i)

                elif hi_z:
                    for j in range(3):
                        output.push_back(coord[j] + shiftX[j] + shiftY[j] - shiftZ[j])
                    indices.push_back(i)

            elif hi_y:
                for j in range(3):
                    output.push_back(coord[j] + shiftX[j] - shiftY[j])
                indices.push_back(i)

                if lo_z:
                    for j in range(3):
                        output.push_back(coord[j] + shiftX[j] - shiftY[j] + shiftZ[j])
                    indices.push_back(i)

                elif hi_z:
                    for j in range(3):
                        output.push_back(coord[j] + shiftX[j] - shiftY[j] - shiftZ[j])
                    indices.push_back(i)

            if lo_z:
                for j in range(3):
                    output.push_back(coord[j] + shiftX[j] + shiftZ[j])
                indices.push_back(i)

            elif hi_z:
                for j in range(3):
                    output.push_back(coord[j] + shiftX[j] - shiftZ[j])
                indices.push_back(i)

        elif hi_x:
            for j in range(3):
                output.push_back(coord[j] - shiftX[j])
            indices.push_back(i)

            if lo_y:
                for j in range(3):
                    output.push_back(coord[j] - shiftX[j] + shiftY[j])
                indices.push_back(i)

                if lo_z:
                    for j in range(3):
                        output.push_back(coord[j] - shiftX[j] + shiftY[j] + shiftZ[j])
                    indices.push_back(i)

                elif hi_z:
                    for j in range(3):
                        output.push_back(coord[j] - shiftX[j] + shiftY[j] - shiftZ[j])
                    indices.push_back(i)

            elif hi_y:
                for j in range(3):
                    output.push_back(coord[j] - shiftX[j] - shiftY[j])
                indices.push_back(i)

                if lo_z:
                    for j in range(3):
                        output.push_back(coord[j] - shiftX[j] - shiftY[j] + shiftZ[j])
                    indices.push_back(i)

                elif hi_z:
                    for j in range(3):
                        output.push_back(coord[j] - shiftX[j] - shiftY[j] - shiftZ[j])
                    indices.push_back(i)

            if lo_z:
                for j in range(3):
                    output.push_back(coord[j] - shiftX[j] + shiftZ[j])
                indices.push_back(i)

            elif hi_z:
                for j in range(3):
                    output.push_back(coord[j] - shiftX[j] - shiftZ[j])
                indices.push_back(i)

        if lo_y:
            for j in range(3):
                output.push_back(coord[j] + shiftY[j])
            indices.push_back(i)

            if lo_z:
                for j in range(3):
                    output.push_back(coord[j] + shiftY[j] + shiftZ[j])
                indices.push_back(i)

            elif hi_z:
                for j in range(3):
                    output.push_back(coord[j] + shiftY[j] - shiftZ[j])
                indices.push_back(i)

        elif hi_y:
            for j in range(3):
                output.push_back(coord[j] - shiftY[j])
            indices.push_back(i)

            if lo_z:
                for j in range(3):
                    output.push_back(coord[j] - shiftY[j] + shiftZ[j])
                indices.push_back(i)

            elif hi_z:
                for j in range(3):
                    output.push_back(coord[j] - shiftY[j] - shiftZ[j])
                indices.push_back(i)

        if lo_z:
            for j in range(3):
                output.push_back(coord[j] + shiftZ[j])
            indices.push_back(i)

        elif hi_z:
            for j in range(3):
                output.push_back(coord[j] - shiftZ[j])
            indices.push_back(i)
    n = indices.size()
    return np.asarray(output, dtype=np.float32).reshape(n, 3), np.asarray(indices, dtype=np.intp)
'''
def undo_augment(results, translation, n_real):
    """Translate augmented indices back to original indices.

    Parameters
    ----------
    results : numpy.ndarray
      Array of dtype ``numpy.int64`` containing coordinate indices, including
      "augmented" indices.
    translation : numpy.ndarray
      Index map of dtype ``numpy.int64`` linking the augmented indices to the
      original particle indices such that
      ``translation[augmented_index] = original_index``.
    nreal : int
      Number of real coordinates, i.e., indices in `results` equal or larger
      than this need to be mapped to their real counterpart.

    Returns
    -------
    results : numpy.ndarray
      Modified input `results` with all the augmented indices translated to
      their corresponding initial original indices.

    Note
    ----
    Modifies the results array in place.

    See Also
    --------
    :meth:`augment_coordinates`


    .. versionadded:: TestVersion
    """
    is_augment = results >= n_real
    indx = results[is_augment]
    results[is_augment] = translation[indx - n_real]
    return results
'''    
@cython.boundscheck(False)
@cython.wraparound(False)
def undo_augment(np.intp_t[:] results, np.intp_t[:] translation, int nreal):
    
    cdef int N
    cdef ssize_t i
    N = results.shape[0]

    for i in range(N):
        if results[i] >= nreal:
            results[i] = translation[results[i] - nreal]
    return np.asarray(results, dtype=np.intp)
'''