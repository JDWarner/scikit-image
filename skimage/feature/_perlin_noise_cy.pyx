#cython boundscheck=False
#cython wraparound=False
#cython cdivision=True

import numpy as np
cimport numpy as cnp

from libc.math cimport sqrt


cdef inline void normalize2(float[2] v):
    cdef float s = sqrt(v[0] * v[0] + v[1] * v[1])
    v[0] = v[0] / s
    v[1] = v[1] / s


cdef inline void normalize3(float[3] v):
    cdef float s = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    v[0] = v[0] / s
    v[1] = v[1] / s
    v[2] = v[2] / s


cdef inline double s_curve(t):
    # Calculate s-curve (weighting for means)
    return t * t * (3. - 2. * t)


cdef inline double linear_interp(t, a, b):
    return t * (b - a) + a


def perlin_generator_2d(double[:, ::1] image_coords,
                        double[:, :, ::1] norms,
                        double[:, ::1] perlin_coords,
                        tuple shape):
    """
    Generates Perlin noise from unit normal data on R**2.

    Parameters
    ----------
    iamge_coords : (N, 2) ndarray
        World coordinates for every point in `shape`.
    perlin_coords : (P, 2) ndarray
        World coordinates for points on Perlin grid.
    norms : ndarray
        (M, N, 2) array, representing a random grid of unit vectors on R**2.
    shape : tuple
        Length-2 tuple of ints, containing the desired shape of the image.
        All `image_coords` must fit in an array of this size!

    Returns
    -------
    noise : ndarray
        (M, N) array of Perlin noise on range [-1, 1]

    """
    cdef:
        double[:, ::1] noise = np.zeros(shape, dtype=np.float64)
        Py_ssize_t i, j, x, y
        list range(norms.shape[1] - 2)

    for i in range(norms.shape[0] - 2):
        for j in range(norms.shape[1] - 2):
            x = norms[i, j, 0]
            y = norms[i, j, 1]
