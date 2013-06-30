import numpy as np
from . import perlin_generator_2d, perlin_generator_3d

__all__ = ['perlin_noise']


def _get_gridshape(shape, grid_res):
    # This is she shape of the Perlin grid at resolution grid_res, expanded
    # sufficiently to allow the desired shape (with 1 pixel symmetric
    # padding) to fit inside.
    gridshape = []
    for i in len(shape):
        if (shape[i] + 2) % grid_res[i] == 0:
            gridshape.append((shape[i] + 2) / grid_res[i] + 1)
        else:
            gridshape.append(np.ceil(shape / grid_res))
    return gridshape


def _random_normal_grads(shape, grid_res, gridshape):
    # Randomized Unit normal vectors in a cartesian coordinate system
    if len(shape) == 2:
        theta = np.random.uniform(0., 2. * np.pi, gridshape)
        x = np.cos(theta)
        y = np.sin(theta)
        return np.concatenate((x[..., np.newaxis],
                               y[..., np.newaxis]), axis=-1)
    elif len(shape) == 3:
        raise NotImplementedError("3D will be awesome, but not yet.")
        theta = np.random.uniform(0., 2. * np.pi, gridshape)
        x = np.cos(theta)
        y = np.sin(theta)
        return np.concatenate((x[..., np.newaxis],
                               y[..., np.newaxis],
                               z[..., np.newaxis]), axis=-1)


def _image_coords(shape, grid_res, gridshape):
    # Actual coordinates for image grid
    n = len(shape)

    # Generate coordinates
    coords = np.mgrid[[slice(-1, s + 1)
                       for s in shape]].reshape(n, -1).T

    # Adjust coordinates for each axis
    for i in range(n):
        coords[:, i] += (gridshape[i] * grid_res[i] - shape[i]) / 2.
        coords[:, i] /= grid_res[i]

    return np.ascontiguousarray(coords.astype(np.float64))


def _perlin_coords(shape, grid_res, gridshape):
    # Actual coordinates for Perlin grid
    n = len(shape)

    # Generate coordinates
    coords = np.mgrid[[slice(0, s)
                       for s in gridshape]].reshape(n, -1).T

    # Adjust coordinates for each axis
    # for i in range(n):
    #     coords[:, i] *= grid_res[i]

    return np.ascontiguousarray(coords.astype(np.float64))


def perlin_noise(shape, grid_res=None):
    """
    Perlin noise generator, to produce natural appearing texture procedurally.

    Parameters
    ----------
    shape : tuple
        Shape of desired noise function.
    grid_res : iterable
        Resolution of the Perlin grid in pixels. Default is 10 for all dims.

    Returns
    -------
    noise : ndarray
        Output array containing Perlin noise of size `shape`.

    Note
    ----
    Perlin noise is defined on the range [-1, 1] and should be approximately
    equally distributed about zero. Thus, adding this noise to an image should
    not significantly change the overall mean.

    """
    if grid_res is None:
        grid_res = [10.] * len(shape)
    else:
        if len(grid_res) != len(shape):
            raise ValueError("grid_res and shape must have same length.")

    gridshape = _get_gridshape(shape, grid_res)

    # Generate Perlin grid and a slightly larger grid of coordinates, to have
    # lookup points on either side
    random_norms = _random_normal_grads(shape, grid_res, gridshape)
    image_coords = _image_coords(shape, grid_res, gridshape)
    perlin_coords = _perlin_coords(shape, grid_res, gridshape)

    if len(shape) == 2:
        noise = perlin_generator_2d(image_coords, random_norms, perlin_coords)
    elif len(shape) == 3:
        noise = perlin_generator_3d(image_coords, random_norms, perlin_coords)

    return noise
