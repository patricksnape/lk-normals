import numpy as np


def cart2sph(x, y, z):
    xy = np.power(x, 2) + np.power(y, 2)
    # for elevation angle defined from XY-plane up
    return np.vstack([np.arctan2(y, x), np.arctan2(z, np.sqrt(xy))]).T


def sph2cart(azimuth, elevation, r):
    cart = np.empty([azimuth.shape[0], 3])
    azi_cos = np.cos(azimuth)
    ele_cos = np.cos(elevation)
    azi_sin = np.sin(azimuth)
    ele_sin = np.sin(elevation)
    cart[:, 0] = r * ele_cos * azi_cos
    cart[:, 1] = r * ele_cos * azi_sin
    cart[:, 2] = r * ele_sin
    return cart


def normalise_vector(v):
    return v / np.sqrt(np.sum(v ** 2, axis=-1))[..., None]


def row_norm(v):
    return np.sqrt(np.sum(v ** 2, axis=-1))


def logmap_cosine(vectors):
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]

    xyz = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    xy = np.sqrt(x ** 2 + y ** 2)

    gx = x / xy
    gy = y / xy
    gz = z / xyz
    sgz = np.sqrt(1 - gz ** 2)

    spher = np.vstack([gx, gy, gz, sgz]).T
    spher[np.isnan(spher)] = 0.0

    return spher


def expmap_cosine(angles):
    gx = angles[:, 0]
    gy = angles[:, 1]
    gz = angles[:, 2]
    sgz = angles[:, 3]

    gzsgz = np.sqrt(gz ** 2 + sgz ** 2)

    gxgy = np.sqrt(gx ** 2 + gy ** 2)

    gx = gx / gxgy
    gy = gy / gxgy
    gz = gz / gzsgz
    sgz = sgz / gzsgz

    phi = np.arctan2(gy, gx)
    theta = np.arctan(gz / sgz)

    cart = sph2cart(phi, theta, np.ones_like(phi))
    cart[np.isnan(cart)] = 0.0

    return cart

