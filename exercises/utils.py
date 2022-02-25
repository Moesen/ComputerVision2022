import numpy as np
import matplotlib as plt
from scipy.interpolate import RegularGridInterpolator
from PIL import Image


def projectpoints(K: np.array, R: np.array, t: np.array, Q: np.array) -> np.array:
    """Projects points given a camera-, rotation and translation matrix
    on points Q

    Args:
        K (np.array): 3x3 matrix
        R (np.array): 3x3 matrix
        t (np.array): 3 vector
        Q (np.array): 3xN matrix of points (orientation is important)

    Returns:
        np.array: Projected homogenious points in 2d
    """
    t = t.reshape(len(t), 1)
    Rt = np.append(R, t, axis=1)
    P = K @ Rt

    Qh = np.append(Q, np.ones((1, Q.shape[1])), axis=0)

    return P @ Qh


def hom_to_inhom(Q: np.array) -> np.array:
    """Transformation from homogenous to inhomogenous points

    Args:
        Q (np.array): 3xN matrix of points (orientation is importnat)

    Returns:
        np.array: inhomogenous points 2xN matrix
    """
    return Q[:-1] / Q[-1]


def inhom_to_hom(Q: np.array, s: int = 1) -> np.array:
    return np.vstack(Q, np.ones(Q.shape[1]) * s)


def projectpointsdist(
    K: np.array, R: np.array, t: np.array, dist: np.array, Q: np.array
):
    t = t.reshape(len(t), 1)
    Rt = np.append(R, t, axis=1)
    homo_Q = np.append(Q, np.ones((1, Q.shape[1])), axis=0)

    RtQ = Rt @ homo_Q
    RtQ_inhomo = RtQ[:-1] / RtQ[-1]

    distances = np.linalg.norm(RtQ_inhomo, axis=0) ** 2
    coffs_dists = distances * dist.reshape(dist.size, 1)
    sums = np.sum(coffs_dists, axis=0)

    dist_p = RtQ_inhomo * (1 + sums)

    RtQ_homo = np.append(dist_p, np.ones((1, RtQ_inhomo.shape[1])), axis=0)

    return K @ RtQ_homo


def distort_image(
    pixel_values: np.array, K: np.array, dist_coffs: np.array
) -> np.array:
    """Transforms an image given distortion coefficients and camera matrix

    Args:
        pixel_values (np.array): Image
        K (np.array): Camera matrix
        dist_coffs (np.array): Coefficients for distortortion [-1;1]

    Returns:
        np.array: Distorted image using grid interpolation
    """
    img_coords = np.array(np.meshgrid(range(1920), range(1080), [1])).T.reshape(-1, 3).T
    inv_K = np.linalg.inv(K)

    homo_inv_camera_coords = inv_K @ img_coords
    inhomo_inv_camera_coords = homo_inv_camera_coords[:-1] / homo_inv_camera_coords[-1]

    # Takes the power in increasing order for each coefficient of the distances
    distances = np.linalg.norm(inhomo_inv_camera_coords, axis=0).reshape(-1, 1)
    # Power to the distances
    powers = np.arange(2, len(dist_coffs) * 2 + 1, step=2).reshape(-1, 1)
    # Distances to the increasing power according to formular
    power_dists = distances**powers.T
    # Coefficients times the distances to the powers
    coff_dists = dist_coffs * power_dists
    # Sums of the distances
    sums = np.sum(coff_dists, axis=1)
    # New distorted projects
    dist_p = inhomo_inv_camera_coords * (1 + sums)
    # New homogenous coordinates
    homo = np.append(dist_p, np.ones((1, inhomo_inv_camera_coords.shape[1])), axis=0)
    # Projecting with camera matrix
    projected = K @ homo
    # Converting to inhomogenous coordinates
    inhomo_projected = projected[:-1] / projected[-1]
    # Creating pixel coordinates
    x, y = np.arange(0, 1920), np.arange(0, 1080)
    # Creating interpolator
    interpolator = RegularGridInterpolator((x, y), pixel_values)
    # Interpolating pixels
    pixels = (
        interpolator(inhomo_projected.T).reshape((1920, 1080, 3)).transpose((1, 0, 2))
    )
    # Normalizing pixels
    normalized = pixels / np.amax(pixels, axis=tuple(range(2)))
    return normalized


def cross_op(Q: np.array) -> np.array:
    """Creates a matrix from a 3d inhomogenous point
    with effect of its matrix multiplied product being the same
    as taking the cross product between two vectors

    Args:
        Q (np.array): 3 vector

    Returns:
        np.array: 3x3 matrix
    """
    return np.array([[0, -Q[2], Q[1]], [Q[2], 0, -Q[0]], [-Q[1], Q[0], 0]], dtype=float)


def generate_fundemental_matrix(k1: np.array, k2: np.array, r: np.array, t: np.array):
    t_op = cross_op(t)
    essential_matrix = t_op @ r
    fundemental_matrix = np.linalg.inv(k2.T) @ essential_matrix @ np.linalg.inv(k1)
    return essential_matrix, fundemental_matrix

/