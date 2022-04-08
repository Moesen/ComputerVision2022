import numpy as np
import matplotlib as plt
from scipy.interpolate import RegularGridInterpolator
from scipy import optimize
from PIL import Image
from typing import Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage


def make_projection_matrix(K, R, t):
    return K @ np.append(R, t.reshape(-1, 1), axis=1)


def projectpoints(
    K: np.array, R: np.array, t: np.array, Q: np.array, return_hom=False
) -> np.array:
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

    # Checking if Q is 1-D vector
    if len(Q.shape) == 1:
        Q = Q.reshape(-1, 1)
    Qh = np.append(Q, np.ones((1, Q.shape[1])), axis=0)

    return_Q = P @ Qh
    if not return_hom:
        return_Q = hom_to_inhom(return_Q)

    return return_Q


def hom_to_inhom(Q: np.array) -> np.array:
    """Transformation from homogenous to inhomogenous points

    Args:
        Q (np.array): 3xN matrix of points (orientation is importnat)

    Returns:
        np.array: inhomogenous points 2xN matrix
    """
    return Q[:-1] / Q[-1]


def inhom_to_hom(Q: np.array, s: int = 1) -> np.array:
    if len(Q.shape) == 1:
        return np.append(Q, 1 * s)
    else:
        return np.vstack([Q, np.ones(Q.shape[1]) * s])


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


def triangulate(qs: np.array, ps: np.array, return_hom=False) -> np.array:
    """Finds a point given pixel coordinates q, and their respective projection matrix p

    Args:
        qs (np.array): inhomogenous pixel coordinates in 2xN shape
        ps (np.array): projection matrices, in 3xN shape

    Returns:
        np.array: _description_
    """
    # We check whether the shape is alright
    assert len(qs.shape) == 1 or qs.shape[1] == 1
    assert ps.shape[0] % 3 == 0

    left_side = np.repeat(ps[2::3], 2, axis=0)
    right_side = np.delete(ps, list(range(2, ps.shape[0], 3)), axis=0)

    B = left_side * qs - right_side
    A = B.T @ B
    _, _, vh = np.linalg.svd(A)
    ret = vh[-1, :]
    if not return_hom:
        ret = hom_to_inhom(ret)
    return ret


def triangulate_nonlin(qs: np.array, ps: np.array) -> np.array:

    """Nonlinear triangulation using least squares method

    Args:
        qs (np.array): N*2X1 vector
        ps (np.array): Projection matrices in 4x3*N shape

    Returns:
        np.array: Closest approximation for point
    """

    assert ps.shape[0] % 3 == 0

    # Function for calculating residuals
    def compute_residuals(Q: np.array) -> np.array:
        """Takes an estimation of q as input, and outputs residuals
        This is done in the following steps:
            1 (projection):         First the projection is done on a homogenous Q.
            2.(inhom_projection):   the projection is made inhomogounous again
            3 (residuals):          Lastly the loss is calculated by subtracting
                                    the estimated points from qs

        Args:
            Q (np.array): 3x1 point
        """
        projection = ps @ inhom_to_hom(Q)
        # print(f"{projection=}")
        inhom_projection = hom_to_inhom(projection.reshape(-1, 3).T).T.reshape(-1)
        # print(f"{inhom_projection=}")
        residuals = inhom_projection - qs.reshape(-1)
        # print(f"{residuals=}")
        return residuals
        # return inner

    x0 = triangulate(qs, ps, return_hom=False)
    # compute_residuals(x0)
    optim = optimize.least_squares(compute_residuals, x0)
    return optim["x"]


def test_points(n_in, n_out, return_hom=False, random_seed: int = None):
    if random_seed:
        np.random.seed(random_seed)

    a = (np.random.rand(n_in) - 0.5) * 10
    b = np.vstack((a, a * 0.5 + np.random.randn(n_in) * 0.25))
    points = np.hstack((b, 2 * np.random.randn(2, n_out)))
    points = np.random.permutation(points.T).T

    if return_hom:
        points = inhom_to_hom(points)

    return points


def estimate_line(a: np.array, b: np.array):
    if np.product(a.shape) == 2:
        a = inhom_to_hom(a)
    if np.product(b.shape) == 2:
        b = inhom_to_hom(b)
    c = np.cross(a, b)
    c[:-1] = c[:-1] / np.linalg.norm(c[:-1])
    return c


def in_or_out(point: np.array, line: np.array, threshold: int = 0.1):
    # point (3x1), line (3x1)
    sign = point @ line
    print(sign)
    return abs(sign) <= threshold


def calc_consenus(
    points: np.array, line: np.array, threshold: int = 0.1, return_points: bool = False
):
    if points.shape[1] > 3:
        points = points.T
    ret_count = np.sum(np.abs(points @ line) <= threshold)
    ret_points = points[np.abs(points @ line) <= threshold]
    if return_points:
        return ret_count, ret_points
    else:
        return ret_count


def draw_random(
    points: np.array, n: int = 2, random_seed: int = None
) -> Tuple[np.array, np.array]:
    if points.shape[1] > points.shape[0]:
        points = points.T
    if random_seed:
        np.random.seed(random_seed)

    idx = np.random.choice(points.shape[0], n)
    while len(np.unique(idx)) == 1:
        idx = np.random.choice(points.shape[0], n)
    return points[idx].T


def RANSAC(
    points: np.array,
    threshold: int = 0.1,
    random_seed=None,
    return_points: bool = False,
    p: int = 0.99,
):
    _max = 0
    _max_line: np.array = None
    _points: np.array = None

    num_datapoints = int(points.shape[1])

    def est_n() -> int:
        _eps = 1 - (_max / num_datapoints)
        _n = np.log(1 - p) / np.log(1 - (1 - _eps) ** num_datapoints)
        return _n

    N = 2
    i = 0

    while i < N:
        a, b = draw_random(points, random_seed=random_seed).T
        line = estimate_line(a, b)
        count, ret_points = calc_consenus(
            points, line, threshold=threshold, return_points=True
        )
        if count > _max:
            _max = count
            _max_line = line
            _points = ret_points
        i += 1
        N = est_n()

    if return_points:
        return _max_line, _points
    else:
        return _max_line


def pca_line(x):  # assumes x is a (2 x n) array of points
    d = np.cov(x)[:, 0]
    d /= np.linalg.norm(d)
    l = [d[1], -d[0]]
    l.append(-(l @ x.mean(1)))
    return np.array(l)


def plot_homo_line(
    line: np.array, points: np.array, threshold: int, ax: plt.Axes = None
):
    assert len(line.shape) == 1
    a = line[0] / line[1]
    b = line[2] / line[1]

    inhom_points = hom_to_inhom(points)
    x = np.linspace(np.min(inhom_points[0]) - 1, np.max(inhom_points[0]) + 1)
    y = -a * x - b

    plot_frame = ax or plt
    plot_frame.plot(x, y, c="orange")
    plot_frame.fill_between(x, y - threshold, y + threshold, alpha=0.3, color="blue")
    plot_frame.plot(x, y - threshold, "--", c="black")
    plot_frame.plot(x, y + threshold, "--", c="black")
    plot_frame.scatter(*inhom_points, c="gray")


def gaussian1DKernel(sigma: float, epsilon: int = None):
    h = epsilon or np.ceil(4 * sigma)
    x = np.arange(-h, h + 1)

    g = np.exp(-(x**2) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
    g /= g.sum()
    gx = (-x / sigma**2) * g

    g = g.reshape(-1, 1)
    gx = gx.reshape(-1, 1)
    return g, gx


def gaussianSmoothing(img: np.array, sigma: float):
    g, gx = gaussian1DKernel(sigma)
    new_img = cv2.filter2D(img, -1, g)
    new_img = cv2.filter2D(new_img, -1, g.T)

    gdx = g @ gx.T
    gdy = gx @ g.T

    Ix = cv2.filter2D(img, -1, gdx)
    Iy = cv2.filter2D(img, -1, gdy)

    return new_img, Ix, Iy


def get_scalespace(im: np.array, sigma: float, n: int) -> np.array:
    space = []
    for i in range(n):
        width = sigma * 2**i
        space_im, *_ = gaussianSmoothing(im, width)
        space.append(space_im)
    return np.array(space)


# def get_DoG(im: np.array, sigma: float, n: int) -> np.array:
def get_DoG(*args) -> np.array:
    im_scales = get_scalespace(*args)
    dogs = []
    for ims1, ims2 in zip(im_scales, im_scales[1:]):
        dogs.append(ims1 - ims2)
    return np.array(dogs)

def detect_blobs(im: np.array, sigma: float, n: int, threshold=.1):
    dogs = get_DoG(im, sigma, n)
    
    # non-maximum suppression
    img2 = ndimage.maximum_filter(dogs, size=1)
    # img_thresh = img2.max() * threshold
    img_thresh = threshold
    labels, num_labels = ndimage.label(img2 > img_thresh)    
    coords = ndimage.measurements.center_of_mass(im, labels=labels, index=np.arange(1, num_labels + 1))

    return np.floor(coords).astype(int)

def transform_im(im: np.array, theta: float, scale: float) -> np.array:
    r_img = cv2.resize(im, None, fx=scale, fy=scale)
    r_img = ndimage.rotate(r_img, theta)
    return r_img
    

########### VIZ STUFF ###########
def show_images(imgs: np.array, titles=None, cmap="gray", size=5, show_axis="off"):
    if titles:
        assert imgs.shape[0] == len(titles)
        
    ratio = imgs.shape[1] / imgs.shape[2]
    h,w = size, size * ratio
    rows = int(np.ceil(imgs.shape[0] / 2))
    cols = 2

    fig, axs = plt.subplots(rows, cols, figsize=(h*cols, w*rows))
    for i in range(imgs.shape[0]):
        ax = axs.flatten()[i]
        ax.imshow(imgs[i,...])
        if titles:
            ax.set_title(titles[i], fontsize=size*1.5)
        ax.axis(show_axis)
    plt.tight_layout()
    plt.show()
    return