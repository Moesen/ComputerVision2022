from __future__ import annotations
import numpy as np
import matplotlib as plt
from scipy.interpolate import RegularGridInterpolator
from scipy import optimize
import matplotlib.pyplot as plt
from cv2 import cv2
from scipy import ndimage
from tqdm import tqdm


def create_cameramatrix(f: int, alpha: int, beta: int, dx: int, dy: int):
    return np.array([[f, beta * f, dx], [0, alpha * f, dy], [0, 0, 1]])


def create_R_matrix(theta):
    return np.array(
        [
            [np.cos(np.deg2rad(theta)), 0, np.sin(np.deg2rad(theta))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(theta)), 0, np.cos(np.deg2rad(theta))],
        ]
    )


def box3d(n: int) -> np.ndarray:
    l = np.linspace(-0.5, 0.5, n)
    ones = np.zeros(n)

    box = np.array([l, ones, ones])
    box = np.append(box, [l, ones - 0.5, ones - 0.5], axis=1)
    box = np.append(box, [l, ones - 0.5, ones + 0.5], axis=1)
    box = np.append(box, [l, ones + 0.5, ones + 0.5], axis=1)
    box = np.append(box, [l, ones + 0.5, ones - 0.5], axis=1)

    box = np.append(box, [ones, l, ones], axis=1)
    box = np.append(box, [ones - 0.5, l, ones - 0.5], axis=1)
    box = np.append(box, [ones - 0.5, l, ones + 0.5], axis=1)
    box = np.append(box, [ones + 0.5, l, ones + 0.5], axis=1)
    box = np.append(box, [ones + 0.5, l, ones - 0.5], axis=1)

    box = np.append(box, [ones, ones, l], axis=1)
    box = np.append(box, [ones - 0.5, ones - 0.5, l], axis=1)
    box = np.append(box, [ones - 0.5, ones + 0.5, l], axis=1)
    box = np.append(box, [ones + 0.5, ones + 0.5, l], axis=1)
    box = np.append(box, [ones + 0.5, ones - 0.5, l], axis=1)

    return box


# Converts color image to gray
def c2g(im, normalize=True):
    g_im = np.mean(im, axis=2)
    if normalize:
        g_im /= 255
    return g_im


def make_projection_matrix(K, R, t):
    return K @ np.append(R, t.reshape(-1, 1), axis=1)


def smoothed_hessian(im, sigma, epsilon):
    _, Ix, Iy = gaussianSmoothing(im, sigma=sigma)
    gx, *_ = gaussian1DKernel(sigma, epsilon)

    C = np.array(
        [
            [
                cv2.filter2D(cv2.filter2D(Ix**2, -1, gx), -1, gx.T),
                cv2.filter2D(cv2.filter2D(Ix * Iy, -1, gx), -1, gx.T),
            ],
            [
                cv2.filter2D(cv2.filter2D(Ix * Iy, -1, gx), -1, gx.T),
                cv2.filter2D(cv2.filter2D(Iy**2, -1, gx), -1, gx.T),
            ],
        ]
    )
    return C


def harris_measure(im, sigma, epsilon, k):
    C = smoothed_hessian(im, sigma, epsilon)
    a, b, c = C[0, 0, :, :], C[1, 1, :, :], C[0, 1, :, :]
    r = a * b - c**2 - k * (a + b) ** 2
    return r


def projectpoints(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, Q: np.ndarray, return_hom=False
) -> np.ndarray:
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


def hom_to_inhom(Q: np.ndarray) -> np.ndarray:
    """Transformation from homogenous to inhomogenous points

    Args:
        Q (np.array): 3xN matrix of points (orientation is importnat)

    Returns:
        np.array: inhomogenous points 2xN matrix
    """
    return Q[:-1] / Q[-1]


def inhom_to_hom(Q: np.ndarray, s: int = 1) -> np.ndarray:
    if len(Q.shape) == 1:
        return np.append(Q, 1 * s)
    else:
        return np.vstack([Q, np.ones(Q.shape[1]) * s])


def projectpointsdist(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, dist: np.ndarray, Q: np.ndarray
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


def distort_image(img: np.ndarray, K: np.ndarray, dist_coffs: np.ndarray) -> np.ndarray:
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
    interpolator = RegularGridInterpolator((x, y), img)
    # Interpolating pixels
    pixels = (
        interpolator(inhomo_projected.T).reshape((1920, 1080, 3)).transpose((1, 0, 2))
    )
    # Normalizing pixels
    normalized = pixels / np.amax(pixels, axis=tuple(range(2)))
    return normalized


def checkerboard_points(n: int, m: int) -> np.ndarray:
    first_row = np.arange(n) - ((n - 1) / 2)
    second_row = np.arange(m) - ((m - 1) / 2)
    combinations = np.array(np.meshgrid(second_row, first_row)).reshape(2, -1)
    combinations = np.vstack([combinations, np.zeros(n * m)])

    return combinations


def cross_op(Q: np.ndarray) -> np.ndarray:
    """Creates a matrix from a 3d inhomogenous point
    with effect of its matrix multiplied product being the same
    as taking the cross product between two vectors

    Args:
        Q (np.array): 3 vector

    Returns:
        np.array: 3x3 matrix
    """
    return np.array([[0, -Q[2], Q[1]], [Q[2], 0, -Q[0]], [-Q[1], Q[0], 0]], dtype=float)


def generate_fundemental_matrix(
    k1: np.ndarray, k2: np.ndarray, r: np.ndarray, t: np.ndarray
):
    t_op = cross_op(t)
    essential_matrix = t_op @ r
    fundemental_matrix = np.linalg.inv(k2.T) @ essential_matrix @ np.linalg.inv(k1)
    return essential_matrix, fundemental_matrix


def triangulate(qs: np.ndarray, ps: np.ndarray, return_hom=False) -> np.ndarray:
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

    B = (left_side * qs) - right_side  # type: ignore
    A = B.T @ B
    _, _, vh = np.linalg.svd(A)
    ret = vh[-1, :]
    if not return_hom:
        ret = hom_to_inhom(ret)
    return ret


def triangulate_nonlin(qs: np.ndarray, ps: np.ndarray) -> np.ndarray:

    """Nonlinear triangulation using least squares method

    Args:
        qs (np.array): N*2X1 vector
        ps (np.array): Projection matrices in 4x3*N shape

    Returns:
        np.array: Closest approximation for point
    """

    assert ps.shape[0] % 3 == 0

    # Function for calculating residuals
    def compute_residuals(Q: np.ndarray) -> np.ndarray:
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
        residuals = inhom_projection - qs.reshape(-1)  # type: ignore
        # print(f"{residuals=}")
        return residuals
        # return inner

    x0 = triangulate(qs, ps, return_hom=False)
    # compute_residuals(x0)
    optim = optimize.least_squares(compute_residuals, x0)
    return optim["x"]


def test_points(n_in, n_out, return_hom=False, random_seed: int | None = None):
    if random_seed:
        np.random.seed(random_seed)

    a = (np.random.rand(n_in) - 0.5) * 10
    b = np.vstack((a, a * 0.5 + np.random.randn(n_in) * 0.25))
    points = np.hstack((b, 2 * np.random.randn(2, n_out)))
    points = np.random.permutation(points.T).T

    if return_hom:
        points = inhom_to_hom(points)

    return points


def estimate_line(a: np.ndarray, b: np.ndarray):
    if np.product(a.shape) == 2:
        a = inhom_to_hom(a)
    if np.product(b.shape) == 2:
        b = inhom_to_hom(b)
    c = np.cross(a, b)
    c[:-1] = c[:-1] / np.linalg.norm(c[:-1])
    return c


def in_or_out(point: np.ndarray, line: np.ndarray, threshold: float = 0.1):
    # point (3x1), line (3x1)
    sign = point @ line
    print(sign)
    return abs(sign) <= threshold


def calc_consenus_line(
    points: np.ndarray,
    line: np.ndarray,
    threshold: float = 0.1,
    return_points: bool = False,
):
    if points.shape[1] > 3:
        points = points.T
    ret_count = np.sum(np.abs(points @ line) <= threshold)
    ret_points = points[np.abs(points @ line) <= threshold]
    if return_points:
        return ret_count, ret_points
    else:
        return ret_count


def calc_sampsons_distance(F, p1, p2):
    a1 = (p2.T @ F)[0] ** 2
    b1 = (p2.T @ F)[1] ** 2
    a2 = (F @ p1)[0] ** 2
    b2 = (F @ p1)[1] ** 2

    return (p2.T @ F @ p1) ** 2 * 1 / (a1+ b1 + a2 + b2)


def calc_consensus_fest(Fest: np.ndarray, q1: np.ndarray, q2: np.ndarray, sigma: float):
    threshold = 3.84 * sigma**2
    inliers = []
    for i in range(q1.shape[1]):
        sdist = calc_sampsons_distance(Fest, q1[:, i], q2[:, i])
        if sdist < threshold:
            inliers.append(i)
    return len(inliers), np.asarray(inliers)


def draw_random(
    points: np.ndarray, n: int = 2, random_seed: int | None = None
) -> np.ndarray:
    if points.shape[1] > points.shape[0]:
        points = points.T
    if random_seed:
        np.random.seed(random_seed)

    idx = np.random.choice(points.shape[0], n)
    while len(np.unique(idx)) == 1:
        idx = np.random.choice(points.shape[0], n)
    return points[idx].T


def get_features_sift(im1: np.ndarray, im2: np.ndarray):
    sift = cv2.SIFT_create(nOctaveLayers = 5,
                           contrastThreshold = 0.04,
                           edgeThreshold = 10000,
                           sigma = 1.6)

    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    bf = cv2.BFMatcher_create(crossCheck=True)
    matches = bf.match(des1, des2)

    p1 = np.array([kp1[m.queryIdx].pt for m in matches]).T
    p2 = np.array([kp2[m.trainIdx].pt for m in matches]).T


    return matches, p1, p2, list(kp1), list(kp2)


def RANSAC_8_point(p1s, p2s, sigma=3, itterations=100):
    _max = 0
    _max_F = None
    _best_idx = None

    for i in tqdm(range(itterations)):
        random_idxs = np.random.choice(np.arange(p1s.shape[1]), 8, replace=False)
        q1 = p1s[:, random_idxs]
        q2 = p2s[:, random_idxs]

        Fest = Fest_8point(q1, q2, normalize=True)

        inlier_count, inlier_idx = calc_consensus_fest(Fest, p1s, p2s, sigma)
        if inlier_count > _max:
            _max_F = Fest
            _best_idx = inlier_idx
            _max = inlier_count
            print("New best estimate: ", _max)
    
    return _max_F, _best_idx


def RANSAC(
    points: np.ndarray,
    threshold: float = 0.1,
    random_seed=None,
    return_points: bool = False,
    p: float = 0.99,
):
    _max = 0
    _max_line: None | np.ndarray = None
    _points: None | np.ndarray = None

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
        count, ret_points = calc_consenus_line(
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


def plot_homo_line(line: np.ndarray, points: np.ndarray, threshold: int, ax=None):
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


def gaussian1DKernel(sigma: float, epsilon: int | None = None):
    h = epsilon or np.ceil(5 * sigma)
    x = np.arange(-h, h + 1)

    g = np.exp(-(x**2) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
    g /= g.sum()
    g = g.reshape(-1, 1)
    gx = -(-(x**2)) / (sigma**2) * g[:, 0]
    gx = gx.reshape(-1, 1)
    return g, gx, x


def gaussianSmoothing(img: np.ndarray, sigma: float):
    # Returns the gaussian smoothed image I, and the image derivatives Ix and Iy.
    # 1 obtain the kernels for gaussian and for differentiation.
    g, gx, _ = gaussian1DKernel(sigma=sigma)
    # 2 Filter the image in both directions and diff in both directions
    I = cv2.filter2D(cv2.filter2D(img, -1, g), -1, g.T)  # smooth I = g * g.T * I
    # 3 Differentiate - d/dx I = g * gx.T * I
    Ix = cv2.filter2D(cv2.filter2D(img, -1, gx.T), -1, g)
    Iy = cv2.filter2D(cv2.filter2D(img, -1, g.T), -1, gx)
    return I, Ix, Iy


def corner_detection(
    im: np.ndarray, sigma: float, eps: int, k: float, tau: float
) -> np.ndarray:
    r = harris_measure(im, sigma, eps, k)
    btau = np.where(r < tau * r.max())
    r[btau] = 0
    i_idx = np.array([1, -1, 0, 0])
    j_idx = np.array([0, 0, 1, -1])

    do = True
    while do:
        do = False
        # Transposing for * trick in for loop
        queue = np.array(np.where(r > 0)).T
        for idx in range(len(queue)):
            i, j = queue[idx]
            # If border pixel set it to zero and return
            if np.any([0 > i + i_idx, i + i_idx > r.shape[0]]) or np.any(
                [0 > j + j_idx, j + j_idx > r.shape[1]]
            ):
                r[i, j] = 0
                continue

            if (
                np.all(r[i, j] > r[i + i_idx, j + j_idx])
                and np.sum(r[i + i_idx, j + j_idx]) > 0
            ):
                do = True
                r[i + i_idx, j + j_idx] = 0

    corners = np.array(np.where(r > 0))
    return corners


def get_scalespace(im: np.ndarray, sigma: float, n: int) -> np.ndarray:
    space = []
    for i in range(n):
        width = sigma * 2**i
        space_im, *_ = gaussianSmoothing(im, width)
        space.append(space_im)
    return np.array(space)


# def get_DoG(im: np.ndarray, sigma: float, n: int) -> np.array:
def get_DoG(*args) -> np.ndarray:
    im_scales = get_scalespace(*args)
    dogs = []
    for ims1, ims2 in zip(im_scales, im_scales[1:]):
        dogs.append(ims1 - ims2)
    return np.array(dogs)


def detect_blobs(im: np.ndarray, sigma: float, n: int, threshold=0.1):
    dogs = get_DoG(im, sigma, n)

    # non-maximum suppression
    img2 = ndimage.maximum_filter(dogs, size=1)
    # img_thresh = img2.max() * threshold
    img_thresh = threshold
    labels, num_labels = ndimage.label(img2 > img_thresh)
    coords = ndimage.measurements.center_of_mass(
        im, labels=labels, index=np.arange(1, num_labels + 1)
    )

    return np.floor(coords).astype(int)


def transform_im(im: np.ndarray, theta: float, scale: float) -> np.ndarray:
    r_img = cv2.resize(im, None, fx=scale, fy=scale)  # type: ignore
    r_img = ndimage.rotate(r_img, theta)
    return r_img


def homogrophy_mapping(p: np.ndarray, H: np.ndarray, set_scale_1=True):
    q = H @ p
    if set_scale_1:
        q = q / q[2]
    return q


def estimate_homopgraphies(Q_omega: np.ndarray, qs: list) -> list:
    # Takes 3d points as input in list form 3xn,
    # and a list of Q projected to the image plain from different views
    assert Q_omega.shape[0] == 3
    Qo_tilde = inhom_to_hom(Q_omega[:2])
    homographies = []
    for q in qs:
        qh = inhom_to_hom(q)
        h = hest(qh, Qo_tilde)
        homographies.append(h)

    return homographies


def make_v_row(H: np.ndarray, i, j):
    return np.array(
        [
            H[0][i] * H[0][j],
            H[0][i] * H[1][j] + H[1][i] * H[0][j],
            H[1][i] * H[1][j],
            H[2][i] * H[0][j] + H[0][i] * H[2][j],
            H[2][i] * H[1][j] + H[1][i] * H[2][j],
            H[2][i] * H[2][j],
        ]
    )


def make_v(Hs: list[np.ndarray]) -> np.ndarray:
    return np.vstack(
        [
            np.vstack([make_v_row(h, 0, 1), make_v_row(h, 0, 0) - make_v_row(h, 1, 1)])  # type: ignore
            for h in Hs
        ]
    )


def estimate_b(Hs: list[np.ndarray]) -> np.ndarray:
    V = make_v(Hs)  # type: ignore
    *_, vh = np.linalg.svd(V)
    b = vh[-1]
    return b


def estimate_intrinsics(Hs: list[np.ndarray]):
    # Estimating b
    b = estimate_b(Hs)
    [b11, b12, b22, b13, b23, b33] = b

    # Intrinsic values from appendix b
    v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12**2)
    l = b33 - (b13**2 + v0 * (b12 * b13 - b11 * b23)) / b11
    alpha = np.sqrt(l / b11)
    beta = np.sqrt(l * b11 / (b11 * b22 - b12**2))
    gamma = b12 * alpha**2 * beta / l
    u0 = (gamma * v0 / beta) - (b13 * alpha**2 / l)

    return np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])


def estimate_extrinsics(K: np.ndarray, Hs: list) -> tuple:
    rots = []
    tran = []

    for H in Hs:
        l = 1 / np.linalg.norm(np.linalg.inv(K) @ H[:, 0], 2)

        r1 = l * np.linalg.inv(K) @ H[:, 0]
        r2 = l * np.linalg.inv(K) @ H[:, 1]
        r3 = np.cross(r1, r2)

        t = (l * np.linalg.inv(K) @ H[:, 2]).T

        rots.append(np.vstack([r1, r2, r3]).T)
        tran.append(t.T)

    return rots, tran


def calibratecamera(qs, Q):
    Hs = estimate_homopgraphies(Q, qs)
    K = estimate_intrinsics(Hs)
    Rs, ts = estimate_extrinsics(K, Hs)
    return K, Rs, ts


def format_H_to_solution(H):
    return H * (-2 / H[0][0])


def normalize2d(Q: np.ndarray):
    mean, std = np.mean(Q, axis=1), np.std(Q, axis=1)
    T = np.array(
        [
            [1 / std[0], 0, -mean[0] / std[0]],
            [0, 1 / std[1], -mean[1] / std[1]],
            [0, 0, 1],
        ]
    )
    return [T @ Q, T]


def makeB(q1: np.ndarray, q2: np.ndarray):
    return np.vstack(
        [np.kron(q1[:, i], cross_op(q2[:, i])) for i in range(q2.shape[1])]
    )


def Fest_8point(q1, q2, normalize=True):
    if normalize:
        q1, t1 = normalize2d(q1)
        q2, t2 = normalize2d(q2)

    B = makeB(q1, q2)

    *_, vh = np.linalg.svd(B.T @ B)

    fest = vh[-1].reshape(3, 3).T

    if normalize:
        fest = t2.T @ fest @ t1  # type: ignore

    return fest


def hest(q1: np.ndarray, q2: np.ndarray, normalize=False) -> np.ndarray:
    t1, t2 = None, None

    # Normalizing
    if normalize:
        q1, t1 = normalize2d(q1)
        q2, t2 = normalize2d(q2)

    B = makeB(q2, q1)
    # Creating the A matrix
    A = B.T @ B
    # Doing svd
    _, _, vh = np.linalg.svd(A)
    # Taking the last row of the vh, as those are the eigenvectors with smallest eigenvalues
    H = np.reshape(vh[-1], (3, 3)).T
    # Normalizing

    if t1 is not None and t2 is not None:
        H = np.linalg.inv(t1) @ H @ t2

    return H


def DLT(Qh: np.ndarray, qh: np.ndarray, normalize=False) -> np.ndarray:
    t1 = None

    if normalize:
        qh, t1 = normalize2d(qh)

    B = makeB(Qh, qh)
    *_, vh = np.linalg.svd(B.T @ B)
    Pest = vh[-1].reshape((4, 3)).T

    if t1 is not None:
        Pest = np.linalg.inv(t1) @ Pest

    Pest = Pest / Pest[-1, -1] * 10
    return Pest


########### VIZ STUFF ###########
def show_images(
    imgs: np.ndarray,
    titles=None,
    cmap="gray",
    size=5,
    show_axis="off",
):
    if titles:
        assert imgs.shape[0] == len(titles)

    ratio = imgs.shape[1] / imgs.shape[2]
    h, w = size, size * ratio
    rows = int(np.ceil(imgs.shape[0] / 2))
    cols = 2

    _, axs = plt.subplots(rows, cols, figsize=(h * cols, w * rows))
    for i in range(imgs.shape[0]):
        ax = axs.flatten()[i]
        ax.imshow(imgs[i, ...], cmap=cmap)
        if titles:
            ax.set_title(titles[i], fontsize=size * 1.5)
        ax.axis(show_axis)
    plt.tight_layout()
    plt.show()
    return
