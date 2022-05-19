from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import utils as u
from cv2 import cv2
from pathlib import Path
import matplotlib as mpl
from tqdm import tqdm
from exam_utils import ImgSaver
mpl.rc("image", cmap="gray")

S = ImgSaver()
data_path = Path("../../data/w11/")

K = np.loadtxt(data_path / "K.txt")


def test_ex1():
    ps, ims, m01, m12, [kp0, kp1, kp2] = ex1()
    fig = plt.figure()
    for i, (p, im) in enumerate(zip(ps, ims)):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(im)
        ax.scatter(*p.T, s=2, c="orange")
    ax = fig.add_subplot(2, 2, 3)
    im01 = cv2.drawMatchesKnn(
        ims[0],
        kp0,
        ims[1],
        kp1,
        m01,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    ax.imshow(im01)

    ax = fig.add_subplot(2, 2, 4)
    im12 = cv2.drawMatchesKnn(
        ims[1],
        kp1,
        ims[2],
        kp2,
        m12,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    ax.imshow(im12)
    plt.tight_layout()
    S.save_fig("ex11-1")

def ex1():
    img_names = [x.as_posix() for x in (data_path / "sequence").iterdir()]
    sequental = sorted(img_names)
    imgs = [u.load_image(path) for path in tqdm(sequental)]
    im0, im1, im2, *_ = imgs

    p0s, kp0, des0 = u.sift_detect_and_compute(im0, nfeatures=2000)
    p1s, kp1, des1 = u.sift_detect_and_compute(im1, nfeatures=2000)
    p2s, kp2, des2 = u.sift_detect_and_compute(im2, nfeatures=2000)

    arr01, m01 = u.match_sift_desc(des0, des1, knn_k=2)
    arr12, m12 = u.match_sift_desc(des1, des2, knn_k=2)

    return [p0s, p1s, p2s], [im0, im1, im2], m01, m12, [kp0, kp1, kp2]

def test_ex2():
    ex2()

def ex2():
    img_names = [x.as_posix() for x in (data_path / "sequence").iterdir()]
    sequental = sorted(img_names)[:2]
    im0, im1 = [u.load_image(path) for path in tqdm(sequental)]
    
    p0, k0, d0 = u.sift_detect_and_compute(im0, nfeatures=2000)
    p1, k1, d1 = u.sift_detect_and_compute(im1, nfeatures=2000)
    

    arr, m = u.match_sift_desc(d0, d1, knn_k=2)
    E, mask = cv2.findEssentialMat(p0[arr[:, 0], :], p1[arr[:, 1], :], K, method=cv2.RANSAC) 

    _, R, t, mask = cv2.recoverPose(E, p0[arr[:, 0], :], p1[arr[:, 1], :], K)

    m_filtered = arr[np.where(mask > 0)[0]]
    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1.imshow(im0)
    ax1.scatter(*p0[m_filtered[:, 0]].T, s=2, c="red")
    ax2.imshow(im1)
    ax2.scatter(*p1[m_filtered[:, 1]].T, s=2, c="red")
    S.save_fig("ex11-2")

def test_ex3():
    ims, points = ex3()
    _, axs = plt.subplots(1, 3)
    for ax, im, ps in zip(axs.flatten(), ims, points):
        ax.imshow(im)
        ax.scatter(*ps.T, s=2)
    S.save_fig("ex11-3")


def ex3():
    img_names = [x.as_posix() for x in (data_path / "sequence").iterdir()]
    sequental = sorted(img_names)[:3]
    im0, im1, im2 = [u.load_image(path) for path in tqdm(sequental)]
    
    p0, k0, d0 = u.sift_detect_and_compute(im0, nfeatures=2000)
    p1, k1, d1 = u.sift_detect_and_compute(im1, nfeatures=2000)
    p2, k2, d2 = u.sift_detect_and_compute(im2, nfeatures=2000)
    

    m01, _ = u.match_sift_desc(d0, d1, knn_k=2)
    m12, _ = u.match_sift_desc(d1, d2, knn_k=2)

    E01, mask01 = cv2.findEssentialMat(p0[m01[:, 0], :], p1[m01[:, 1], :], K, method=cv2.RANSAC) 
    E12, mask12 = cv2.findEssentialMat(p1[m12[:, 0], :], p2[m12[:, 1], :], K, method=cv2.RANSAC) 

    _, R01, t01, mask01 = cv2.recoverPose(E01, p0[m01[:, 0], :], p1[m01[:, 1], :], K)
    _, R12, t12, mask12 = cv2.recoverPose(E12, p1[m12[:, 0], :], p2[m12[:, 1], :], K)

    m01_filtered = m01[np.where(mask01 > 0)[0]]
    m12_filtered = m12[np.where(mask12 > 0)[0]]
    
    _, idx01, idx12 = np.intersect1d(m01_filtered[:, 1], m12_filtered[:, 0], return_indices=True)

    common_match01 = m01_filtered[idx01]
    common_match12 = m12_filtered[idx12]

    idx0 = common_match01[:, 0]
    idx1 = common_match01[:, 1]
    idx2 = common_match12[:, 1]

    points0 = p0[idx0]
    points1 = p1[idx1]
    points2 = p2[idx2]
     
    return [im0, im1, im2], [points0, points1, points2]

def test_ex4():
    ex4()

def ex4():
    img_names = [x.as_posix() for x in (data_path / "sequence").iterdir()]
    sequental = sorted(img_names)[:3]
    im0, im1, im2 = [u.load_image(path) for path in tqdm(sequental)]
    
    p0, k0, d0 = u.sift_detect_and_compute(im0, nfeatures=2000)
    p1, k1, d1 = u.sift_detect_and_compute(im1, nfeatures=2000)
    p2, k2, d2 = u.sift_detect_and_compute(im2, nfeatures=2000)
    

    m01, _ = u.match_sift_desc(d0, d1, knn_k=2)
    m12, _ = u.match_sift_desc(d1, d2, knn_k=2)

    E01, mask01 = cv2.findEssentialMat(p0[m01[:, 0], :], p1[m01[:, 1], :], K, method=cv2.RANSAC) 
    E12, mask12 = cv2.findEssentialMat(p1[m12[:, 0], :], p2[m12[:, 1], :], K, method=cv2.RANSAC) 

    _, R01, t01, mask01 = cv2.recoverPose(E01, p0[m01[:, 0], :], p1[m01[:, 1], :], K)
    _, R12, t12, mask12 = cv2.recoverPose(E12, p1[m12[:, 0], :], p2[m12[:, 1], :], K)

    m01_filtered = m01[np.where(mask01 > 0)[0]]
    m12_filtered = m12[np.where(mask12 > 0)[0]]
    
    _, idx01, idx12 = np.intersect1d(m01_filtered[:, 1], m12_filtered[:, 0], return_indices=True)

    common_match01 = m01_filtered[idx01]
    common_match12 = m12_filtered[idx12]

    idx0 = common_match01[:, 0]
    idx1 = common_match01[:, 1]
    idx2 = common_match12[:, 1]

    points0 = p0[idx0]
    points1 = p1[idx1]
    points2 = p2[idx2]

    P0 = K@np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K@np.hstack((R01, t01))

    Q = cv2.triangulatePoints(P0, P1, points0.T, points1.T)
    Q = u.hom_to_inhom(Q)
    _, R, t, inliers = cv2.solvePnPRansac(Q.T, points2, K, None)
    R, _ = cv2.Rodrigues(R)
    t = -R.T.dot(t)
    
    tx = ty = tz = 100
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*Q, s=5)
    ax.grid("off")
    ax.axis(False)
    ax.view_init(90, 90)
    S.save_fig("ex11-4")
     
    return 
    


if __name__ == "__main__":
    # test_ex1()
    # test_ex2()
    # test_ex3()
    test_ex4()
