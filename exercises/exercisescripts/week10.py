from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import utils as u
from cv2 import cv2
from pathlib import Path
import matplotlib as mpl
from exam_utils import ImgSaver

mpl.rc("image", cmap="gray")

S = ImgSaver()
im_path = Path("../../data/panorama/")
im1_path = im_path / "im1.jpg"
im2_path = im_path / "im2.jpg"


def test_ex1():
    im = cv2.drawMatchesKnn(
        *ex1(), None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.imshow(im)
    S.save_fig("ex10-1")


def ex1():
    im2 = u.load_image(im1_path)
    im1 = u.load_image(im2_path)

    m, p1, p2, kp1, kp2 = u.get_features_sift(im1, im2, knn_k=2)

    return im1, kp1, im2, kp2, m


def test_ex2():
    im = cv2.drawMatchesKnn(
        *ex2(), None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.imshow(im)
    S.save_fig("ex10-2")


def ex2():
    im2 = u.load_image(im1_path)
    im1 = u.load_image(im2_path)

    m, p1, p2, kp1, kp2 = u.get_features_sift(im1, im2, knn_k=2)
    p1h = u.inhom_to_hom(p1)
    p2h = u.inhom_to_hom(p2)
    Hest, inlier_idx = u.RANSAC_Hest(p1h, p2h)
    return im1, kp1, im2, kp2, np.array(m)[inlier_idx]


def test_ex4():
    img1_warp, img1_mask, img2_warp, img2_mask = ex4()
    _, [ax1, ax2] = plt.subplots(1, 2)
    ax1.imshow(img1_warp) 
    ax2.imshow(img2_warp) 
    plt.tight_layout()
    S.save_fig("ex10-4")    


def ex4():
    im1 = u.load_image(im2_path)
    im2 = u.load_image(im1_path)

    m, p1, p2, kp1, kp2 = u.get_features_sift(im1, im2, knn_k=2)
    p1h = u.inhom_to_hom(p1)
    p2h = u.inhom_to_hom(p2)
    Hest, inlier_idx = u.RANSAC_Hest(p1h, p2h, itterations=10000)
    xrange = [0, 1400]
    yrange = [0, im1.shape[0]]

    img1_warp, img1_mask = u.warp_image(im1, np.identity(3), xrange, yrange)
    img2_warp, img2_mask = u.warp_image(im2, Hest, xrange, yrange)

    return img1_warp, img1_mask, img2_warp, img2_mask

def test_ex5():
    im = ex5()
    plt.imshow(im)
    S.save_fig("ex10-5")

def ex5():
    im1 = u.load_image(im2_path)
    im2 = u.load_image(im1_path)

    m, p1, p2, kp1, kp2 = u.get_features_sift(im1, im2, knn_k=2)
    p1h = u.inhom_to_hom(p1)
    p2h = u.inhom_to_hom(p2)
    Hest, inlier_idx = u.RANSAC_Hest(p1h, p2h, itterations=10000)
    xrange = [-1, 1.2]
    yrange = [-1, 2]

    imwarp = u.stich_images(im2, im1, Hest, xrange, yrange)
    return imwarp


if __name__ == "__main__":
    # test_ex1()
    # test_ex2()
    # test_ex4()
    test_ex5()
