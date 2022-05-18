from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import utils as u
from cv2 import cv2
from pathlib import Path
import os
import matplotlib as mpl
from tqdm import tqdm
from exam_utils import ImgSaver
from week3 import ex8


S = ImgSaver()

mpl.rc("image", cmap="gray")
point_data_path = Path("../../data/qs.npy")
img_path = Path("../../data/TwoImageData.npy")


def test_ex1():
    Ftrue, fest, fnest = ex1()
    print(f"{np.sum(Ftrue - fest)=}")
    print(f"{np.sum(Ftrue - fnest)=}")


def ex1():
    *_, Ftrue = ex8()
    q1, q2 = np.load(point_data_path, allow_pickle=True).item().values()

    fest = u.Fest_8point(q1, q2, normalize=False)
    fest = fest / fest[0, 0] * Ftrue[0, 0]

    fnest = u.Fest_8point(q1, q2, normalize=False)
    fnest = fnest / fnest[0, 0] * Ftrue[0, 0]
    return Ftrue, fest, fnest


def test_ex2():
    ex2()


def ex2():
    im1, im2, r1, r2, t1, t2, k = np.load(img_path, allow_pickle=True).item().values()
    sift = cv2.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    bf = cv2.BFMatcher_create(crossCheck=True)
    matches = bf.match(des1, des2)

    s_matches = sorted(matches, key=lambda x: x.distance)
    match_img = cv2.drawMatches(im1, kp1, im2, kp2, s_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return

def test_ex3():
    Fest, Ftrue = ex3()
    Fest = Fest/Fest[0,0]*Ftrue[0, 0]

    print(Fest)
    print(Ftrue)
    print(np.linalg.norm(Ftrue - Fest))
    

def ex3():
    im1, im2, r1, r2, t1, t2, k = np.load(img_path, allow_pickle=True).item().values()
    s_matches, p1, p2, kp1, kp2 = u.get_features_sift(im1, im2)
    
    p1h = u.inhom_to_hom(p1)
    p2h = u.inhom_to_hom(p2)

    Fest, _ = u.RANSAC_8_point(p1h, p2h, itterations=100)
    Rtilde = r2@r1.T
    ttilde = t2 - Rtilde@t1
    _, Ftrue = u.generate_fundemental_matrix(k, k, Rtilde, ttilde)
    return Fest, Ftrue


if __name__ == "__main__":
    # test_ex1()
    # test_ex2()
    test_ex3()
