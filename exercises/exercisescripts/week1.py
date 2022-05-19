from __future__ import annotations
import utils as u
from exam_utils import ImgSaver
import matplotlib.pyplot as plt
import numpy as np

S = ImgSaver()


def test_ex11():
    box = ex11()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*box)
    S.save_fig("ex1-11")


def ex11() -> np.ndarray:
    n = 16
    return u.box3d(n)


def test_ex12():
    projected = ex12()
    plt.figure(figsize=(10, 10))
    plt.scatter(*projected, s=20)
    plt.grid(True)
    plt.xlim((-0.5, 0.5))
    plt.ylim((-0.5, 0.5))
    S.save_fig("ex1-12")


def ex12() -> np.ndarray:
    box = u.box3d(16)
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = np.array([[0, 0, 4]]).T
    R = K.copy()
    projected = u.projectpoints(K, R, t, box)
    return projected


def test_ex13():
    projected = ex13()
    plt.figure(figsize=(10, 10))
    plt.scatter(*projected[:2], s=20)
    plt.grid(True)
    plt.xlim((-0.5, 0.5))
    plt.ylim((-0.5, 0.5))
    S.save_fig("ex1-13")


def ex13(theta: int = -30, t: np.ndarray = np.array([0, 0, 4])) -> np.ndarray:
    box = u.box3d(16)
    R = u.create_R_matrix(theta)
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    projected = u.projectpoints(K, R, t, box, return_hom=True)
    plt.xlim((-0.5, 0.5))
    plt.ylim((-0.5, 0.5))
    return projected

def test_ex14():
    print("""
    # 1.14 what does R and t do?
    # R:
    # - Scales the object while transforming to camera plane
    # - Rotate, stretch or shrink the object
    # t: moves the object in the camera plane.
    """)

if __name__ == "__main__":
    test_ex11()
    test_ex12()
    test_ex13()
    test_ex14()
