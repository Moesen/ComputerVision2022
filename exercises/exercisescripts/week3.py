from __future__ import annotations
import utils as u
from exam_utils import ImgSaver
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

S = ImgSaver()
data_path = Path("../../data/TwoImageData.npy")

k = np.array([[1000, 0, 300], [0, 1000, 200], [0, 0, 1]])

r1 = np.identity(3)
t1 = np.array([0, 0, 0]).reshape((-1, 1))

r2 = Rotation.from_euler("xyz", [0.7, -0.5, 0.8]).as_matrix()
t2 = np.array([0.2, 2, 1]).reshape((-1, 1))


def test_ex1():
    q1, q2 = ex1()
    s = f"{q1=}\n {q2=}"
    S.save_txt("ex3-1", s)


def ex1():
    Q = np.array([1, 0.5, 4]).reshape(-1, 1)
    q1, q2 = u.projectpoints(k, r1, t1, Q), u.projectpoints(k, r2, t2, Q)

    return q1, q2


def test_ex8():
    img1, img2, F = ex8()
    _, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 5))
    ax0.imshow(img1, cmap="gray")
    ax0.grid(False)
    ax0.axis("off")
    ax1.imshow(img2, cmap="gray")
    ax1.grid(False)
    ax1.axis("off")
    plt.tight_layout()
    S.save_fig("ex3-8")
    s = str(F)
    S.save_txt("ex3-8", s)


def ex8():
    obj = np.load(data_path, allow_pickle=True).item()
    img1, img2, r1, r2, t1, t2, k = obj.values()
    E, F = u.generate_fundemental_matrix(k, k, r2, t2)
    return img1, img2, F

def draw_line(l, shape, ax):
    # Checks where the line intersects the four sides of the image
    # and finds the two intersections that are within the frame
    def in_frame(l_im):
        q = np.cross(l.flatten(), l_im)
        q = q[:2] / q[2]
        if all(q >= 0) and all(q + 1 <= shape[1::-1]):
            return q

    lines = [[1, 0, 0], [0, 1, 0], [1, 0, 1 - shape[1]], [0, 1, 1 - shape[0]]]
    P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]
    ax.plot(*np.array(P).T)

# Actually just uses the code from ex8
def demo_ex9():
    img1, img2, F = ex8()

    _, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 5))
    ax1.grid(False)
    ax2.grid(False)
    ax1.imshow(img1, cmap="gray")
    ax2.imshow(img2, cmap="gray")
    
    while True:
        plt.waitforbuttonpress()
        p1 = np.append(np.array(plt.ginput()),1)
        print(p1)
        l = F @ p1
        draw_line(l, img2.shape, ax2)
    plt.close()

def test_ex9():
    img1, img2, F = ex8()

    _, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 5))
    ax1.grid(False)
    ax2.grid(False)
    ax1.imshow(img1, cmap="gray")
    ax2.imshow(img2, cmap="gray")

    points = np.array([[100 , 500, 700 ],[300, 100, 500]])
    ax1.scatter(*points)
    for point in points.T:
        p = np.append(point, 1)
        l = F @ p
        draw_line(l, img2.shape, ax2)

    S.save_fig("ex3-9")



if __name__ == "__main__":
    # test_ex1()
    # test_ex8()
    # demo_ex9()
    test_ex9()
