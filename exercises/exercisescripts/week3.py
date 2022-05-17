from __future__ import annotations
import utils as u
from exam_utils import ImgSaver
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.spatial.transform import Rotation

S = ImgSaver()
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
    Q = np.array([1, .5, 4]).reshape(-1, 1)
    q1, q2 = u.projectpoints(k, r1, t1, Q), u.projectpoints(k, r2, t2, Q)

    return q1, q2


if __name__ == "__main__":
    test_ex1()
