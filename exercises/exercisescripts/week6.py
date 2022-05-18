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

S = ImgSaver()

mpl.rc("image", cmap="gray")
img_path = Path("../../data/week06_data")


def test_ex1():
    g, gx, x = ex1()
    plt.plot(x, gx)
    plt.plot(x, g)
    plt.legend(["gx", "g"])
    S.save_fig("ex6-1")


def ex1():
    return u.gaussian1DKernel(10)

def test_ex2():
    I, Ix, Iy = ex2()
    _, [ax2, ax3, ax4] = plt.subplots(1, 3, figsize=(20, 5))
    ax2.imshow(I)
    ax2.set_title("Filtered image")
    ax3.imshow(Ix)
    ax3.set_title("X direction derived image")
    ax4.imshow(Iy)
    ax4.set_title("Y direction derived image")
    plt.tight_layout()
    S.save_fig("ex6-2")
    plt.close()

def ex2():
    img = cv2.imread((img_path / "TestIm1.png").as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return u.gaussianSmoothing(img, 10)
    
def test_ex3(): 
    C = ex3()
    _, axs = plt.subplots(2, 2)
    C = C.reshape(4, 300, 300)
    for ax, C in zip(axs.flatten(), C):
        ax.imshow(C)
    S.save_fig("ex6-3")

def ex3():
    img = cv2.imread((img_path / "TestIm1.png").as_posix(), cv2.IMREAD_GRAYSCALE) 
    h_img = u.smoothed_hessian(img, 10, 600)
    return h_img

def test_ex4():
    r, img = ex4()
    _, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 10))
    ax1.imshow(img)
    ax2.imshow(r)
    x, y = np.where(r < 0)
    ax1.scatter(y, x, s = 2, c="blue")
    x, y = np.where(r > r.max() * 0.977)
    ax1.scatter(y, x, s=10, c="red")
    S.save_fig("ex6-4")

def ex4():
    img = cv2.imread((img_path / "TestIm1.png").as_posix(), cv2.IMREAD_GRAYSCALE) 
    sigma = 0.7
    r = u.harris_measure(img, sigma, 2*np.ceil(sigma*3), 0.06)
    return r, img

def test_ex5():
    ex5()

def ex5():
    img = cv2.imread((img_path / "TestIm1.png").as_posix(), cv2.IMREAD_GRAYSCALE) 
    sigma = 0.7
    epsilon = 2 * np.ceil(sigma * 3)
    k = .06
    tau = .977
    return 

if __name__ == "__main__":
    # test_ex1()
    # test_ex2()
    # test_ex3()
    test_ex4()
