from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import utils as u
from cv2 import cv2
from pathlib import Path
import os
import matplotlib as mpl
from tqdm import tqdm


mpl.rc("image", cmap="gray")
img_path = Path("../../data/calibration")


def load_imgs(img_path: str | list) -> list:
    if type(img_path) == str:
        img_path = [img_path]
    imgs = map(cv2.imread, tqdm(img_path, postfix="loading imgs"))
    imgs = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), imgs)
    imgs = map(lambda x: cv2.resize(x, (600, 400)), imgs)  # type: ignore
    return list(imgs)


def find_checkerpoints(imgs: list) -> tuple[list, list]:
    points = []
    tru_imgs = []
    for img in imgs:
        ret, corners = cv2.findChessboardCorners(img, (7, 10))
        if ret == True:
            points.append(corners)
            tru_imgs.append(img)
    return tru_imgs, points


def test_ex6(imgs: list, image_points: list) -> None:
    _, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        ax = axs.flatten()[i]
        painted = cv2.drawChessboardCorners(imgs[i], (7, 10), image_points[i], True)
        ax.imshow(painted)
    plt.show()


def ex6() -> tuple[list, list]:
    img_fns = [x.as_posix() for x in img_path.iterdir() if ".jpg" in x.as_posix()]
    img_fns = img_fns
    imgs = load_imgs(img_fns)
    imgs, img_points = find_checkerpoints(imgs)
    return imgs, img_points

def test_ex7():
    pass

def ex7(img: list, image_points: list):
    checker_points = u.checkerboard_points(7, 10)
    objpoints = []

if __name__ == "__main__":
    images, image_points = ex6()
#    test_ex6(images, image_points)
    ex7(images, image_points)
