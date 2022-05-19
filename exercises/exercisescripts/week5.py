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


def ex6(samples=None) -> tuple[list, list]:
    img_fns = [x.as_posix() for x in img_path.iterdir() if ".jpg" in x.as_posix()]
    img_fns = img_fns[:samples]
    imgs = load_imgs(img_fns)
    imgs, img_points = find_checkerpoints(imgs)
    return imgs, img_points


def test_ex7():
    ex7()


def ex7():
    images, image_points = ex6()
    checker_points = u.checkerboard_points(10, 7)
    # Making found image_points 2d
    x, _, y = image_points[0].shape
    ps2d = [q.reshape(x, y).T for q in image_points]

    K, R, t = u.calibratecamera(ps2d, checker_points)
    return K, R, t, checker_points, ps2d, images


def test_ex8():
    images, qs, detected_points = ex8()
    fig, axs = plt.subplots(3, 5, figsize=(25,10))
    for ax, img, q, dp  in zip(axs.flatten(), images, qs, detected_points):
        ax.imshow(img)
        ax.scatter(*q, s=10, c="red")
        err = np.sqrt(np.mean((q - dp)**2))
        ax.set_title(f"{err=:.2f}")
        ax.grid("off")
        ax.axis("off")
    plt.tight_layout()
    S.save_fig("ex5-8")


def ex8():
    K, Rs, ts, checker_points, detected_points, images = ex7()
    Ps = [u.make_projection_matrix(K, R, t) for R, t in zip(Rs, ts)]
    qs = [u.hom_to_inhom( P @ u.inhom_to_hom(checker_points)) for P in Ps]
    return images, qs, detected_points


if __name__ == "__main__":
    # images, image_points = ex6()
    # test_ex6(images, image_points)
    # test_ex7()
    test_ex8()
