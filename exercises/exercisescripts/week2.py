from __future__ import annotations
import utils as u
from exam_utils import ImgSaver
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

S = ImgSaver()
img_path = Path("../../data/gopro_robot.jpg")


def test_ex1():
    projected_point = ex1()
    s = f"P1 Projects to: {projected_point.T}"
    S.save_txt("ex2-1", s)


def ex1():
    f = 600
    alpha = 1
    beta = 0
    dx = 400
    dy = 400

    K = u.create_cameramatrix(f, alpha, beta, dx, dy)

    theta = 0
    R = u.create_R_matrix(theta)
    t = np.array([0, 0.2, 1.5])

    P1 = np.array([[-0.5, -0.5, -0.5]]).T
    projected_point = u.projectpoints(K, R, t, P1)
    return projected_point


def test_ex2():
    projected_points = ex2()
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(*projected_points[:2])
    plt.xlim(0, 800)
    plt.ylim(0, 800)
    S.save_fig("ex2-2")


def ex2():
    f = 600
    alpha = 1
    beta = 0
    dx = 400
    dy = 400

    K = u.create_cameramatrix(f, alpha, beta, dx, dy)

    theta = 0
    # Rotation
    R = u.create_R_matrix(theta)
    # Translation
    t = np.array([0, 0.2, 1.5])
    # Points in a 3dBox
    box = u.box3d(16)
    # Distortion
    dist = np.array([-0.2])

    projected_points = u.projectpointsdist(K, R, t, dist, box)

    return projected_points


def test_ex4():
    undistorted_img, img = ex4()
    _, [ax1, ax2] = plt.subplots(2, 1)
    ax1.imshow(img.transpose(1, 0, 2))
    ax2.imshow(undistorted_img)
    S.save_fig("ex2-4")


def ex4():
    img = np.asarray(Image.open(img_path))
    img = img.transpose((1, 0, 2))
    h, w, d = img.shape

    f = int(0.455732 * h)
    alpha = 1
    beta = 0
    dx = int(0.5 * h)
    dy = int(0.5 * w)

    K = u.create_cameramatrix(f, alpha, beta, dx, dy)

    k3 = -0.245031
    k5 = 0.071524
    k7 = -0.00994978

    dist = np.array([k3, k5, k7])
    undistorted_img = u.distort_image(img, K, dist)

    return undistorted_img, img


def test_ex5():
    points = ex5()
    S.save_txt("ex2-5", str(points))


def ex5():
    p = np.array([[1, 0, 2, 2], [1, 3, 3, 4]])
    ph = u.inhom_to_hom(p)
    H = np.array([[-2, 0, 1], [1, -2, 0], [0, 0, 3]])
    qh = u.homogrophy_mapping(ph, H)
    qi = u.hom_to_inhom(qh)
    return qi


def test_ex6():
    H, Hest = ex6()
    s = f"{H}\n{np.linalg.norm(H)=}"
    S.save_txt("ex2-6", s)
    print(H)
    print(Hest)
    print(u.format_H_to_solution(Hest).round(5))


def ex6():
    p = np.array([[1, 0, 2, 2], [1, 3, 3, 4]])
    ph = u.inhom_to_hom(p)
    H = np.array([[-2, 0, 1], [1, -2, 0], [0, 0, 3]])
    qh = u.homogrophy_mapping(ph, H)
    Hest = u.hest(qh, ph)
    return H, Hest


def test_ex7():
    points, t = ex7()
    assert (np.isclose(np.std(points, axis=1), [1, 1])).all()
    assert (np.isclose(np.mean(points, axis=1), [0, 0])).all()

    ps = f"{ points = }\n"
    stds = f"Point std:\n{np.round( np.std(points, axis=1), 5 )}\n"
    means = f"Point Mean:\n{np.round( np.mean(points, axis=1),5 )}\n"
    t = f"{t = }"
    s = ps + stds + means + t
    S.save_txt("ex2-7", s)
    print("ex7")
    print(s)


def ex7():
    p = np.random.randint(0, 10, size=(2, 10))
    ph = u.inhom_to_hom(p)
    pn, t = u.normalize2d(ph)
    ph = u.hom_to_inhom(pn)
    return ph, t


def test_ex8():
    H, Hest, Hnest = ex8()
    Hest = u.format_H_to_solution(Hest).round(5)
    Hnest = u.format_H_to_solution(Hnest).round(5)
    s = f"H:\n{H=}\n Hest:\n{Hest=}\n Hnest:\n{Hnest=}\n"
    S.save_txt("ex2-8", s)
    print("ex8")
    print(s)


def ex8():
    p = np.array([[1, 0, 2, 2], [1, 3, 3, 4]])
    ph = u.inhom_to_hom(p)
    H = np.array([[-2, 0, 1], [1, -2, 0], [0, 0, 3]])
    qh = u.homogrophy_mapping(ph, H)
    Hnest = u.hest(qh, ph, normalize=True)
    Hest = u.hest(qh, ph)
    return H, Hest, Hnest


def test_ex9():
    H, Hest = ex9()
    Hest = u.format_H_to_solution(Hest).round(5)
    s = f"H:\n{H=}\n Hest:\n{Hest=}\n"
    S.save_txt("ex2-9", s)
    print("ex9")
    print(s)


def ex9():
    ph = np.random.randint(-100, 100, size=(3, 100))
    H = np.array([[-2, 0, 1], [1, -2, 0], [0, 0, 3]])
    qh = u.homogrophy_mapping(ph, H, set_scale_1=True)
    Hest = u.hest(qh, ph)
    return H, Hest


def test_ex10():
    pass


def ex10():
    pass


if __name__ == "__main__":
    #   test_ex1()
    #   test_ex2()
    test_ex4()
    #   test_ex5()
    # test_ex6()
    # test_ex7()
    # test_ex8()
    # test_ex9()
