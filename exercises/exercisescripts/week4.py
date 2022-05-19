from __future__ import annotations
import utils as u
from exam_utils import ImgSaver
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

S = ImgSaver()
data_path = Path("../../data/TwoImageData.npy")


def test_ex1():
    P, Q, q = ex1()
    qs = "\n".join([f"p:{x} -projects> q:{y}" for x, y in zip(Q.T, q.T)])
    s = f"{P = }\n{qs}"
    S.save_txt("ex4-1", s)


def ex1():
    R = np.array(
        [[np.sqrt(0.5), -np.sqrt(0.5), 0], [np.sqrt(0.5), np.sqrt(0.5), 0], [0, 0, 1]]
    )

    t = np.array([[0, 0, 10]]).T
    Q = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).reshape(3, -1)
    Qh = u.inhom_to_hom(Q)

    f = 1000
    w, h = 1920, 1080
    dx, dy = w // 2, h // 2

    K = u.create_cameramatrix(f, 1, 0, dx, dy)
    P = u.make_projection_matrix(K, R, t)

    q = P @ Qh
    q = u.hom_to_inhom(q)
    return P, Q, q


def test_ex2():
    p, p_est, p_nest = ex2()
    print(np.linalg.norm(p - p))  # type: ignore
    print(np.linalg.norm(p - p_est))  # type: ignore
    print(np.linalg.norm(p - p_nest))  # type: ignore


def ex2():
    _, Q, q = ex1()
    qh = u.inhom_to_hom(q)
    Qh = u.inhom_to_hom(Q)
    Pest = u.DLT(Qh, qh)
    Pnest = u.DLT(Qh, qh, normalize=True)

    p_est = u.hom_to_inhom(Pest @ Qh)
    p_nest = u.hom_to_inhom(Pnest @ Qh)

    np.set_printoptions(suppress=True)
    return q, p_est, p_nest


def test_ex3():
    points = ex3()
    plt.scatter(*points)


def ex3():
    points = u.checkerboard_points(7, 10)
    return points


def test_ex4():
    qs, p_qs = ex4()

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    [ax.scatter3D(*q) for q in qs]
    S.save_fig("ex4-4-1")
    plt.close()

    plt.figure(figsize=(10, 10))
    [plt.scatter(*q) for q in p_qs]
    S.save_fig("ex4-4-2")


def ex4():
    P, Q, q = ex1()

    ra = Rotation.from_euler("xyz", [np.pi / 10, 0, 0]).as_matrix()
    rb = Rotation.from_euler("xyz", [0, 0, 0]).as_matrix()
    rc = Rotation.from_euler("xyz", [-np.pi / 10, 0, 0]).as_matrix()

    board = u.checkerboard_points(10, 20)
    a = ra @ board
    b = rb @ board
    c = rc @ board

    qa = u.hom_to_inhom(P @ u.inhom_to_hom(a))
    qb = u.hom_to_inhom(P @ u.inhom_to_hom(b))
    qc = u.hom_to_inhom(P @ u.inhom_to_hom(c))

    return [a, b, c], [qa, qb, qc]


def test_ex5():
    Q_omega, qs, homographies = ex5()
    Qo_tilde = u.inhom_to_hom(Q_omega[:2])

    qa_esth = homographies[0] @ Qo_tilde
    qb_esth = homographies[1] @ Qo_tilde
    qc_esth = homographies[2] @ Qo_tilde

    qa_est = u.hom_to_inhom(qa_esth)
    qb_est = u.hom_to_inhom(qb_esth)
    qc_est = u.hom_to_inhom(qc_esth)

    aerr = np.sum((qs[0] - qa_est) / np.linalg.norm(qs[0]))
    berr = np.sum((qs[1] - qb_est) / np.linalg.norm(qs[1]))
    cerr = np.sum((qs[2] - qc_est) / np.linalg.norm(qs[2]))
    print(aerr)
    print(berr)
    print(cerr)


def ex5():
    Q_omega = u.checkerboard_points(10, 20)
    _, qs = ex4()
    homographies = u.estimate_homopgraphies(Q_omega, qs)

    return Q_omega, qs, homographies


def test_ex6():
    f = 1000
    w, h = 1920, 1080
    dx, dy = w // 2, h // 2

    K = u.create_cameramatrix(f, 1, 0, dx, dy)

    true_b = np.linalg.inv(K).T @ np.linalg.inv(K)
    b = ex6()

    [b11, b12, b22, b13, b23, b33] = b
    B = np.array([[b11, b12, b13], [b12, b22, b23], [b13, b23, b33]])
    B = (B / B[-1, -1]) * true_b[-1, -1]
    np.set_printoptions(suppress=True)
    print(true_b)
    print("====================")
    print(B)
    print("====================")
    print(np.linalg.norm(true_b - B) / np.linalg.norm(true_b))


def ex6() -> np.ndarray:
    *_, Hs = ex5()
    b = u.estimate_b(Hs)
    return b


def test_ex7():
    f = 1000
    w, h = 1920, 1080
    dx, dy = w // 2, h // 2
    K = u.create_cameramatrix(f, 1, 0, dx, dy)

    Kest = ex7()

    np.set_printoptions(suppress=True)
    print(K)
    print(Kest)


def ex7():
    *_, Hs = ex5()
    Kest = u.estimate_intrinsics(Hs)
    return Kest


def test_ex8():
    Kest = ex7()

    ra = Rotation.from_euler("xyz", [np.pi / 10, 0, 0]).as_matrix()
    rb = Rotation.from_euler("xyz", [0, 0, 0]).as_matrix()
    rc = Rotation.from_euler("xyz", [-np.pi / 10, 0, 0]).as_matrix()

    rrots = [ra, rb, rc]

    t = np.array([[0, 0, 10]]).T

    rots, tran = ex8()
    
    np.set_printoptions(suppress=True)
    for rr, r in zip(rrots, rots):
        print(r/r[0,0])
        print(rr) 

    print(tran)
    print(t)


def ex8():
    Kest = ex7()
    *_, Hs = ex5()
    rots, tran = u.estimate_extrinsics(Kest, Hs)
    return rots, tran


if __name__ == "__main__":
    # test_ex1()
    test_ex2()
    # test_ex3()
    # test_ex4()
    # test_ex5()
    # test_ex6()
    # test_ex7()
    test_ex8()
