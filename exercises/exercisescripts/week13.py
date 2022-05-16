import numpy as np
from pathlib import Path
from cv2 import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d

folder_path = Path( "../../data/casper/" )
img_path = folder_path / "sequence"
k0, d0, k1, d1, r, t = np.load(
            folder_path / "calib.npy", 
            allow_pickle=True
            ).item().values()

def ex0():
    # Comment on how to rectify intrinsic
    # and extrinsic values
    pass

def c2g(im, normalize=True):
    g_im =  np.mean(im, axis=2)
    if normalize: g_im /= 255
    return g_im

def rectify_imgs(img_names: list, maps: list):
    r_imgs = []
    for im_name in img_names:
        im = c2g(cv2.imread(im_name), normalize=False) 
        r_im = cv2.remap(im, *maps, cv2.INTER_LINEAR)
        r_imgs.append(r_im)
    return r_imgs

def ex1() -> tuple:
    im0  = cv2.imread((img_path / "frames0_0.png").as_posix())
    size = (im0.shape[1], im0.shape[0])
    stereo = cv2.stereoRectify(k0, d0, k1, d1,
                               size, r, t, flags=0)
    R0, R1, P0, P1 = stereo[:4]
    maps0 = cv2.initUndistortRectifyMap(k0, d0, R0, P0, size, cv2.CV_32FC2)
    maps1 = cv2.initUndistortRectifyMap(k1, d1, R1, P1, size, cv2.CV_32FC2)
    
    ims0_names = [(img_path / f"frames0_{i}.png").as_posix() for i in range(26)]
    ims1_names = [(img_path / f"frames1_{i}.png").as_posix() for i in range(26)]
    ims0 = rectify_imgs(ims0_names, maps0)
    ims1 = rectify_imgs(ims1_names, maps1)
    
    return ims0, ims1, [R0, R1, P0, P1]
   
def test_ex1(im0, im1):
    _, ( ax0, ax1 ) = plt.subplots(1, 2)
    ax0.imshow(im0)
    ax1.imshow(im1)
    plt.show()

def unwrap(ims: list):
    # List of primary and secondary img
    prime_ims = ims[2:18]     
    second_ims = ims[18:]
    
    n_primary = 40

    # Checking lengths are correct
    assert len( prime_ims ) == 16 and len( second_ims ) == 8 
    
    # Wrapped phases
    fft_prime = np.fft.rfft(prime_ims, axis=0)[1]
    theta_prime = np.angle(fft_prime)

    fft_second = np.fft.rfft(second_ims, axis=0)[1]
    theta_second = np.angle(fft_second)

    # Phace queue
    theta_c = np.mod((theta_prime - theta_second), 
                     (2*np.pi) )

    # Primary order
    o_prime = np.rint((n_primary * theta_c - theta_prime)/(2 * np.pi))
    
    # Theta_est
    theta_est = ( (2 * np.pi * o_prime + theta_prime)/n_primary ) % (2 * np.pi)
    
    return theta_est


def ex2(ims0: list, ims1: list):
    theta0 = unwrap(ims0)
    theta1 = unwrap(ims1)
    return theta0, theta1

def test_ex2(theta0, theta1):
    _, ( ax0, ax1 ) = plt.subplots(1, 2)
    ax0.imshow(theta0)
    ax1.imshow(theta1)
    plt.show() 

def create_mask(ims: list, threshold=15):
    mask = (ims[0] - ims[1]) > threshold
    return mask

def ex3(ims0, ims1):
    mask0 = create_mask(ims0)
    mask1 = create_mask(ims1)
    return mask0, mask1

def test_ex3(mask0, mask1):
    _, ( ax0, ax1 ) = plt.subplots(1, 2)
    ax0.imshow(mask0, cmap="gray")
    ax1.imshow(mask1, cmap="gray")
    plt.show() 

def compute_disparity_matrix(t0, t1, m0, m1, verbose=True):
    disp = np.zeros_like(t0)
    q0s = []
    q1s = []
    if verbose: print("Finding closest matches") 
    for i0, row in tqdm(enumerate(t0), total=t0.shape[0],disable = not verbose):
        for j0, col in enumerate(row):
            if m0[i0, j0]:
                valid_idx = np.where(m1[i0] == True)[0]
                if len(valid_idx) == 0:
                    continue
                # Kind of a weird workaround
                closest = np.argmin(abs(col - t1[i0, valid_idx]))
                j1 = valid_idx[closest]
                disp[i0, j0] = j0 - j1
                
                q0s.append([j0, i0])
                q1s.append([j1, i0])
                
    return disp, q0s, q1s 

def ex4(t0, t1, m0, m1, verbose=True):
    return compute_disparity_matrix(t0, t1, m0, m1, verbose) 

def test_ex4(disp):
    im = cv2.medianBlur(disp.astype(np.float32), 5)
    # breakpoint()
    plt.imshow(im)
    plt.show()

def triangulate_points():
    pass

def ex5(q0, q1, p0, p1):
    # Reshaping matches to adhere
    q0_t = np.array(q0, dtype=np.float32).T
    q1_t = np.array(q1, dtype=np.float32).T

    Q = cv2.triangulatePoints(p0, p1, q0_t, q1_t)
    Q = Q[:-1] / Q[-1]
    return Q

def test_ex5(Q):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Q.T)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    ims0, ims1, [*_, p0, p1] = ex1()
    # test_ex1(ims0[0], ims1[0])
    theta0, theta1 = ex2(ims0, ims1)
    # test_ex2(theta0, theta1)
    mask0, mask1 = ex3(ims0, ims1)
    # test_ex3(mask0, mask1)
    disparity, q0s, q1s = ex4(theta0, theta1, mask0, mask1, verbose=False)
    # test_ex4(disparity)
    Q = ex5(q0s, q1s, p0, p1)
    test_ex5(Q)
