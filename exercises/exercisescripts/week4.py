import utils as u
import matplotlib.pyplot as plt

if __name__ == "__main__":
    points = u.checkerboard_points(7, 10)   
    plt.scatter(*points)
    plt.show()

