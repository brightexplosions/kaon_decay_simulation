import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def gaussian_2d_angle_sample(N, sig_theta):

    norm = stats.norm(0, sig_theta)
    uniform = stats.uniform(0,np.pi*2)

    alpha = norm.rvs(N)
    beta = uniform.rvs(N)

    return np.array([alpha*np.cos(beta), alpha*np.sin(beta)])


if __name__ == "__main__":
    N = 1000
    x, y = gaussian_2d_angle_sample(N, sig_theta = 1 * 10**(-3))
    plt.axis("equal")
    plt.plot(x, y, "r.")


    n = 0
    for xi, yi in zip(x, y):
        if xi**2 + yi**2 < 0.001**2:
            n += 1

    print(n*100/N)
    plt.show()
