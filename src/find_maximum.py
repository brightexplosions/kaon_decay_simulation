import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt



data_files = \
    ['data_1000000_215to360in80_552.468999999749-0.dat',
    'data_1000000_215to360in80_562.310999999519-0.dat',
    'data_1000000_215to360in80_572.360999999319-0.dat',
    'data_1000000_215to360in80_572.360999999319-1.dat']


z_space = np.arange(215,360,0.005)

def cubabola(x: object, a: object, b: object, c: object, d: object) -> object:
    return a + b*x +c*x**2 + d*x**3


max_z_list = []

for file in data_files:
    path = "saves" + os.sep + file
    z, p = np.loadtxt(path).T
    popt, _ = curve_fit(cubabola, z, p)

    prob_fit = cubabola(z_space, *popt)
    max_p = max(prob_fit)
    max_index = list(prob_fit).index(max_p)
    max_z = z_space[max_index]
    max_z_list.append(max_z)


    plt.plot(z, p, "x")
    plt.plot(z_space, cubabola(z_space, *popt))
    plt.axhline(max_p)
    plt.axvline(max_z)

plt.show()

print(max_z_list)