import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


data = np.loadtxt('saves_10/data_1000000_215to350in85_562.310999999519-165.dat')

x = data[:,0]
y = data[:,1]


space = np.linspace(min(x), max(x), 10000)

def f(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3

popt, _ = curve_fit(f, x, y)


# Latex style settings

plt.rc('text.latex',preamble=r'\usepackage{lmodern}')
plt.rc('text',usetex=True)
plt.rc('font', family='serif', size=12)
plt.rc('font', weight='bold')
#    plt.rc('xtick.major', size=5, pad=7)
plt.rc('xtick', labelsize=12)
plt.rc('ytick.major', size=5, pad=3)
plt.rc('ytick', labelsize=12)

plt.figure(figsize=(5.7, 4.2))
plt.xlabel("Z distance of the detector [m]")
plt.ylabel("hit percentage [\%]")
plt.plot(x, y, 'o')
plt.grid()
plt.plot(space, f(space, *popt))
plt.savefig('../report/images/sample_figure_1000000.pdf')

plt.show()
