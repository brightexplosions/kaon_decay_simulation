"""
read the data of the optimal z distances
and plot the histograms
and calculate uncertainties

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

mpl.rcParams['legend.fontsize'] = 10


Z = []
Z_scatter = []
P = []

with open("maxima3.dat") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line[0] == "#":
            continue
        
        n, _, use_scatter, p, z = line.split("\t")
        use_scatter = use_scatter.lower() == "true"
	
        if use_scatter:
            Z_scatter.append(float(z))
        else:
            P.append(float(p))
            Z.append(float(z))


avg = np.average(Z)
sigma = np.std(Z)

avg_scattered = np.average(Z_scatter)
sigma_scattered = np.std(Z_scatter)

print("######## without scattering #########")
print("average z:", avg)
print("sigma z  :", sigma)
print("quadratic added error: ", np.sqrt(sigma**2/avg**2 + (10/562)**2) * avg)
print("#####################################")
print("######## with scattering    #########")
print("average z:", avg_scattered)
print("sigma z  :", sigma_scattered)
print("quadratic added error: ", np.sqrt(sigma_scattered**2/avg_scattered**2 + (10/562)**2) * avg_scattered)
print("#####################################")
print(len(Z) + len(Z_scatter))


plt.figure(figsize=(5.7, 3.9))
plt.subplot(121)
plt.ylabel("N maxima in that region")
plt.xlabel("z position [m]")
plt.title("Optimal z for aligned beam\n {} simulations".format(len(Z)))
nbins, pos,_ = plt.hist(Z, 40)
mids = 0.5*pos[:-1] + 0.5*pos[1:]
diffs = -pos[:-1] + pos[1:]
area = np.dot(diffs, nbins)
gauss = stats.norm(avg, sigma).pdf(mids)
plt.plot(mids, area*gauss)

plt.subplot(122)
plt.title("Optimal z for scatterd beam\n {} simulations".format(len(Z_scatter)))
plt.xlabel("z position [m]")
nbins, pos,_ = plt.hist(Z_scatter, 40)
mids = 0.5*pos[:-1] + 0.5*pos[1:]
diffs = -pos[:-1] + pos[1:]
area = np.dot(diffs, nbins)
gauss = stats.norm(avg_scattered, sigma_scattered).pdf(mids)
plt.plot(mids, area*gauss)

plt.savefig("optimal_zpos.pdf")

plt.show()
