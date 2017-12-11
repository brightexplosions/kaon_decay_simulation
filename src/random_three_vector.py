import numpy as np
from scipy import stats
from rotation_matrices import rot_x, rot_y

def isotropic_unit_vec(N):
    """Generates N isotropically distributed 3d unit vectors.

    Parameters
    ----------
    N : int
        The amount of vectors to be generated.

    Return
    ------
    Nx3 array
        An array containing N isotropically distributed 3d unit vecotrs (3x1 matrices).
    """

    phi_int = stats.uniform(0, 1).rvs(N) #the random basis from which we calculate the non-uniformly distributed agle phi
    psi = stats.uniform(0, 2*np.pi).rvs(N) #the uniformly distributed angle psi

    phi = np.arcsin(2*phi_int-1) #the formula with which we calculate phi from the uniform distribution (you can find the derivation in the report)

    start_vector = np.matrix([0, 0, 1]).T #the vector that will be rotated by random angles
    vectors = np.array([rot_y(b)*rot_x(a)*start_vector for a, b in zip(phi, psi)]) #an array containing the isotropically distributed vectors
    return vectors


# should be a faster version..
def isotropic_unit_vec_optimized(N):
    phi = stats.uniform(0, np.pi * 2).rvs(N)
    costheta = stats.uniform(-1, 2).rvs(N)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.array([x, y, z]).T


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    vec = isotropic_unit_vec(300)
    x = np.array(vec)[:,0]
    y = np.array(vec)[:,1]
    z = np.array(vec)[:,2]


    #plot to check whether we messed up
    #fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
    fig = plt.figure(figsize=(5.7, 4.2))
#    plt.figure(figsize=(5.7, 4.2))

    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.scatter(x, y, z, s=10, c='r', zorder=10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.savefig('../report/images/three_vector.pdf')
    plt.show()
