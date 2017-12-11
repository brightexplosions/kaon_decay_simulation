import numpy as np

def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    return (x,y,z)

def spherical_sample(N=100):
    points = []

    for i in range(N):
        points.append(random_three_vector())

    return np.array(points).T


if __name__ == "__main__":
    # Example IPython code to test the uniformity of the distribution
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    x, y, z = spherical_sample(1000)

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
    ax.scatter(x, y, z, s=100, c='r', zorder=10)
    plt.show()