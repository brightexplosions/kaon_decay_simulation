import numpy as np

def rot_x(phi):
    """Calculates a rotation matrix around the x-axis for the angle phi.
    
    Parameters
    ----------
    phi : float
        The value of the angle in radians.
        
    Return
    ------
    3x3 matrix
        A 3x3 matrix, such that its multiplication with a vector will rotate the vector by phi radians around the x-axis. 
    """
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    rot_mat = np.matrix([
        [1, 0, 0],
        [0, cos_phi, sin_phi ],
        [0, -sin_phi, cos_phi]])
    return rot_mat

def rot_y(phi):
    """Calculates a rotation matrix around the y-axis for the angle phi.
    
    Parameters
    ----------
    phi : float
        The value of the angle in radians.
        
    Return
    ------
    3x3 matrix
        A 3x3 matrix, such that its multiplication with a vector will rotate the vector by phi radians around the y-axis. 
    """
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    rot_mat = np.matrix([
        [cos_phi, 0, sin_phi ],
        [0, 1, 0],
        [-sin_phi, 0, cos_phi]])
    return rot_mat

def rot_x_optimzed(phi):
    n = len(phi)

    sin_phi = np.sin(phi).reshape((n, 1))
    cos_phi = np.cos(phi).reshape((n, 1))

    rotx = np.zeros((n, 3, 3))
    rotx[:, 1::3, 1] = cos_phi
    rotx[:, 1::3, 2] = sin_phi
    rotx[:, 2::3, 1] = -sin_phi
    rotx[:, 2::3, 2] = cos_phi
    rotx[:, 0::3, 0] = np.ones((n, 1))

    return rotx

def rot_y_optimzed(psi):
    n = len(psi)

    sin_psi = np.sin(psi).reshape((n, 1))
    cos_psi = np.cos(psi).reshape((n, 1))

    roty = np.zeros((n, 3, 3))
    roty[:, 0::3, 0] = cos_psi
    roty[:, 0::3, 2] = sin_psi
    roty[:, 2::3, 0] = -sin_psi
    roty[:, 2::3, 2] = cos_psi
    roty[:, 1::3, 1] = np.ones((n, 1))

    return roty

if __name__ == "__main__":
    a = np.array([1,2,3,4])
    b = np.array([1,2,3,4])
    c = rot_x_optimzed(a)
    d = rot_y_optimzed(b)
    print(c)
    print(d)

