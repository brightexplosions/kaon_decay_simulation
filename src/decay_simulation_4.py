import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib as mpl
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
from time import time

import multiprocessing

from random_three_vector import isotropic_unit_vec
from random_2d_gaussian import gaussian_2d_angle_sample
from rotation_matrices import rot_x, rot_y


mpl.rcParams['legend.fontsize'] = 10
enableMultiProcessing = True

#############################
### general configuration ###
#############################

N = 10000 # amount of Kaons shot

show_3d_plot = True
use_scattering = False  # False: Beam parallel to the z axis, True: Divergent beam

detector_diameter = 4 # meters
detector_distance_plot = 500 # meters

# plot the first X decays that decayed in front of the detector
max_decays_to_plot = 40 # use N to plot all

z_space = np.linspace(210, 360, 50)


############
### Data ###
############

from data import kaon, pion_plus, pion_null
from data import SPEED_OF_LIGHT as C

avg_decay_length_kaons = 536.564512372  # lambda value of the exponentially decaying decay lengths
sig_theta = 0.001 # standard deviation of the normally distributed angle at the source

# pi 0 K frame
energy_pi_0 = (kaon.m ** 2 - pion_plus.m ** 2 + pion_null.m ** 2) / (2 * kaon.m)
p_pi_0 = np.sqrt(energy_pi_0 ** 2 - pion_null.m ** 2)

# pi + K frame
energy_pi_plus = (kaon.m ** 2 + pion_plus.m ** 2 - pion_null.m ** 2) / (2 * kaon.m)
p_pi_plus = np.sqrt(energy_pi_plus ** 2 - pion_plus.m ** 2)

############


# the matrix to change the referenceframe from kaon to lab
boost_mat = np.matrix([
      [kaon.gamma, 0, 0, kaon.beta*kaon.gamma],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [kaon.beta*kaon.gamma, 0, 0, kaon.gamma]])


def make_4_vec(momentum, mass):
    energy = np.sqrt(mass ** 2 + np.linalg.norm(momentum) ** 2)
    return np.matrix([energy, momentum[0], momentum[1], momentum[2]])


def boost_to_lab(four_vecs):
    """four_vecs: list of 4vectors"""
    return [boost_mat * fv.T for fv in four_vecs]


def boost_with_rotation_to_lab(four_vecs, rot, inv_rot):
    """rotated back, boost the 4 vecs, rotate again
        very ugly..
    """
    arr = []
    for fv, rotation, inv_rotation in zip(four_vecs, rot, inv_rot):
        e = fv.item(0)
        p = fv[:, [1, 2, 3]]
        p = inv_rotation * p.T
        fv = np.matrix([e, p.item(0), p.item(1), p.item(2)])

        fv = boost_mat * fv.T
        fv = fv.T

        e = fv.item(0)
        p = fv[:, [1, 2, 3]]
        p = rotation * p.T
        fv = np.matrix([e, p.item(0), p.item(1), p.item(2)])

        arr.append(fv)

    return arr


def simulate_kaons(n):
    """do the experiment for n kaons
        :returns
            decay positions
            4 vectors pi + (lab frame)
            4 vectors pi 0 (lab frame)
    """
    z_positions = stats.expon(scale = avg_decay_length_kaons).rvs(n)  # randomly generated decay lenghts
    decay_positions = np.array([np.zeros(n), np.zeros(n), z_positions]).T #the position vectors of the decay location without random rotation at the source

    phi, psi = gaussian_2d_angle_sample(n, sig_theta)  # random angles which will determine the direction of the kaon at the source
    rotation_matrices = [rot_y(b)*rot_x(a) for a, b in zip(phi, psi)]  # the matrices with which to rotate away from the z-axis
    rotation_matrices_inv = [rot_x(-a)*rot_y(-b) for a, b in zip(phi, psi)]  # the matrices with which to rotate back to the z-axis

    # apply rotations on the position vectors (turn off for 3)
    if use_scattering:
        for i in range(n):
            pos_vec = np.matrix(decay_positions[i]).T
            decay_positions[i] = (rotation_matrices[i] * pos_vec).squeeze()

    sample = isotropic_unit_vec(n)  # iscotropically distributed unit vectors
    momentum_pi_0 = sample * p_pi_0  # the impulse of pi_0 in the rest frame of the kaon
    momentum_pi_plus = -sample * p_pi_plus  # the impulse of pi_plus in the rest frame of the kaon

    pi_plus_4_vecs = []
    for momentum in momentum_pi_plus:
        vec = make_4_vec(momentum, pion_plus.m)
        pi_plus_4_vecs.append(vec)

    pi_null_4_vecs = []
    for momentum in momentum_pi_0:
        vec = make_4_vec(momentum, pion_null.m)
        pi_null_4_vecs.append(vec)

    if use_scattering:
        pi_plus_4_vecs_lab = boost_with_rotation_to_lab(pi_plus_4_vecs, rotation_matrices, rotation_matrices_inv)
        pi_null_4_vecs_lab = boost_with_rotation_to_lab(pi_null_4_vecs, rotation_matrices, rotation_matrices_inv)
    else:
        pi_plus_4_vecs_lab = boost_to_lab(pi_plus_4_vecs)
        pi_null_4_vecs_lab = boost_to_lab(pi_null_4_vecs)

    # for i in range(10):
    #     momentum1 = pi_null_4_vecs[i][:, [1, 2, 3]]
    #     print("angle", angle(momentum1, np.array([0,0,1])))
    #
    #     print(np.linalg.norm(momentum1))
    #
    #     print(pi_null_4_vecs[i])
    #     print(pi_null_4_vecs_lab[i] + pi_plus_4_vecs_lab[i])
    #     print("----------------")

    # reduce the dimension [[1],[1]] -> [1, 1]
    pi_null_4_vecs_lab = np.array(pi_null_4_vecs_lab).squeeze()
    pi_plus_4_vecs_lab = np.array(pi_plus_4_vecs_lab).squeeze()

    return decay_positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab


def angle(a, b):
    alpha = np.arccos(np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b)))
    return alpha


def is_in_circle(x, y, radius):
    return x**2 + y**2 <= radius**2


def single_hit(decay_position, fvec, z_detector, detector_radius):
    """check if one particle hit the detector"""
    p = fvec[1:]
    x, y, z = decay_position
    if z > z_detector:
        return False

    direction = p / np.linalg.norm(p)
    scale_factor = (z_detector - z) / direction[2]
    offset_vec = direction * scale_factor
    ox, oy, oz = offset_vec

    return is_in_circle(x + ox, y + oy, detector_radius)


def double_hit(decay_position, fvec1, fvec2, z_detector, detector_radius):
    """check if both particles hit the detector"""
    hit1 = single_hit(decay_position, fvec1, z_detector, detector_radius)
    hit2 = single_hit(decay_position, fvec2, z_detector, detector_radius)
    return hit1 and hit2


def count_double_hits(decay_positions, particle1_4vec, particle2_4vec, r=4, z_detector=100):
    counter = 0
    for decay_position, fvec1, fvec2 in zip(decay_positions, particle1_4vec, particle2_4vec):
        if double_hit(decay_position, fvec1, fvec2, z_detector, detector_radius=r):
            counter += 1

    return counter


def plot_decays(decay_positions, four_vecs_lab_particle1, four_vecs_lab_particle2, z_detector, r=5):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    decay_plot_count = 0

    for decay_pos, fv1, fv2 in zip(decay_positions, four_vecs_lab_particle1, four_vecs_lab_particle2):

            x, y, z = decay_pos
            if z > z_detector:
                continue

            decay_plot_count += 1
            if decay_plot_count > max_decays_to_plot:
                break

            p1 = fv1[1:]
            p2 = fv2[1:]
            direction1 = p1/np.linalg.norm(p1)
            direction2 = p2/np.linalg.norm(p2)

            scale_factor1 = (z_detector - z)/direction1[2]
            scale_factor2 = (z_detector - z)/direction2[2]

            dx1, dy1, dz1 = direction1 * abs(scale_factor1)
            dx2, dy2, dz2 = direction2 * abs(scale_factor2)

            if double_hit(decay_pos, fv1, fv2, z_detector, r):
                ax.plot([z, z+dz1], [y, y+dy1], [x, x+dx1], color="green")
                ax.plot([z, z+dz2], [y, y+dy2], [x, x+dx2], color="green")
                continue

            if single_hit(decay_pos, fv1, z_detector, r):
                ax.plot([z, z+dz1], [y, y+dy1], [x, x+dx1], color="orange")
                ax.plot([z, z+dz2], [y, y+dy2], [x, x+dx2], color="orange")
                continue

            if single_hit(decay_pos, fv2, z_detector, r):
                ax.plot([z, z+dz1], [y, y+dy1], [x, x+dx1], color="orange")
                ax.plot([z, z+dz2], [y, y + dy2], [x, x+dx2], color="orange")
                continue

            ax.plot([z, z + dz1], [y, y + dy1], [x, x + dx1], color="red")
            ax.plot([z, z + dz2], [y, y + dy2], [x, x + dx2], color="red")



    ax.plot([0, z_detector], [0, 0], [0, 0])
    ax.scatter([0], [0], [0], s=60, label='origin')

    #Mald Zentrum des Detektors
    ax.scatter([z_detector], [0], [0], "y",s=60, label='detector', color="red")

    #Malt den Detektorumfang, check nicht wie man die Achsen richtig skaliert
    edge_steps = 300
    edge = np.zeros((edge_steps, 3))
    for counter in range(edge_steps):
        angle = 2*np.pi*counter/edge_steps
        edge[counter] = [z_detector, r*np.sin(angle), r*np.cos(angle)]
    ax.scatter(edge[:,0], edge[:,1], edge[:,2], s=20, color="red")

    #setzt die Axen fest
    ax.set_ylim3d(-2.5, 2.5)
    ax.set_zlim3d(-2.5, 2.5)

    #zeichnet eine schoene legende
    red_line = art3d.Line3D(0, 0, 0, color="red")
    green_line = art3d.Line3D(0, 0, 0, color="green")
    blue_line = art3d.Line3D(0, 0, 0, color="blue")
    orange_line = art3d.Line3D(0, 0, 0, color="orange")
#    ax.legend([green_line, red_line, orange_line, blue_line],
#              ["hit, both particles", "miss, both particles", "miss, but other particle hit", "hit, but other missed"])
    plt.show()


def count_hits(space, positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab, r):

    hits = []
    for z in space:
        hits.append(count_double_hits(positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab, r=r, z_detector=z))
    return hits


def cubabola(x: object, a: object, b: object, c: object, d: object) -> object:
    return a + b*x +c*x**2 + d*x**3


if __name__ == "__main__":

    start_time = time()

    positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab = simulate_kaons(N)

    if enableMultiProcessing:
        # fancy way of executing count_hits
        number_cpus = multiprocessing.cpu_count()
        print("using multiprocessing with {} cpu cores.".format(number_cpus))


        n = int(np.ceil(len(z_space)/number_cpus))
        z_space_chunks = [z_space[i:i + n] for i in range(0, len(z_space), n)]

        args = [(space, positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab, detector_diameter/2) for space in z_space_chunks]

        p = multiprocessing.Pool(number_cpus)

        hits = p.starmap(count_hits, args)
        end_time = time()
        elapsed_time = end_time - start_time

        hits = sum(hits, [])

    else:
        print("not using multiprocessing.")
        start_time = time()
        hits = count_hits(z_space, positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab, detector_diameter/2)
        end_time = time()
        elapsed_time = end_time - start_time

    probability = np.array(hits)*100/N
    popt_cubab, _ = curve_fit(cubabola, z_space, probability)
    probability_fit = cubabola(z_space, *popt_cubab)
    max_y = max(probability_fit)
    max_index = list(probability_fit).index(max_y)


    print('Time elapsed: {} s'.format(elapsed_time))
    print("The maximum is reached at {} percent with a z-value of {} m".format(max_y, z_space[max_index]))

    # plot the acceptance of the detector
    fig2 = plt.figure(2)
    plt.plot(z_space, probability, "o")
    plt.plot(z_space, probability_fit)
    plt.title("Changing the z position of the detector ({} decays)\nwith sigma_theta of {} rad".format(N, sig_theta))
    plt.xlabel("Z distance of the detector [m]")
    plt.ylabel("hit percentage [%]")

    plt.show()

    if show_3d_plot:
        plot_decays(positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab,
                    z_detector=detector_distance_plot, r=detector_diameter / 2.0)