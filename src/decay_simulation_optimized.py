"""
the optimized version of decay_simulation.py

generates positions of decay positions, pion momenta.
determine best z with maximal acceptance.
saves data and a pdf to saves folder

see general configuration parameters

"""
from time import time
import os
import multiprocessing
from functools import partial

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


from random_three_vector import isotropic_unit_vec_optimized as isotropic_unit_vec
from random_2d_gaussian import gaussian_2d_angle_sample
from rotation_matrices import rot_x, rot_y
from rotation_matrices import rot_x_optimzed, rot_y_optimzed

#############################
### general configuration ###
#############################

enableMultiProcessing = True # disable if it does not run

N = 1200000 # amount of Kaons shot

use_scattering = True  # False: Beam parallel to the z axis, True: Divergent beam
show_plot = True # show the final plot
save_data = False

# the space of z values for the detector
start, stop, steps = 200, 380, 50
z_space = np.linspace(start, stop, steps)

detector_diameter = 4 # meters

############
### Data ###
############

from data import kaon, pion_plus, pion_null
from data import SPEED_OF_LIGHT as C

avg_decay_length_kaons_error_plus = 10.0499999998
avg_decay_length_kaons_error_minus = 9.84199999977
avg_decay_length_kaons = 562.310999999519

sig_theta = 0.001 # standard deviation of the normally distributed angle at the source

# pi 0 K frame
energy_pi_0 = (kaon.m ** 2 - pion_plus.m ** 2 + pion_null.m ** 2) / (2 * kaon.m)
p_pi_0 = np.sqrt(energy_pi_0 ** 2 - pion_null.m ** 2)

# pi + K frame
energy_pi_plus = (kaon.m ** 2 + pion_plus.m ** 2 - pion_null.m ** 2) / (2 * kaon.m)
p_pi_plus = np.sqrt(energy_pi_plus ** 2 - pion_plus.m ** 2)


# the matrix to change the referenceframe from kaon to lab
boost_mat = np.matrix([
      [kaon.gamma, 0, 0, kaon.beta*kaon.gamma],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [kaon.beta*kaon.gamma, 0, 0, kaon.gamma]])


def make_4_vec(momentum, mass):
    energy = np.sqrt(mass ** 2 + np.linalg.norm(momentum) ** 2)
    return np.matrix([energy, momentum[0], momentum[1], momentum[2]])

def make_four_vecs(momentum, mass, n):
    """create four vecs given momentum and mass"""
    f_vecs = np.zeros((n, 4))
    f_vecs[:,1:] = momentum.squeeze()
    f_vecs[:,0] = np.sqrt(np.einsum("ij, ij -> i", momentum.squeeze(), momentum.squeeze()) + mass**2)
    return np.matrix(f_vecs)


def boost_to_lab(four_vecs):
    """four_vecs: list of 4vectors"""
    return np.einsum("ij, li -> lj", boost_mat, four_vecs)


def boost_with_rotation_to_lab(four_vecs, rotations):
    """boost to lab frame and apply rotations"""
    boosted = np.einsum("ij, li -> lj", boost_mat, four_vecs)
    p = boosted[:,1:]
    boosted[:,1:] = np.einsum("ijk, ik -> ij", rotations, p)
    return boosted


def simulate_kaons(n, use_scattering):
    """do the experiment for n kaons
        :returns
            decay positions
            4 vectors pi + (lab frame)
            4 vectors pi 0 (lab frame)
    """
    z_positions = stats.expon(scale = avg_decay_length_kaons).rvs(n)  # randomly generated decay lenghts
    decay_positions = np.array([np.zeros(n), np.zeros(n), z_positions]).T #the position vectors of the decay location without random rotation at the source

    sample = isotropic_unit_vec(n)  # iscotropically distributed unit vectors

    momentum_pi_0 = sample * p_pi_0  # the impulse of pi_0 in the rest frame of the kaon
    momentum_pi_plus = -sample * p_pi_plus  # the impulse of pi_plus in the rest frame of the kaon

    pi_plus_4_vecs = make_four_vecs(momentum_pi_plus, pion_plus.m, n)
    pi_null_4_vecs = make_four_vecs(momentum_pi_0, pion_null.m, n)


    if use_scattering:
        phi, psi = gaussian_2d_angle_sample(n, sig_theta)  # random angles which will determine the direction of the kaon at the source

        rotation_x = rot_x_optimzed(phi)
        rotation_y = rot_y_optimzed(psi)

        # apply rotations on the position vectors
        rotation_matrices = np.einsum("ijk, ikl -> ijl", rotation_x, rotation_y)
        decay_positions = np.einsum("ijk, ik -> ij", rotation_matrices, decay_positions)

        pi_plus_4_vecs_lab = boost_with_rotation_to_lab(pi_plus_4_vecs, rotation_matrices)
        pi_null_4_vecs_lab = boost_with_rotation_to_lab(pi_null_4_vecs, rotation_matrices)
    else:
        pi_null_4_vecs_lab = boost_to_lab(pi_null_4_vecs)
        pi_plus_4_vecs_lab = boost_to_lab(pi_plus_4_vecs)

    # reduce the dimension [[1],[1]] -> [1, 1]
    pi_null_4_vecs_lab = np.array(pi_null_4_vecs_lab).squeeze()
    pi_plus_4_vecs_lab = np.array(pi_plus_4_vecs_lab).squeeze()

    return decay_positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab


def angle(a, b):
    alpha = np.arccos(np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b)))
    return np.rad2deg(alpha)


def single_hit(decay_position, four_vecs, z_detector, detector_radius):
    """check if one particle hit the detector"""
    p = four_vecs[:, 1:]
    x, y, z = decay_position.T

    # q is the position the particle will pass in the plane through z_detector
    q = p.T * (-z + z_detector) / p[:,2]
    qx, qy, qz = q

    # return whether the particles hit the circular disk
    return (x + qx)**2 + (y + qy)**2 < detector_radius**2


def count_double_hits(decay_positions, particle1_4vec, particle2_4vec, r, z_detector):
    """count for how many of the decays, particle1 and particle2 hit the detector"""

    # numpy filter to filter the ones that decay in front of the detector
    in_front = decay_positions[:,2] < z_detector

    # drop the decays that are behind
    decay_positions = decay_positions[in_front]
    particle1_4vec = particle1_4vec[in_front]
    particle2_4vec = particle2_4vec[in_front]

    # boolean array. True if hit, False if miss
    hits1 = single_hit(decay_positions, particle1_4vec, z_detector, r)
    hits2 = single_hit(decay_positions, particle2_4vec, z_detector, r)

    # combine the with elementwise and.
    hits = hits1 & hits2
    return len(decay_positions[hits])


def count_hits(space, positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab, r):
    """returns a list of hit counts for z values in space"""
    return [count_double_hits(positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab, r=r, z_detector=z) for z in space]


def poly_3(x, a, b, c, d):
    return a + b*x +c*x**2 + d*x**3

def do_experiment(use_scattering, N, plot_save_location, data_save_location, maximum_file):

    print("### starting simulation ###")

    start_time = time()

    # fix the use_scattering argument
    simulate_kaons_ = partial(simulate_kaons, use_scattering=use_scattering)

    #positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab = simulate_kaons(N)

    if enableMultiProcessing:

        number_cpus = multiprocessing.cpu_count()
        print("using multiprocessing with {} cpu cores.".format(number_cpus))

        # generate the data fancy way
        # simple way: positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab = simulate_kaons(N)
        with multiprocessing.Pool(number_cpus) as pool:
            simulation_args = [N//number_cpus] * (number_cpus - 1)
            simulation_args.append(N - sum(simulation_args))

            result = pool.map(simulate_kaons_, simulation_args)

            positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab = [], [], []
            for pos, pi_plus4v, pi_null4v in result:
                positions.extend(pos)
                pi_null_4_vecs_lab.extend(pi_null4v)
                pi_plus_4_vecs_lab.extend(pi_plus4v)

            positions = np.array(positions)
            pi_null_4_vecs_lab = np.array(pi_null_4_vecs_lab)
            pi_plus_4_vecs_lab = np.array(pi_plus_4_vecs_lab)

            print("time to generate data: {}".format(time() - start_time))

            # count the hits fancy way
            n = int(np.ceil(len(z_space)/number_cpus))
            z_space_chunks = [z_space[i:i + n] for i in range(0, len(z_space), n)]
            args = [(space, positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab, detector_diameter/2) for space in z_space_chunks]

            start_time1 = time()
            hits = pool.starmap(count_hits, args)
            end_time1 = time()

            print("time to count hits: ", end_time1 - start_time1)

        print("done evaluating.")

        hits = sum(hits, [])

    else:
        print("not using multiprocessing.")
        start_time = time()
        positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab = simulate_kaons_(N)
        hits = count_hits(z_space, positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab, detector_diameter/2)

    end_time = time()
    elapsed_time = end_time - start_time

    probability = np.array(hits)*100/N
    popt_cubab, _ = curve_fit(poly_3, z_space, probability)

    z_space_smooth = np.arange(start, stop, 0.001)

    probability_fit = poly_3(z_space_smooth, *popt_cubab)
    max_y = max(probability_fit)
    max_index = list(probability_fit).index(max_y)
    max_z = z_space_smooth[max_index]


    print('Total time elapsed: {:.2f} s'.format(elapsed_time))
    print("The maximum is reached at {:.2f} percent with a z-value of {:.2f} m".format(max_y, max_z))

    # plot the acceptance of the detector and save it

    fig = plt.figure()
    plt.plot(z_space, probability, "o")
    plt.plot(z_space_smooth, probability_fit)
    plt.title("Changing the z position of the detector ({} decays)\nwith sigma_theta of {} rad, scattering = {}".format(N, sig_theta, use_scattering))
    plt.xlabel("Z distance of the detector [m]")
    plt.ylabel("hit percentage [%]")
    plt.xlim([start, stop])


    if save_data:

        # save the data
        np.savetxt(data_save_location, np.array([z_space, probability]).T,
                   header="the maximum is determined with a cubic fit. (see plot pdf)\n"
                          "max_z: {}\n"
                          "max_p: {}\n"
                          "\n"
                          "z_space,\t\t\t\t probaiblity with {} kaons\n".format(max_y, max_z, N))


        # add entry with maxium etc to maximum_file
        with open(maximum_file, "a") as f:
            f.write("{}\t{}\t{}\t{}\t{}\n".format(N, avg_decay_length_kaons, use_scattering, max_y, max_z))

        # save the figure
        plt.savefig(plot_save_location)

    if show_plot:
        plt.show()

    plt.close()



def next_file(filename, base_dir = "saves"):
    """filename of the form 'name...%s.pdf' """
    plot_save_location = base_dir + os.sep + filename

    i = 0
    while os.path.exists(plot_save_location % i):
        i += 1

    return plot_save_location % i




if __name__ == "__main__":

    maximum_file = "saves" + os.sep + "maxima.dat"
    data_location = next_file("data_{}_{}to{}in{}_{}-%s.dat"
                              .format(N, start, stop, steps,avg_decay_length_kaons))
    plot_location = next_file("figure_{}_{}to{}in{}_{}-%s.pdf"
                              .format(N, start, stop, steps, avg_decay_length_kaons))

    do_experiment(use_scattering=use_scattering, N=N,
                  maximum_file=maximum_file, data_save_location=data_location,
                  plot_save_location=plot_location)
