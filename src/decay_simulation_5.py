"""
first arg: N
second arg: use_scattering
third arg: z_space (start,stop,step)

example usage:
python3 decay_simulation_5.py 10000 True (200,400,80)

"""
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib as mpl
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
from time import time
import os
import sys

import multiprocessing

from random_three_vector import isotropic_unit_vec
from random_2d_gaussian import gaussian_2d_angle_sample
from rotation_matrices import rot_x, rot_y

mpl.rcParams['legend.fontsize'] = 10
enableMultiProcessing = False


#############################
### general configuration ###
#############################

N = 40000 # amount of Kaons shot

use_scattering = False  # False: Beam parallel to the z axis, True: Divergent beam

show_plot = True # show the final plot

show_3d_plot = False
# plot the first X decays that decayed in front of the detector
max_decays_to_plot = 50 # use N to plot all
detector_diameter = 4 # meters
detector_distance_plot = 500 # meters

start, stop, steps = 215, 350, 80
z_space = np.linspace(start, stop, steps)

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

############
## comand line args ##
if len(sys.argv) == 1:
    pass
elif len(sys.argv) == 2:
    N = int(sys.argv[1])

elif len(sys.argv) == 3:
    N = int(sys.argv[1])
    assert sys.argv[2].lower() in ["true", "false"] , "second arg need to be true or false"
    use_scattering = sys.argv[2].lower() == "true"

elif len(sys.argv) == 4:
    import re
    linspace_string = sys.argv[3]

    match = re.match("\((\d+),(\d+),(\d+)\)", linspace_string)
    if match:
        start = int(match.group(1))
        stop = int(match.group(2))
        steps = int(match.group(3))
        z_space = np.linspace(start, stop, steps)
    else:
        raise ValueError("third arg needs to be of the form (a,b,c)")


else:
    raise ValueError("too many arguments supplied.")

print("########################################")
print("## running with the following options ##")
print("use_scattering = ", use_scattering)
print("N = ", N)
print("z_space = ({},{},{})".format(start, stop, steps))
print("########################################")

###############
### file stuff ###

maximum_file = "saves" + os.sep + "maxima.dat"

plot_save_location = "saves" + os.sep + "figure_{}_{}to{}in{}_{}-%s.pdf".format(N, start, stop, steps,
                                                                               avg_decay_length_kaons)
data_save_location = "saves" + os.sep + "data_{}_{}to{}in{}_{}-%s.dat".format(N, start, stop, steps,
                                                                             avg_decay_length_kaons)

i = 0
while os.path.exists(plot_save_location % i):
    i += 1
plot_save_location = plot_save_location % i

i = 0
while os.path.exists(data_save_location % i):
    i += 1
data_save_location = data_save_location % i


#################


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

    # reduce the dimension [[1],[1]] -> [1, 1]
    pi_null_4_vecs_lab = np.array(pi_null_4_vecs_lab).squeeze()
    pi_plus_4_vecs_lab = np.array(pi_plus_4_vecs_lab).squeeze()

    return decay_positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab


def angle(a, b):
    alpha = np.arccos(np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b)))
    return np.rad2deg(alpha)


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


def count_double_hits(decay_positions, particle1_4vec, particle2_4vec, r, z_detector):
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
    green_line = art3d.Line3D(0, 0, 0, color="green")
    red_line = art3d.Line3D(0, 0, 0, color="red")
    orange_line = art3d.Line3D(0, 0, 0, color="orange")
    ax.legend([green_line, red_line, orange_line],
              ["hit, both particles", "miss, both particles", "1 hit, 1 missed"])
    plt.show()


def count_hits(space, positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab, r):

    hits = []
    for z in space:
        hits.append(count_double_hits(positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab, r=r, z_detector=z))
    return hits


def cubabola(x, a, b, c, d):
    return a + b*x +c*x**2 + d*x**3


if __name__ == "__main__":

    start_time = time()
    #positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab = simulate_kaons(N)

    if enableMultiProcessing:

        number_cpus = multiprocessing.cpu_count()
        print("using multiprocessing with {} cpu cores.".format(number_cpus))

        # generate the data fancy way
        # simple way: positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab = simulate_kaons(N)
        with multiprocessing.Pool(number_cpus) as pool:
            simulation_args = [N//number_cpus] * (number_cpus - 1)
            simulation_args.append(N - sum(simulation_args))
            a = 1
            result = pool.map(simulate_kaons, simulation_args)

        positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab = [], [], []
        for pos, pi_plus4v, pi_null4v in result:
            positions.extend(pos)
            pi_null_4_vecs_lab.extend(pi_null4v)
            pi_plus_4_vecs_lab.extend(pi_plus4v)

        print("done generating data.")

        # count the hits fancy way
        with multiprocessing.Pool(number_cpus) as pool:
            n = int(np.ceil(len(z_space)/number_cpus))
            z_space_chunks = [z_space[i:i + n] for i in range(0, len(z_space), n)]
            args = [(space, positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab, detector_diameter/2) for space in z_space_chunks]

            p = multiprocessing.Pool(number_cpus)

            start_time1 = time()
            hits = p.starmap(count_hits, args)
            end_time1 = time()

            print("time of count_hits: ", end_time1 - start_time1)


        print("done evaluating.")

        hits = sum(hits, [])

    else:
        print("not using multiprocessing.")
        start_time = time()
        positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab = simulate_kaons(N)
        hits = count_hits(z_space, positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab, detector_diameter/2)


    end_time = time()
    elapsed_time = end_time - start_time

    probability = np.array(hits)*100/N
    popt_cubab, _ = curve_fit(cubabola, z_space, probability)

    z_space_smooth = np.arange(start, stop, 0.01)

    probability_fit = cubabola(z_space_smooth, *popt_cubab)
    max_y = max(probability_fit)
    max_index = list(probability_fit).index(max_y)
    max_z = z_space_smooth[max_index]


    print('Time elapsed: {:.2f} s'.format(elapsed_time))
    print("The maximum is reached at {:.2f} percent with a z-value of {:.2f} m".format(max_y, max_z))

    # save the data
    np.savetxt(data_save_location, np.array([z_space, probability]).T,
               header="the maximum is determined with a cubic fit. (see plot pdf)\n"
                      "max_z: {}\n"
                      "max_p: {}\n"
                      "\n"
                      "z_space,\t\t\t\t probaiblity with {} kaons\n".format(max_y, max_z, N))

    with open(maximum_file, "a") as f:
        f.write("{}\t{}\t{}\t{}\t{}\n".format(N, avg_decay_length_kaons, use_scattering, max_y, max_z))

    # plot the acceptance of the detector and save it

    fig2 = plt.figure(2)
    plt.plot(z_space, probability, "o")
    plt.plot(z_space_smooth, probability_fit)
    plt.title("Changing the z position of the detector ({} decays)\nwith sigma_theta of {} rad, scattering = {}".format(N, sig_theta, use_scattering))
    plt.xlabel("Z distance of the detector [m]")
    plt.ylabel("hit percentage [%]")
    plt.xlim([start, stop])
    plt.savefig(plot_save_location)

    if show_plot:
        plt.show()

    if show_3d_plot:
        plot_decays(positions, pi_plus_4_vecs_lab, pi_null_4_vecs_lab,
                    z_detector=detector_distance_plot, r=detector_diameter / 2.0)
