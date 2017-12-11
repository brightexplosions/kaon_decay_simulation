import numpy as np


# constants
SPEED_OF_LIGHT = 299792458 # m/s

# kaon +
class kaon:
    tau = 1.2380 * 10**(-8) #s http://pdg.lbl.gov/2017/listings/rpp2017-list-K-plus-minus.pdf (2)
    mtau = 0.0020 * 10**(-8)

    m = 493.677 * 10**6 #eV http://pdg.lbl.gov/2017/listings/rpp2017-list-K-plus-minus.pdf (1)
    mm = 0.016 * 10**6

    p = 75 * 10**9 #eV

    energy = np.sqrt(m**2 + p**2)

    beta = abs(p)/energy
    gamma = abs(energy)/m


# pion +
class pion_plus:
    m = 139.57061 * 10**6 #eV http://pdg.lbl.gov/2017/listings/rpp2017-list-pi-plus-minus.pdf (1)
    mm = 0.00024 * 10**6

    mean_decay_length = 4188 #m
    
    p = 75 * 10**9 # eV
    
    energy = np.sqrt(m**2 + p**2)
    
    beta = abs(p)/energy
    gamma = abs(energy)/m

# pion 0
class pion_null:
    m = 134.9770 * 10**6 #eV http://pdg.lbl.gov/2017/listings/rpp2017-list-pi-zero.pdf (1)
    mm = 0.0005 * 10**6


# decay data
#decay_lengths_filename = "../dec_lengths.dat"
#decay_lengths = np.loadtxt(decay_lengths_filename)