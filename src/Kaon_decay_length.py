#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 09:28:43 2017

@author: stefanie
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from data import SPEED_OF_LIGHT as c
from data import kaon, pion_plus



# Latex style settings
plt.rc('text.latex',preamble=r'\usepackage{lmodern}')
plt.rc('text',usetex=True)
plt.rc('font', family='serif', size=12)
plt.rc('font', weight='bold')
#    plt.rc('xtick.major', size=5, pad=7)
plt.rc('xtick', labelsize=12)
plt.rc('ytick.major', size=5, pad=3)
plt.rc('ytick', labelsize=12)


##########
## Data ##
##########

lengths = np.loadtxt("../dec_lengths.dat")
length_Ks = np.arange(542, 583, 0.1)
tau_pi = 4188 /  (pion_plus.beta * pion_plus.gamma * c)
length_pi = 4188

#######################
## Histogram of Data ##
#######################

plt.figure(figsize=(5.7, 4.2))
plt.hist(lengths, bins = 25, range=[0, 12000])
plt.xlabel('Decay Length [m]')
plt.ylabel('Number of particles')
plt.savefig('../report/images/histogram_decay_length.pdf')
#plt.show()

########################################
## Log Likelihood Function Definition ##
########################################

def f(length, length_K):
    return np.log((0.16/ length_K) * np.exp(-length / length_K) + (0.84/ length_pi) * np.exp(-length/ length_pi))


def ll(length_K):
    total = 0
    for length in lengths:
        total += f(length, length_K)
    return total

#####################################
## Execute Log Likelihood Function ##
#####################################

LLs = ll(length_Ks)

#################################
## Find optimum of LL Function ##
#################################

length_K_hut = length_Ks[list(LLs).index(max(LLs))]
print(length_Ks)

###########################
## Error of Decay Length ##
###########################

a = LLs - max(LLs) > -0.5
length_low = min(length_Ks[a])
length_high = max(length_Ks[a])
print(50 * "=")
print("Average Decay Length of K+ = {}".format(length_K_hut))
print(50 * "-")
print("Error plus = ", length_high - length_K_hut)
print("Error minus = ", length_K_hut - length_low)
print(50 * "=")


#########################
## Conversion into Tau ##
## Error of Tau        ##
#########################

tau_K = length_K_hut / (kaon.beta * kaon.gamma * c)
tau_K_low = length_low / (kaon.beta * kaon.gamma * c)
tau_K_high = length_high / (kaon.beta * kaon.gamma * c)
error_plus = tau_K_high - tau_K
error_minus = tau_K - tau_K_low
print(50 * "=")
print("Tau_K = ", tau_K)
print(50 * "-")
print("Error plus = ", error_plus)
print("Error minus = ", error_minus)
print(50 * "=")

##################################
## Plot Log Likelihood Function ##
##################################

plt.figure(figsize=(5.7, 4.2))
plt.plot(length_Ks, LLs)
plt.axhline(max(LLs))
plt.axvline(length_K_hut)
plt.axhline(max(LLs)-0.5, linestyle = "dashed", label = 'max$(\ln L) - 0.5$', color = "red")
plt.axvline(length_low, linestyle = "dashed", label = '$adl_K \pm \sigma$', color = "orange")
plt.axvline(length_high, linestyle = "dashed", color = "orange")
plt.xlabel('Average Decay Length of $K^+$ [m]')
plt.ylabel('$\ln L(dl_i|adl_{K^+})$')
plt.legend(loc='best')
plt.savefig('../report/images/log_likelihood_fuction.pdf')
#plt.show()



