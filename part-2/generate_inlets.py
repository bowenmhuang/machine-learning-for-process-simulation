'''
Generate inlets for the ethylene oxide flash drum model.
'''

import pickle

import numpy as np

from find_enthalpy import find_enthalpy


def generate_inlet():
    T = np.random.uniform(0, 1)
    P = np.random.uniform(0, 1)
    n = np.random.uniform(0, 1)
    E = np.random.uniform(0, 1)
    O = np.random.uniform(0, 1)
    EO = np.random.uniform(0, 1)
    C = np.random.uniform(0, 1)
    W = np.random.uniform(0.5, 1)
    N = np.random.uniform(0, 1)
    molar_flows = [E, O, EO, C, W, N]
    mole_fracs = [molar_flows[i] / sum(molar_flows[:]) for i in range(6)]
    stream = [T, P, n, *mole_fracs]
    H = find_enthalpy(stream)
    stream.append(H)
    vf = np.random.uniform(0, 1)
    return [stream, vf]


def generate_inlets(pts):
    random_inlets = []
    for i in range(pts):
        random_inlets.append(generate_inlet())
    with open("part2/pickled_data/random_inlets_{}.pkl".format(pts), "wb") as f:
        pickle.dump(random_inlets, f)


