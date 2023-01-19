"""
These functions generate the training data of inlet-outlet pairs of the reactor,
for different sampling techniques of the input space: random, stratified, latin hypercube.
"""

import time
import pickle
from itertools import product

import numpy as np
from skopt.space import Space
from skopt.sampler import Lhs

from ethylene_oxide_reactor_model import UnitOp, Stream


def generate_data_random_sampling(pts):
    """random sampling"""
    t0 = time.process_time()
    inlets = []
    outlets = []
    # generate a set of randomly sampled inlet streams within the potential operating conditions - 5 input variables T,C2H4,O2,Q,cat
    for i in range(pts):
        # generate inlet. pressure is not varying. there are no products in the inlet.
        T = np.random.uniform(0, 1)
        P = 0.5
        C2H4 = np.random.uniform(0, 1)
        O2 = np.random.uniform(0, 1)
        Q = np.random.uniform(0, 1)
        cat = np.random.uniform(0, 1)
        s = [T, P, C2H4, O2, 0, 0, 0]
        # calculate outlet
        result = UnitOp(Stream(*s), [Q, cat]).results()
        # record the inlet and outlet
        inlets.append([T, C2H4, O2, Q, cat])
        outlets.append(result)
    with open("part1/pickled_data/random_inlets_{}.pkl".format(pts), "wb") as f:
        pickle.dump(inlets, f)
    with open("part1/pickled_data/random_outlets_{}.pkl".format(pts), "wb") as f:
        pickle.dump(outlets, f)
    return (time.process_time() - t0) / pts * 100


def generate_data_stratified_sampling(pts):
    """stratified sampling"""
    t0 = time.process_time()
    inlets = []
    outlets = []
    # number of segments to split each input variable of the 5-dimensional space into, for stratified sampling.
    # divide the space into a rubik's cube so that there is one data point in each segment
    # actual number of data points is powers of 5 [32,243,1024,3125,7776]
    if pts == 100:
        segments = 2
    else:
        segments = round(pts ** (1 / 5))
    span = np.linspace(0, 1, num=segments + 1)
    # generate a stratified set of inlet streams within the 5 dimensional input space.
    for i, k, l, m, n in product(range(segments), repeat=5):
        # generate inlet
        T = np.random.uniform(span[i], span[i + 1])
        P = 0.5
        C2H4 = np.random.uniform(span[k], span[k + 1])
        O2 = np.random.uniform(span[l], span[l + 1])
        Q = np.random.uniform(span[m], span[m + 1])
        cat = np.random.uniform(span[n], span[n + 1])
        s = [T, P, C2H4, O2, 0, 0, 0]
        # calculate outlet
        result = UnitOp(Stream(*s), [Q, cat]).results()
        inlets.append([T, C2H4, O2, Q, cat])
        outlets.append(result)
    with open("part1/pickled_data/strat_inlets_{}.pkl".format(pts), "wb") as f:
        pickle.dump(inlets, f)
    with open("part1/pickled_data/strat_outlets_{}.pkl".format(pts), "wb") as f:
        pickle.dump(outlets, f)
    return (time.process_time() - t0) / pts * 100


def generate_data_latin_hypercube_sampling(pts):
    """latin hypercube sampling"""
    t0 = time.process_time()
    inlets = []
    outlets = []
    # generate a latin hypercube sampled set of inlet streams within the 5 dimensional input space.
    space = Space([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    lhs = Lhs(criterion="maximin", iterations=1000)
    lhs_points = lhs.generate(space.dimensions, pts)
    for i in range(pts):
        # generate inlet
        T, C2H4, O2, Q, cat = lhs_points[i]
        P = 0.5
        s = [T, P, C2H4, O2, 0, 0, 0]
        # calculate outlet
        result = UnitOp(Stream(*s), [Q, cat]).results()
        inlets.append([T, C2H4, O2, Q, cat])
        outlets.append(result)
    with open("part1/pickled_data/lhs_inlets_{}.pkl".format(pts), "wb") as f:
        pickle.dump(inlets, f)
    with open("part1/pickled_data/lhs_outlets_{}.pkl".format(pts), "wb") as f:
        pickle.dump(outlets, f)
    return (time.process_time() - t0) / pts * 100


def generate_training_data(pts_set):
    """generate the inlet-outlet training data pairs for different sampling techniques and different numbers of pts"""
    times = []
    for pts in pts_set:
        times.append(
            [
                generate_data_random_sampling(pts),
                generate_data_stratified_sampling(pts),
                generate_data_latin_hypercube_sampling(pts),
            ]
        )
    with open("part1/pickled_data/gen_times_{}.pkl".format(pts_set[-1]), "wb") as f:
        pickle.dump(times, f)


def generate_test_data(pts):
    """generate the inlet-outlet test data pairs for a given number of pts.
    the test data is randomly sampled and the same test data is used across all tests to make it fair."""
    inlets = []
    outlets = []
    for i in range(pts):
        # generate inlet
        T = np.random.uniform(0, 1)
        P = 0.5
        C2H4 = np.random.uniform(0, 1)
        O2 = np.random.uniform(0, 1)
        Q = np.random.uniform(0, 1)
        cat = np.random.uniform(0, 1)
        s = [T, P, C2H4, O2, 0, 0, 0]
        # calculate outlet
        result = UnitOp(Stream(*s), [Q, cat]).results()
        inlets.append([T, C2H4, O2, Q, cat])
        outlets.append(result)
    with open("part1/pickled_data/test_inlets.pkl", "wb") as f:
        pickle.dump(inlets, f)
    with open("part1/pickled_data/test_outlets.pkl", "wb") as f:
        pickle.dump(outlets, f)
