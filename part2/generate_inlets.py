import time
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from find_enthalpy import find_enthalpy
from ethylene_oxide_flash_drum import ethylene_oxide_flash_drum


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
    mole_fracs = [molar_flows[i] / molar_flows(molar_flows[:]) for i in range(6)]
    stream = [T, P, n, *mole_fracs]
    H = find_enthalpy(stream)
    stream.append(H)
    vf = np.random.uniform(0, 1)
    return [stream, vf]


def generate_inlets(pts):
    random_inlets = []
    for i in range(pts):
        random_inlets.append(generate_inlet())
    with open("random_inlets_{}.pkl".format(pts), "wb") as f:
        pickle.dump(random_inlets, f)


def generate_test_inlets(test_pts):
    random_inlets = []
    for i in range(test_pts):
        random_inlets.append(generate_inlet())
    with open("test_inlets.pkl", "wb") as f:
        pickle.dump(random_inlets, f)


def generate_test_set(test_pts):
    t0 = time.process_time()
    with open("test_inlets.pkl", "rb") as f:
        all_data = pickle.load(f)
    initial_data = []
    initial_results = []
    data = []
    results = []
    RF = None
    for i in range(len(all_data)):
        s = all_data[i][0]
        vf = all_data[i][1]
        result = ethylene_oxide_flash_drum(s, vf, RF)
        if not result == None:
            initial_data.append([*s, vf])
            initial_results.append([*result[0], *result[1]])
    time_before = (time.process_time() - t0) / test_pts * 100
    inlet_df = pd.DataFrame(initial_data)
    outlet_df = pd.DataFrame(initial_results)
    RF = RandomForestRegressor()
    RF.fit(inlet_df, outlet_df)
    for i in range(test_pts):
        s = all_data[i][0]
        vf = all_data[i][1]
        result = ethylene_oxide_flash_drum(s, vf, RF)
        if not result == None:
            data.append([*s, vf])
            results.append([*result[0], *result[1]])
    inlet_df = pd.DataFrame(data)
    outlet_df = pd.DataFrame(results)
    with open("final_test_inlets.pkl", "wb") as f:
        pickle.dump(inlet_df, f)
    with open("final_test_outlets.pkl", "wb") as f:
        pickle.dump(outlet_df, f)
    return [len(initial_data), len(data), len(all_data), time_before]
