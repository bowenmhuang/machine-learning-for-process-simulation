'''
Generate test inlets and outlets to score the ML model.
'''

import time
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from generate_inlets import generate_inlet
from ethylene_oxide_flash_drum import ethylene_oxide_flash_drum


def generate_test_inlets(test_pts):
    random_inlets = []
    for i in range(test_pts):
        random_inlets.append(generate_inlet())
    with open("part2/pickled_data/test_inlets.pkl", "wb") as f:
        pickle.dump(random_inlets, f)


def generate_test_set(test_pts):
    t0 = time.process_time()
    with open("part2/pickled_data/test_inlets.pkl", "rb") as f:
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
    # Train a random forest on the initial results of the test set
    # Then run again in order to make the test set have a higher convergence percentage.
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
    with open("part2/pickled_data/test_inlets_after_RF.pkl", "wb") as f:
        pickle.dump(inlet_df, f)
    with open("part2/pickled_data/test_outlets_after_RF.pkl", "wb") as f:
        pickle.dump(outlet_df, f)
    return [len(initial_data), len(data), len(all_data), time_before]
