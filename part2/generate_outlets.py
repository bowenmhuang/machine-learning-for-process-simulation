import time
import pickle

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

from ethylene_oxide_flash_drum import ethylene_oxide_flash_drum


def run_model(pts, RF):
    t0 = time.process_time()
    with open("random_inlets_{}.pkl".format(pts), "rb") as f:
        data = pickle.load(f)
    results = []
    for i in range(pts):
        random_inlet = data[i][0]
        vf = data[i][1]
        result = ethylene_oxide_flash_drum(random_inlet, vf, RF)
        results.append(result)
    with open("results_{}.pkl".format(pts), "wb") as f:
        pickle.dump(results, f)
    return (time.process_time() - t0) / pts * 100


def run_model_on_test(RF):
    t0 = time.process_time()
    with open("test_inlets.pkl", "rb") as f:
        all_data = pickle.load(f)
    results = []
    fails = 0
    is_success = []
    data_df = []
    for i in range(len(all_data)):
        random_inlet = all_data[i][0]
        vf = all_data[i][1]
        data_df.append([*random_inlet, vf])
        result = ethylene_oxide_flash_drum(random_inlet, vf, RF)
        results.append(result)
        if result == None:
            fails += 1
            is_success.append(False)
        else:
            is_success.append(True)
    data_df = pd.DataFrame(
        data_df, columns=["Ti", "P", "n", "z1", "z2", "z3", "z4", "z5", "z6", "h", "vf"]
    )
    data_df["is_success"] = is_success
    sns.pairplot(data_df, kind="scatter", hue="is_success")
    plt.show()
    time_after = (time.process_time() - t0) / len(all_data) * 100
    return [(len(all_data) - fails) / len(all_data), time_after]
