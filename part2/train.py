import time
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_random_forest(pts):
    t0 = time.process_time()
    with open("random_inlets_{}.pkl".format(pts), "rb") as f:
        data = pickle.load(f)
    with open("results_{}.pkl".format(pts), "rb") as f:
        results = pickle.load(f)
    initial_data = []
    initial_results = []
    for i in range(len(results)):
        if not results[i] == None:
            random_inlet = data[i][0]
            vf = data[i][1]
            result = [*results[i][0], *results[i][1]]
            initial_results.append(result)
            initial_data.append([*random_inlet, vf])
    inlet_df = pd.DataFrame(initial_data)
    outlet_df = pd.DataFrame(initial_results)
    RF = RandomForestRegressor(n_estimators=50)  # n_estimators=50
    RF.fit(inlet_df, outlet_df)
    with open("EO_RF_{}.pkl".format(pts), "wb") as f:
        pickle.dump(RF, f)
    return (time.process_time() - t0) / pts * 100
