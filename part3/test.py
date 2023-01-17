import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def test_RF_obj(RF, pts, ops):
    with open("test_inlets_{}_{}.pkl".format(pts, ops), "rb") as f:
        data = pickle.load(f)
    with open("test_outlets_{}_{}.pkl".format(pts, ops), "rb") as f:
        objs = pickle.load(f)
    all_data = pd.DataFrame(data)
    all_data[len(all_data.columns)] = objs
    all_data = all_data.dropna()
    # all_data=all_data.fillna(0)
    inlet_df = all_data.loc[:, 0:5]
    outlet_df = all_data[6]
    RF = RandomForestRegressor(n_estimators=50)  # n_estimators=50
    RF.fit(inlet_df, outlet_df.values.ravel())
    a = RF.score(inlet_df, outlet_df)
    return a


def test_RF_successes(RF, pts):
    with open("test_inlets_{}_{}.pkl".format(pts, ops), "rb") as f:
        data = pickle.load(f)
    with open("test_successes_{}_{}.pkl".format(pts, ops), "rb") as f:
        is_successes = pickle.load(f)
    inlet_df = pd.DataFrame(data)
    outlet_df = pd.DataFrame(is_successes)
    # RF = RandomForestRegressor(n_estimators=50)
    outlet_df = outlet_df.replace(to_replace=0, value="False")
    outlet_df = outlet_df.replace(to_replace=1, value="True")
    RF = RandomForestClassifier(n_estimators=50)  # n_estimators=50
    RF.fit(inlet_df, outlet_df.values.ravel())
    a = RF.score(inlet_df, outlet_df)
    return a
