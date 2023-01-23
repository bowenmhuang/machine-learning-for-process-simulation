import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def train_RF_obj(data, objs, pts):
    with open("training_inlets_{}.pkl".format(pts), "rb") as f:
        data = pickle.load(f)
    with open("training_outlets_{}.pkl".format(pts), "rb") as f:
        objs = pickle.load(f)
    all_data = pd.DataFrame(data)
    all_data[len(all_data.columns)] = objs
    all_data = all_data.dropna()
    inlet_df = all_data.loc[:, 0:5]
    outlet_df = all_data[6]
    RF = RandomForestRegressor(n_estimators=50)
    RF.fit(inlet_df, outlet_df.values.ravel())
    a = RF.score(inlet_df, outlet_df)
    return RF, a


def train_RF_successes(data, is_successes, pts):
    with open("training_inlets_{}.pkl".format(pts), "rb") as f:
        data = pickle.load(f)
    with open("training_successes_{}.pkl".format(pts), "rb") as f:
        is_successes = pickle.load(f)
    inlet_df = pd.DataFrame(data)
    outlet_df = pd.DataFrame(is_successes)
    outlet_df = outlet_df.replace(to_replace=0, value="False")
    outlet_df = outlet_df.replace(to_replace=1, value="True")
    RF = RandomForestClassifier(n_estimators=50)
    RF.fit(inlet_df, outlet_df.values.ravel())
    a = RF.score(inlet_df, outlet_df)
    return RF, a
