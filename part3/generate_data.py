import pickle

import numpy as np
import pandas as pd
import seaborn as sns

import solve_section


def generate_inlet():
    HX = np.random.uniform(0, 2)
    GLE = np.random.uniform(2, 7)
    dP = np.random.uniform(0, 2)
    VLE1 = np.random.uniform(5, 10)
    VLE2 = np.random.uniform(0, 5)
    HX = 0
    return [HX, GLE, dP, VLE1, VLE2]


def generate_data(pts, ops):
    data = []
    objs = []
    is_successes = []
    successes = 0
    for i in range(pts):
        inlet = generate_inlet()
        data.append([float(i) for i in inlet])
        obj, is_success = solve_section(inlet, ops)
        objs.append(obj)
        is_successes.append(is_success)
        if is_success == 1:
            successes += 1
    if ops == 1:
        data_df = pd.DataFrame(data, columns=["HX", "GLE"])
    else:
        data_df = pd.DataFrame(data, columns=["HX", "GLE", "dP", "VLE1", "VLE2"])
    data_df["accuracy"] = is_successes
    sns.pairplot(data_df, kind="scatter", hue="accuracy")
    with open("training_inlets_{}_{}.pkl".format(pts, ops), "wb") as f:
        pickle.dump(data, f)
    with open("training_outlets_{}_{}.pkl".format(pts, ops), "wb") as f:
        pickle.dump(objs, f)
    with open("training_successes_{}_{}.pkl".format(pts, ops), "wb") as f:
        pickle.dump(is_successes, f)
    print(successes / pts)
    return data, objs, is_successes


def generate_test_data(pts, ops):
    data = []
    objs = []
    is_successes = []
    successes = 0
    for i in range(pts):
        inlet = generate_inlet()
        data.append([float(i) for i in inlet])
        obj, is_success = solve_section(inlet)
        objs.append(obj)
        is_successes.append(is_success)
        if is_success == 1:
            successes += 1
    with open("test_inlets_{}_{}.pkl".format(pts, ops), "wb") as f:
        pickle.dump(data, f)
    with open("test_outlets_{}_{}.pkl".format(pts, ops), "wb") as f:
        pickle.dump(objs, f)
    with open("test_successes_{}_{}.pkl".format(pts, ops), "wb") as f:
        pickle.dump(is_successes, f)
    print(successes / pts)
    return data, objs, is_successes
