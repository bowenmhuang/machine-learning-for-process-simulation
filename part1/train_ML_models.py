"""
These functions train and test different ML models (random forest regression, Gaussian process regression, deep neural networks)
for different sampling techniques of the input space: random, stratified, latin hypercube.

train_algs calculates the scores and times for each ML model / sampling technique combination, for different numbers of training pts.
"""

import time
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from keras.layers import Dense
from keras.models import Sequential


def train_and_test_random_forest_regressor(sampling_technique, pts):
    # random forest regression
    t0 = time.process_time()

    # get the data
    inlets, outlets, test_inlets, test_outlets = unpickle_data(sampling_technique, pts)
    inlet_df = pd.DataFrame(inlets)
    outlet_df = pd.DataFrame(outlets)
    test_inlet_df = pd.DataFrame(test_inlets)
    test_outlet_df = pd.DataFrame(test_outlets)

    # define hyperparameters of ML model
    random_forest_regressor = RandomForestRegressor(n_estimators=50)
    # train the ML model
    random_forest_regressor.fit(inlet_df, outlet_df)
    # calculate the test score and time taken
    score = random_forest_regressor.score(test_inlet_df, test_outlet_df)
    time_taken = (time.process_time() - t0) / pts * 100

    return [score, time_taken]


def train_and_test_gaussian_process_regressor(sampling_technique, pts):
    # Gaussian process regression

    t0 = time.process_time()

    # get the data
    inlets, outlets, test_inlets, test_outlets = unpickle_data(sampling_technique, pts)
    inlet_df = pd.DataFrame(inlets)
    outlet_df = pd.DataFrame(outlets)
    test_inlet_df = pd.DataFrame(test_inlets)
    test_outlet_df = pd.DataFrame(test_outlets)

    # define hyperparameters of ML model
    kernel = C(1e-5) * RBF() + RBF(0.1) + WhiteKernel()
    # train the ML model
    gaussian_process_regressor = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3
    )
    gaussian_process_regressor.fit(inlet_df, outlet_df)
    # calculate the test score and time taken
    score = gaussian_process_regressor.score(test_inlet_df, test_outlet_df)
    time_taken = (time.process_time() - t0) / pts * 100

    return [score, time_taken]


def train_and_test_neural_network(sampling_technique, pts):
    # deep neural networks

    t0 = time.process_time()

    # get the data
    inlets, outlets, test_inlets, test_outlets = unpickle_data(sampling_technique, pts)
    # neural networks require normalized inputs
    sc = StandardScaler()
    inlet_df = sc.fit_transform(inlets)
    test_inlet_df = sc.transform(test_inlets)
    outlet_df = sc.fit_transform(outlets)
    test_outlet_df = sc.transform(test_outlets)

    # define hyperparameters of the ML model
    model = Sequential()
    model.add(Dense(units=10, activation="relu", input_dim=5))
    model.add(Dense(units=10, activation="relu"))
    model.add(Dense(units=10, activation="relu"))
    model.add(Dense(units=10, activation="relu"))
    model.add(Dense(units=10, activation="relu"))
    model.add(Dense(units=6))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
    # train the ML model
    NN = model.fit(inlet_df, outlet_df, batch_size=50, epochs=20, validation_split=0.1)
    # calculate the test score and time taken
    pred_outlet_df = model.predict(test_inlet_df)
    score = r2_score(test_outlet_df, pred_outlet_df)
    time_taken = (time.process_time() - t0) / pts * 100

    return [score, time_taken]


def train_and_test_models(pts_set):
    """trains and tests the ML models (RF,GP,NN) for different sampling techniques (random,strat,lhs) and number of training pts
    pickles the scores and times for each combination of model/sampling/no. of pts
    """
    scores = np.empty([3, 3, len(pts_set)])
    times = np.empty([3, 3, len(pts_set)])
    for i, method in enumerate(["random", "strat", "lhs"]):
        for k, pts in enumerate(pts_set):
            RFscores, RFtimes = train_and_test_random_forest_regressor(method, pts)
            GPscores, GPtimes = train_and_test_gaussian_process_regressor(method, pts)
            NNscores, NNtimes = train_and_test_neural_network(method, pts)
            scores[0][i][k], scores[1][i][k], scores[2][i][k] = (
                RFscores,
                GPscores,
                NNscores,
            )
            times[0][i][k], times[1][i][k], times[2][i][k] = RFtimes, GPtimes, NNtimes
    with open("pickled_data/scores_{}.pkl".format(pts_set[-1]), "wb") as f:
        pickle.dump(scores, f)
    with open("pickled_data/times_{}.pkl".format(pts_set[-1]), "wb") as f:
        pickle.dump(times, f)


def unpickle_data(sampling_technique, pts):
    # unpickles the training and test data for the ML training and testing functions
    with open(
        "pickled_data/{}_inlets_{}.pkl".format(sampling_technique, pts), "rb"
    ) as f:
        inlets = pickle.load(f)
    with open(
        "pickled_data/{}_outlets_{}.pkl".format(sampling_technique, pts), "rb"
    ) as f:
        outlets = pickle.load(f)
    with open("pickled_data/test_inlets.pkl", "rb") as f:
        test_inlets = pickle.load(f)
    with open("pickled_data/test_outlets.pkl", "rb") as f:
        test_outlets = pickle.load(f)
    return inlets, outlets, test_inlets, test_outlets
