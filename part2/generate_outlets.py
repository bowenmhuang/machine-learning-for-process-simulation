import time
import pickle

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

from ethylene_oxide_flash_drum import ethylene_oxide_flash_drum


def generate_outlets(pts, RF):
    t0 = time.process_time()
    with open("part2/pickled_data/random_inlets_{}.pkl".format(pts), "rb") as f:
        data = pickle.load(f)
    results = []
    for i in range(pts):
        random_inlet = data[i][0]
        vf = data[i][1]
        result = ethylene_oxide_flash_drum(random_inlet, vf, RF)
        results.append(result)
    with open("part2/pickled_data/results_{}.pkl".format(pts), "wb") as f:
        pickle.dump(results, f)
    return (time.process_time() - t0) / pts * 100


def generate_outlets_before_and_after_ML_prediction(pts_set):
    with open("part2/pickled_data/test_inlets.pkl", "rb") as f:
        inlet_df = pickle.load(f)
    with open("part2/pickled_data/test_outlets.pkl", "rb") as f:
        outlet_df = pickle.load(f)
    success_rates_after = []
    scores = []
    times = []
    for pts in pts_set:
        generate_inlets(pts)
        RF = None
        RF_sampling_time = generate_outlets(pts, RF)
        RF_training_time = train_random_forest(pts)
        with open("part2/pickled_data/EO_RF_{}.pkl".format(pts), "rb") as f:
            RF = pickle.load(f)
        a = run_model_on_test(RF)
        success_rates_after.append(a[0])
        times.append([a[1], RF_sampling_time, RF_training_time])
        scores.append(RF.score(inlet_df, outlet_df))
    with open(
        "part2/pickled_data/success_rates_after_{}.pkl".format(pts_set[-1]), "wb"
    ) as f:
        pickle.dump(success_rates_after, f)
    with open("part2/pickled_data/scores_{}.pkl".format(pts_set[-1]), "wb") as f:
        pickle.dump(scores, f)
    with open("part2/pickled_data/times_{}.pkl".format(pts_set[-1]), "wb") as f:
        pickle.dump(times, f)