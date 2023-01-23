import pickle

from generate_inlets import generate_inlets
from generate_outlets import generate_outlets
from train_random_forest import train_random_forest
from calculate_success_rate_on_test import calculate_success_rate_on_test

def generate_outlets_before_and_after_ML_prediction(pts_set):
    with open("part2/pickled_data/test_inlets_after_RF.pkl", "rb") as f:
        inlet_df = pickle.load(f)
    with open("part2/pickled_data/test_outlets_after_RF.pkl", "rb") as f:
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
        a = calculate_success_rate_on_test(RF)
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
