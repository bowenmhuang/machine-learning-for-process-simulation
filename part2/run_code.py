import pickle

from generate_inlets import generate_inlets
from run_model import run_model, run_model_on_test
from train import train_random_forest


def run_code(pts_set):
    with open("final_test_inlets.pkl", "rb") as f:
        inlet_df = pickle.load(f)
    with open("final_test_outlets.pkl", "rb") as f:
        outlet_df = pickle.load(f)
    success_rates_after = []
    scores = []
    times = []
    for pts in pts_set:
        generate_inlets(pts)
        RF = None
        RF_sampling_time = run_model(pts, RF)
        RF_training_time = train_random_forest(pts)
        with open("EO_RF_{}.pkl".format(pts), "rb") as f:
            RF = pickle.load(f)
        a = run_model_on_test(RF)
        success_rates_after.append(a[0])
        times.append([a[1], RF_sampling_time, RF_training_time])
        scores.append(RF.score(inlet_df, outlet_df))
    with open("success_rates_after_{}.pkl".format(pts_set[-1]), "wb") as f:
        pickle.dump(success_rates_after, f)
    with open("scores_{}.pkl".format(pts_set[-1]), "wb") as f:
        pickle.dump(scores, f)
    with open("times_{}.pkl".format(pts_set[-1]), "wb") as f:
        pickle.dump(times, f)


# test_pts = 1000
# generate_test_inlets(test_pts)
# generate_test_set(test_pts)

# pts_set=[100,300,1000,3000,10000]
pts_set = [2]
run_code(pts_set)
# plot(pts_set)
# plot_scores(pts_set)
