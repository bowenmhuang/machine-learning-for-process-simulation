from generate_inlets import generate_test_inlets, generate_test_set
from generate_outlets import generate_outlets_before_and_after_ML_prediction
from train_random_forest import train_random_forest
from plot import plot_test_outlet_successes, plot_success_rates, plot_scores

def run_code():
    test_pts = 1000
    generate_test_inlets(test_pts)
    generate_test_set(test_pts)

    pts_set = [10, 30, 100, 300, 1000]
    run_model_before_and_after_ML_prediction(pts_set)
    plot_success_rates(pts_set)
    plot_scores(pts_set)

run_code()