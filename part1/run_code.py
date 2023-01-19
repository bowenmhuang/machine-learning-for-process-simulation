"""
This is where the code is executed.

Aim: to investigate the accuracy and speed of surrogate modelling an ethylene oxide reactor using different ML models and
different sampling techniques to generate the training data.

Workflow: 
1) generate inlets within the relevant operating bounds, through different sampling techniques
2) calculate corresponding outlets using ethylene_oxide_reactor_model
3) train different ML models on the inlet-outlet pairs (training data)
4) plot the accuracy and speed of the ML models.

Output: Graphs showing the R^2 test scores and CPU times for different combinations of ML model / sampling technique / number of pts.
"""

from generate_data import generate_test_data, generate_training_data
from train_and_test_ML_models import train_and_test_ML_models
from plot import plot_scores_and_times


def run_code():
    # generate test data. calculates inlet-outlet pairs using physical_model. test data is generated once, via random sampling, for all tests.
    test_pts = 1000
    generate_test_data(test_pts)

    # generate training data. calculates inlet-outlet pairs using physical_model, for different inlet sampling techniques.
    pts_set = [100, 300, 1000, 3000, 10000]
    generate_training_data(pts_set)

    # calculates test score and CPU times for different ML models and sampling techniques
    train_and_test_ML_models(pts_set)

    # plots the results from train_and_test_ML_models
    plot_scores_and_times(pts_set)


run_code()
