from generate_test_set import generate_test_inlets, generate_test_set
from generate_outlets_before_and_after import generate_outlets_before_and_after_ML_prediction


def run_code():
    test_pts = 1000
    generate_test_inlets(test_pts)
    generate_test_set(test_pts)

    pts_set = [10, 30, 100, 300, 1000]
    generate_outlets_before_and_after_ML_prediction(pts_set)

run_code()