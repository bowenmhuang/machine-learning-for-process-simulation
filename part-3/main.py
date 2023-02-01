
from generate_data import generate_data, generate_test_data
from train import train_RF_obj, train_RF_successes, train_RF_both
from test import test_RF_obj, test_RF_successes
from plot import plot_corr_obj

def main():
    pts = 100
    data, objs, is_successes = generate_data(pts)

    RF, a = train_RF_obj(data, objs, pts)
    plot_corr_obj(pts, RF)
    RF = train_RF_successes(data, is_successes)
    RF = train_RF_both(data, objs, is_successes)

    test_pts = 1000
    data, objs, is_successes = generate_test_data(pts)
    b = test_RF_obj(RF, test_pts)
    test_RF_successes(RF, test_pts)
    print(a, b)

if __name__ == "__main__":
    main()
