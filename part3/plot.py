import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_corr(pts, ops):
    with open("training_inlets_{}_{}.pkl".format(pts, ops), "rb") as f:
        data = pickle.load(f)
    with open("training_successes_{}_{}.pkl".format(pts, ops), "rb") as f:
        is_successes = pickle.load(f)
    if ops == 1:
        data_df = pd.DataFrame(data, columns=["HX", "GLE"])
    else:
        data_df = pd.DataFrame(data, columns=["HX", "GLE", "dP", "VLE1", "VLE2"])
    data_df["accuracy"] = is_successes
    sns.pairplot(data_df, kind="scatter", hue="accuracy")


def plot_corr_test(pts):
    with open("test_inlets_{}.pkl".format(pts), "rb") as f:
        data = pickle.load(f)
    with open("test_successes_{}.pkl".format(pts), "rb") as f:
        is_successes = pickle.load(f)
    data_df = pd.DataFrame(data, columns=["HX", "GLE", "dP", "VLE1", "VLE2"])
    data_df["accuracy"] = is_successes
    sns.pairplot(data_df, kind="scatter", hue="accuracy")


def plot_corr_obj(pts, RF):
    with open("test_inlets_{}.pkl".format(pts), "rb") as f:
        data = pickle.load(f)
    with open("test_outlets_{}.pkl".format(pts), "rb") as f:
        objs = pickle.load(f)
    all_data = pd.DataFrame(data)
    all_data[len(all_data.columns)] = objs
    all_data = all_data.dropna()
    pred = RF.predict(all_data.iloc[:, :6])
    for i in [1]:
        plt.figure()
        plt.scatter(all_data.iloc[:, i], all_data[6], label="True value")
        plt.scatter(all_data.iloc[:, i], pred, label="RF prediction")
        plt.legend()
        plt.xlabel("input variable {}".format(1))
        plt.ylabel("$log_{10}$(objective)")
        plt.ylim(-0.1, 0.05)
        plt.savefig("corr obj {}.png".format(i), bbox_inches="tight", dpi=200)
    # data_df = pd.DataFrame(data,columns=['HX', 'GLE', 'dP', 'VLE1', 'VLE2', 'sink'])
    # data_df['accuracy']=is_successes


def plot_obj(pts_set):
    x = ["before", "after"]
    success_rates = [0.923, 0.989]
    x_pos = [0, 1.2]
    plt.bar(x_pos, success_rates, color=["orange", "red"])
    plt.ylabel("Success rate")
    plt.xticks(x_pos, x)
    plt.savefig("EO section_{}".format(pts_set[-1]), bbox_inches="tight", dpi=200)


def plot_succ(pts_set):
    x = ["before", "after"]
    success_rates = [0.415, 0.475]
    x_pos = [0, 1.2]
    plt.bar(x_pos, success_rates, color=["orange", "red"])
    plt.ylabel("RF test score")
    plt.xticks(x_pos, x)
    plt.savefig(
        "EO flash bar chart scores bounds_{}".format(pts_set[-1]),
        bbox_inches="tight",
        dpi=200,
    )


plot(pts_set)
plot_succ(pts_set)
