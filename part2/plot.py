import matplotlib as plt


def plot_test_outlet_successes(RF):
    t0 = time.process_time()
    with open("part2/pickled_data/test_inlets.pkl", "rb") as f:
        all_data = pickle.load(f)
    results = []
    fails = 0
    is_success = []
    data_df = []
    for i in range(len(all_data)):
        random_inlet = all_data[i][0]
        vf = all_data[i][1]
        data_df.append([*random_inlet, vf])
        result = ethylene_oxide_flash_drum(random_inlet, vf, RF)
        results.append(result)
        if result == None:
            fails += 1
            is_success.append(False)
        else:
            is_success.append(True)
    data_df = pd.DataFrame(
        data_df, columns=["Ti", "P", "n", "z1", "z2", "z3", "z4", "z5", "z6", "h", "vf"]
    )
    data_df["is_success"] = is_success
    sns.pairplot(data_df, kind="scatter", hue="is_success")
    plt.show()
    time_after = (time.process_time() - t0) / len(all_data) * 100
    return [(len(all_data) - fails) / len(all_data), time_after]


def plot_success_rates(pts_set):
    x = ["before", "after"]
    success_rates = [0.923, 0.989]
    x_pos = [0, 1.2]
    plt.bar(x_pos, success_rates, color=["orange", "red"])
    plt.ylabel("Success rate")
    plt.xticks(x_pos, x)
    plt.savefig(
        "part2/plots/EO flash bar chart success rate bounds_{}".format(pts_set[-1]),
        bbox_inches="tight",
        dpi=200,
    )


def plot_scores(pts_set):
    x = ["before", "after"]
    success_rates = [0.415, 0.475]
    x_pos = [0, 1.2]
    plt.bar(x_pos, success_rates, color=["orange", "red"])
    plt.ylabel("RF test score")
    plt.xticks(x_pos, x)
    plt.savefig(
        "part2/plots/EO flash bar chart scores bounds_{}".format(pts_set[-1]),
        bbox_inches="tight",
        dpi=200,
    )
