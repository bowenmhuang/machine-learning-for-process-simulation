import matplotlib as plt


def plot(pts_set):
    x = ["before", "after"]
    success_rates = [0.923, 0.989]
    x_pos = [0, 1.2]
    plt.bar(x_pos, success_rates, color=["orange", "red"])
    plt.ylabel("Success rate")
    plt.xticks(x_pos, x)
    plt.savefig(
        "EO flash bar chart success rate bounds_{}".format(pts_set[-1]),
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
        "EO flash bar chart scores bounds_{}".format(pts_set[-1]),
        bbox_inches="tight",
        dpi=200,
    )
