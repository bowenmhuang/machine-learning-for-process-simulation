import time
import pickle

import pandas as pd

from ethylene_oxide_flash_drum import ethylene_oxide_flash_drum 

def calculate_success_rate_on_test(RF):
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
    # sns.pairplot(data_df, kind="scatter", hue="is_success")
    # plt.show()
    time_after = (time.process_time() - t0) / len(all_data) * 100
    return [(len(all_data) - fails) / len(all_data), time_after]