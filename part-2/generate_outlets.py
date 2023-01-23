'''
Generate outlets by running the ethylene oxide flash drum model. If model does not converge, outlet is None.
'''

import time
import pickle

from ethylene_oxide_flash_drum import ethylene_oxide_flash_drum


def generate_outlets(pts, RF):
    t0 = time.process_time()
    with open("part2/pickled_data/random_inlets_{}.pkl".format(pts), "rb") as f:
        data = pickle.load(f)
    results = []
    for i in range(pts):
        random_inlet = data[i][0]
        vf = data[i][1]
        result = ethylene_oxide_flash_drum(random_inlet, vf, RF)
        results.append(result)
    with open("part2/pickled_data/results_{}.pkl".format(pts), "wb") as f:
        pickle.dump(results, f)
    return (time.process_time() - t0) / pts * 100
