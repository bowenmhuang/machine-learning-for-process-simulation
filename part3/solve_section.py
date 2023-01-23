'''
Solve the ethylene oxide purification section.
'''

import numpy as np

from ethylene_oxide_section import EthyleneOxideEnv


def solve_section(inlet):
    env = EthyleneOxideEnv()
    env.feed_method = "set"
    env.reset()
    print("feed ", np.round(env.state, 5))
    state, reward, done, _ = env.step([4, 7])  ##[HX, rxn, VLE, GLE, Add(O2), dP, sink]
    print("air", np.round(reward, 3), np.round(state, 5))
    state, reward, done, _ = env.step([0, 10])
    print("HX", np.round(reward, 3), np.round(state, 3))
    state, reward, done, _ = env.step([1, 10])
    print("PFR", np.round(reward, 3), np.round(state, 2))
    state, reward, done, _ = env.step([6, 4])
    print("sink", np.round(reward, 3), np.round(state, 2))
    print(env.key_path)

    HX, GLE, dP, VLE1, VLE2 = [*inlet]
    env.reset()
    env.sg = 1
    env.state[-1] = 1
    try:
        state, reward, done, _ = env.step([0, HX])
        print("HX", np.round(reward, 3), np.round(state, 3))
        state, reward, done, _ = env.step(
            [3, GLE]
        )  ##[HX, rxn, VLE, GLE, Add(O2), dP, sink]
        print("GLE", np.round(reward, 3), np.round(state, 3))
        state, reward, done, _ = env.step(
            [5, dP]
        )  ##[HX, rxn, VLE, GLE, Add(O2), dP, sink]
        print("dP", np.round(reward, 3), np.round(state, 3))
        state, reward, done, _ = env.step(
            [2, VLE1]
        )  ##[HX, rxn, VLE, GLE, Add(O2), dP, sink]
        print("VLE", np.round(reward, 3), np.round(state, 3))
        state, reward, done, _ = env.step(
            [2, VLE2]
        )  ##[HX, rxn, VLE, GLE, Add(O2), dP, sink]
        print("VLE", np.round(reward, 3), np.round(state, 3))
        state, reward, done, _ = env.step(
            [6, 4]
        )  ##[HX, rxn, VLE, GLE, Add(O2), dP, sink]
        print("sink", np.round(reward, 3), np.round(state, 3))
        print("##")
        env.reset()
        obj = env.recycle(
            12, env.pdf.iloc[7]["outprop"][1], "EO"
        )  ##recyc stream prop: pdf.iloc[-1]['outprop'][1]
        is_success = 1
    except:
        obj = None
        is_success = 0
    return obj, is_success
