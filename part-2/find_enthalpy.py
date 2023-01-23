"""
Credit to my supervisor for the following physical models of an ethylene oxide flash drum.

Physical model for finding the enthalpy of a multi-component stream.
"""

import numpy as np
import pandas as pd

from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Param,
    Reals,
    NonNegativeReals,
    Var,
    Constraint,
)
from pyomo.opt import SolverFactory

# 10 elements: T, P, n, E, O, EO, CO2, W, N2, H
slope = np.array(
    [232, 1.9e6, 100, 0.99994, 0.99994, 0.99994, 0.99994, 0.99994, 0.99994, 1.001e7]
)
intcp = np.array([298, 1e5, 0, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, -1e4])


def stream_bounds(m, i):
    lb = {
        1: 298,
        2: 1e5,
        3: 0,
        4: 1e-5,
        5: 1e-5,
        6: 1e-5,
        7: 1e-5,
        8: 1e-5,
        9: 1e-5,
        10: -1e4,
    }
    ub = {
        1: 530,
        2: 2e6,
        3: 100,
        4: 0.99995,
        5: 0.99995,
        6: 0.99995,
        7: 0.99995,
        8: 0.99995,
        9: 0.99995,
        10: 1e7,
    }
    return (lb[i], ub[i])


def find_enthalpy(stream):
    # takes s with values between 0 and 1,and finds H which is mainly a function of H_vap of water
    m = ConcreteModel()
    m.props = RangeSet(10)
    m.components = RangeSet(6)
    m.psat_coef_set = RangeSet(6)
    stream = np.array(stream) * slope[:-1] + intcp[:-1]
    m.psat_coef = Param(
        m.psat_coef_set,
        initialize=dict(zip([1, 2, 3], [11.21, 2345.731, 7.559])),
        within=Reals,
    )
    cp_df = pd.DataFrame(index=np.arange(1, 7), columns=np.arange(1, 7))
    # cp coeff for: E, O, EO, CO2, W, N2
    cp_df.iloc[0, :] = [
        -6.38788,
        184.4019,
        -112.9718,
        28.49593,
        0.31554,
        48.17332 - 52.46694,
    ]
    cp_df.iloc[1, :] = [31.32234, -20.23531, 57.86644, -36.50624, -0.007374, -8.903471]
    cp_df.iloc[2, :] = [
        -23.22302,
        275.6997,
        -188.9729,
        51.0335,
        0.38693,
        -55.09156 + 52.63514,
    ]
    cp_df.iloc[3, :] = [
        25.0275,
        55.18696,
        -33.69137,
        7.948387,
        -0.136638,
        -403.6075 + 393.5224,
    ]
    cp_df.iloc[4, :] = [
        -203.5699,
        1523.29,
        -3196.413,
        2474.455,
        3.855326,
        -256.5478 + 285.8304,
    ]
    cp_df.iloc[5, :] = [29.00202, 1.853978, -9.647459, 16.63537, 0.000117, -8.671914]
    cp_df = cp_df.stack().to_dict()

    m.cp_coef = Param(m.components, m.psat_coef_set, initialize=cp_df, within=Reals)
    m.Hvap_coef = Param(initialize=40660)  ## only for water, we assume others' Hvap=0
    m.T = Param(initialize=stream[0])
    m.P = Param(initialize=stream[1])
    m.n = Param(initialize=stream[2])
    m.z = Param(
        m.components,
        initialize={
            1: stream[3],
            2: stream[4],
            3: stream[5],
            4: stream[6],
            5: stream[7],
            6: stream[8],
        },
    )
    m.vley = Var(initialize=0, bounds=(0, 0.99999), within=NonNegativeReals)
    m.h = Var(
        m.components,
        initialize=dict(zip([1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0])),
        within=Reals,
    )  ## components enthalpy
    m.H = Var(
        initialize=0, bounds=(intcp[-1], slope[-1] + intcp[-1]), within=Reals
    )  ## stream enthalpy

    def psat(T):
        return 10 ** (m.psat_coef[1] - (m.psat_coef[2] / (T + m.psat_coef[3])))

    def miny(T):
        return 0.5 * (
            psat(m.T) / stream[1]
            + m.z[5]
            - ((psat(m.T) / stream[1] - m.z[5]) ** 2 + 0.1**2) ** 0.5
        )

    def antoine(m):
        return m.vley == 0.5 * (miny(m.T) + (miny(m.T) ** 2 + 0.00005**2) ** 0.5)

    def h_i(m, comp):
        return (
            m.h[comp] / 1000
            == m.cp_coef[comp, 1] * (m.T) / 1000
            + m.cp_coef[comp, 2] / 2 * ((m.T / 1000) ** 2)
            + m.cp_coef[comp, 3] / 3 * (m.T / 1000) ** 3
            + m.cp_coef[comp, 4] / 4 * (m.T / 1000) ** 4
            - m.cp_coef[comp, 5] / (m.T / 1000)
            + m.cp_coef[comp, 6]
        )

    def Hbal(m):
        return m.H == m.n * m.vley * m.Hvap_coef + m.n * sum(
            m.z[comp] * m.h[comp] for comp in m.components
        )

    m.c1 = Constraint(expr=antoine)
    m.c2 = Constraint(m.components, expr=h_i)
    m.c3 = Constraint(expr=Hbal)
    solver = SolverFactory("ipopt")
    solver.options = {"tol": 1e-8, "max_iter": 100}
    results = solver.solve(m, tee=False)
    if results.Solver.status == "ok":
        return (m.H.value - intcp[-1]) / slope[-1]
    else:
        print("results.Solver.status is not ok")
        return 0
