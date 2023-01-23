"""
Credit to my supervisor for the following physical models of an ethylene oxide flash drum.

Physical model for calculating the output streams of an ethylene oxide flash drum at chemical equilibrium.
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
    value,
    Any,
    Constraint,
    exp
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


def ethylene_oxide_flash_drum(s, vf, RF):
    m = ConcreteModel()
    m.props = RangeSet(10)
    m.components = RangeSet(6)
    m.cp_coef_set = RangeSet(6)
    m.psat_coef_set = RangeSet(3)
    m.he_coef_set = RangeSet(2)
    s = np.array(s) * slope + intcp
    m.inlet = Var(
        m.props,
        initialize=dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], s)),
        bounds=stream_bounds,
        within=Reals,
    )
    m.inlet.fix()
    m.psat_coef = Param(
        m.psat_coef_set,
        initialize=dict(zip([1, 2, 3], [11.21, 2354.731, 7.559])),
        within=Reals,
    )  # for water only
    m.Hvap_coef = Param(initialize=40660)  ## only for water, we assume others' Hvap=0
    m.constr_cnt = Param(initialize=1, within=Any, mutable=True)
    cp_df = pd.DataFrame(index=np.arange(1, 7), columns=np.arange(1, 7))
    ### cp coeff for: E, O, EO, CO2, W, N2
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
        -23.25802,
        275.6997,
        -188.9729,
        51.0335,
        0.38693,
        -55.09156 + 52.63514,
    ]
    cp_df.iloc[3, :] = [
        24.99735,
        55.18696,
        -33.69137,
        7.948387,
        -0.136638,
        -403.6075 + 393.5224,
    ]
    cp_df.iloc[4, :] = [
        -203.0606,
        1523.29,
        -3196.413,
        2474.455,
        3.855326,
        -256.5478 + 285.8304,
    ]
    cp_df.iloc[5, :] = [28.98641, 1.853978, -9.647459, 16.63537, 0.000117, -8.671914]
    cp_df = cp_df.stack().to_dict()
    m.cp_coef = Param(m.components, m.cp_coef_set, initialize=cp_df, within=Reals)
    henry_df = pd.DataFrame(index=np.arange(1, 7), columns=np.arange(1, 3))
    ### henry coeff for: E, O, EO, CO2, N2 (No Water!)
    henry_df.iloc[0, :] = [4.8e-5, 2000]
    henry_df.iloc[1, :] = [1.2e-5, 1500]
    henry_df.iloc[2, :] = [8.3e-2, 3200]
    henry_df.iloc[3, :] = [3.4e-4, 2400]
    henry_df.iloc[4, :] = [None, None]
    henry_df.iloc[5, :] = [6.4e-6, 1300]
    henry_df = henry_df.stack().to_dict()
    m.he_coef = Param(m.components, m.he_coef_set, initialize=henry_df, within=Reals)
    try:
        pred_o = RF.predict([[*s, vf]])[0]
        m.outletv = Var(
            m.props,
            initialize=dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], pred_o[10:])),
            bounds=stream_bounds,
            within=Reals,
        )
        m.outletl = Var(
            m.props,
            initialize=dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], pred_o[:10])),
            bounds=stream_bounds,
            within=Reals,
        )
    except:
        m.outletv = Var(
            m.props,
            initialize=dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], s)),
            bounds=stream_bounds,
            within=Reals,
        )
        m.outletl = Var(
            m.props,
            initialize=dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], s)),
            bounds=stream_bounds,
            within=Reals,
        )
    m.h = Var(
        m.components,
        initialize=dict(zip([1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0])),
        within=Reals,
    )
    m.q = Var(initialize=0, bounds=(-2, 4), within=Reals)
    m.vf = Var(initialize=vf, within=NonNegativeReals, bounds=(0, 1))
    m.vf.fix()

    def psat(T):
        return 10 ** (m.psat_coef[1] - (m.psat_coef[2] / (T + m.psat_coef[3])))

    def vf_eqn(m):
        return m.outletv[3] * m.outletv[8] / (m.inlet[3] * m.inlet[8]) == m.vf

    def henry(comp, T):
        if comp == 5:
            return 1 / ((-2.0998 * 10**9 + 7.8892 * 10**6 * T) * exp(240 / T))
        else:
            return (
                (m.he_coef[comp, 1] * exp(m.he_coef[comp, 2] * (1 / T - 1 / 298)))
                * (1.83)
                * (1 / 101325)
            )  ## Hcp* (Hxp/Hcp) * (atm/Pa)

    def totmol(m):
        return m.outletl[3] + m.outletv[3] == m.inlet[3]

    def masbal(m, comp):
        return (
            m.outletl[comp + 3] * m.outletl[3] + m.outletv[comp + 3] * m.outletv[3]
            == m.inlet[comp + 3] * m.inlet[3]
        )

    def h_i(m, comp):
        return (
            m.h[comp] / 1000
            == m.cp_coef[comp, 1] * (m.outletv[1]) / 1000
            + m.cp_coef[comp, 2] / 2 * ((m.outletv[1] / 1000) ** 2)
            + m.cp_coef[comp, 3] / 3 * (m.outletv[1] / 1000) ** 3
            + m.cp_coef[comp, 4] / 4 * (m.outletv[1] / 1000) ** 4
            - m.cp_coef[comp, 5] / (m.outletv[1] / 1000)
            + m.cp_coef[comp, 6]
        )

    def Hl(m):
        return m.outletl[10] == m.outletl[3] * sum(
            m.outletl[comp + 3] * m.h[comp] for comp in m.components
        )

    def Hv(m):
        return m.outletv[10] == m.outletv[3] * m.outletv[9] * m.Hvap_coef + m.outletv[
            3
        ] * sum(m.outletv[comp + 3] * m.h[comp] for comp in m.components)

    def Hbal(m):
        return m.inlet[10] + m.q * 1000000 == m.outletl[10] + m.outletv[10]

    def vlebal(m):
        return (
            sum(m.outletl[comp + 3] - m.outletv[comp + 3] for comp in m.components) == 0
        )

    def vle(m, comp):
        if comp == 5:  # Raoult's law for water
            return m.outletv[comp + 3] * m.outletv[2] == m.outletl[comp + 3] * psat(
                m.outletv[1]
            )  ## y * P = x * Psat
        else:  # Henry's law for other 5 components
            return (henry(comp, m.outletv[1])) * m.outletv[comp + 3] * m.outletv[
                2
            ] == m.outletl[
                comp + 3
            ]  ## Hxp * y * P = x

    def TT2(m):
        return m.outletv[1] == m.outletl[1]

    def PP1(m):
        return m.outletv[2] == m.inlet[2]

    def PP2(m):
        return m.outletl[2] == m.inlet[2]

    eqs = [
        ["", "vf_eqn"],
        ["", "totmol"],
        ["m.components,", "masbal"],
        ["m.components,", "h_i"],
        ["", "Hl"],
        ["", "Hv"],
        ["", "Hbal"],
        ["", "vlebal"],
        ["m.components,", "vle"],
        ["", "TT2"],
        ["", "PP1"],
        ["", "PP2"],
    ]
    for eq in eqs:
        const = compile(
            "m.constraint"
            + str(m.constr_cnt.value)
            + "=Constraint("
            + eq[0]
            + "expr="
            + eq[1]
            + ")",
            "<string>",
            "exec",
        )
        exec(const)
        m.constr_cnt.value += 1
    solver = SolverFactory("ipopt")
    try:
        solver.options = {"tol": 1e-5, "max_iter": 500}
        results = solver.solve(m, tee=False)
        if results.Solver.status == "ok":
            return [
                [value(m.outletl[i]) for i in m.props],
                [value(m.outletv[i]) for i in m.props],
            ]
        else:
            return None
    except:
        return None
