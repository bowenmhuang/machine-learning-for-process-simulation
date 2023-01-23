"""
Credit to my supervisor for the following physical models of various chemical processes.

Here, we build a 'lego' approach to unit (contraint) de/activation.
Pyomo models for the unit blocks including:
heat exchanger (HX), mixing (mixer), reaction (rxn), liquid liquid extraction (LLE),
flash drum (flash), valve (dP), and recycle
"""

import time
import logging

import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.dae import *


logging.getLogger("pyomo.core").setLevel(logging.CRITICAL)


s = [0, 1, 0.025, 0.94996, 1e-5, 1e-5, 0.05, 1e-5, 1e-5]
slope = np.array([232, 1.9e6, 200, 1, 1, 1, 1, 1, 1, 1.01e7])
intcp = np.array([298, 1e5, 0, 0, 0, 0, 0, 0, 0, -1e5])
# feed =[298,1e6,5,0.9995,1e-5,1e-5,1e-5,1e-5,1e-5,-43.094768]
empty = [298, 1e6, 0, 0.16, 0.16, 0.16, 0.16, 0.16, 0.17, 0]
# fdval={}
emptval = {}
for i in range(len(empty)):
    # fdval[i+1]= feed[i]
    emptval[i + 1] = empty[i]


def strm_bnds(m, i):
    lb = {1: 298, 2: 1e5, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: -1e5}
    ub = {1: 530, 2: 2e6, 3: 200, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1e7}
    return (lb[i], ub[i])


def air_bnds(m, i):
    lb = {1: 298, 2: 1e5, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: -1e5}
    ub = {1: 530, 2: 2e6, 3: 100, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1e7}
    return (lb[i], ub[i])


def h_bnds(m, i):  ## compnent enthalpy bounds
    lb = {1: -2e5, 2: -2e5, 3: -2e5, 4: -2e5, 5: -2e5, 6: -2e5}
    ub = {1: 2e5, 2: 2e5, 3: 2e5, 4: 2e5, 5: 2e5, 6: 2e5}
    return (lb[i], ub[i])


def hpfr_bnds(m, i, t):  ## compnent enthalpy bounds
    lb = {1: -2e5, 2: -2e5, 3: -2e5, 4: -2e5, 5: -2e5, 6: -2e5}
    ub = {1: 2e5, 2: 2e5, 3: 2e5, 4: 2e5, 5: 2e5, 6: 2e5}
    return (lb[i], ub[i])


def start_episode():
    m = ConcreteModel()
    m.props = RangeSet(10)
    m.components = RangeSet(6)
    m.cp_coef_set = RangeSet(6)
    m.psat_coef_set = RangeSet(3)
    m.he_coef_set = RangeSet(2)
    m.max_var = RangeSet(50)
    m.two_set = RangeSet(2)
    m.rxns = RangeSet(2)
    ##counters
    m.old_constr_cnt = Param(initialize=1, within=Any, mutable=True)
    m.constr_cnt = Param(initialize=1, within=Any, mutable=True)  ### goes to env
    m.stream_cnt = Param(initialize=1, within=Any, mutable=True)  ### goes to env
    m.control_cnt = Param(initialize=1, mutable=True)
    m.control_dict = Param(m.max_var, within=Any, mutable=True)
    m.cost_cnt = Param(initialize=1, mutable=True)
    m.heat_constr = Var(
        m.max_var,
        within=Any,
        initialize=dict(
            zip(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
        ),
    )
    m.heat_constr_cnt = Param(initialize=1, mutable=True)
    m.cost_dict = Var(
        m.max_var,
        initialize=dict(
            zip(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
        ),
        within=NonNegativeReals,
    )
    ##parameters
    m.R = Param(initialize=8.314)
    m.psat_coef = Param(
        m.psat_coef_set,
        initialize=dict(zip([1, 2, 3], [11.21, 2345.731, 7.559])),
        within=Reals,
    )  # for water only
    m.Hvap_coef = Param(initialize=40660)  ## only for water, we assume others' Hvap=0
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
    nu = pd.DataFrame(index=np.arange(1, 3), columns=np.arange(1, 7))
    nu.iloc[0, :] = [-1, -0.5, 1, 0, 0, 0]
    nu.iloc[1, :] = [-1, -3, 0, 2, 2, 0]
    nu = nu.stack().to_dict()
    m.nu = Param(m.rxns, m.components, initialize=nu)
    m.eps = Param(initialize=0.4)
    m.rho_cat = Param(initialize=630)  ##kg/m3
    m.k0 = Param(
        m.rxns, initialize={1: 6.275e6, 2: 1.206e7}
    )  ##mol/(kg_cat * Pa^m) where m=[1.1,1]
    m.EA0 = Param(m.rxns, initialize={1: 74900, 2: 89800})  ##kj/mol
    m.K0 = Param(m.rxns, initialize={1: 1.985e2, 2: 1.08e2})  ## Pa^-1
    m.Tads = Param(m.rxns, initialize={1: 2400, 2: 1530})  ### K
    m.exp_e = Param(m.rxns, initialize={1: 0.6, 2: 0.5})
    m.exp_o = Param(m.rxns, initialize={1: 0.5, 2: 0.5})
    m.j_coef = Param(m.components, initialize={1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0})
    # m.q_cost=Param(initialize=0.0028) ## 2p/kWh = £0.0056/Mj
    # m.Pw_cost=Param(initialize=1e-5)  ## 2p/kWh = £2e-5/W
    # m.Et_cost=Param(initialize=0.03) ## $1000/metric_ton = £0.02/mol
    # m.EO_cost=Param(initialize=0.0388)## $1000/metric_ton = £0.0388/mol
    # m.W_cost=Param(initialize=7.74e-5)## $4.3/m3 = £7.74e-5/mol
    m.q_cost = Param(initialize=0.0028)  ## 2p/kWh = £0.0056/Mj
    m.Pw_cost = Param(initialize=1e-5)  ## 2p/kWh = £2e-5/Mj
    # m.Et_cost=Param(initialize=0.02) ## $1000/metric_ton = £0.02/mol
    m.EO_cost = Param(initialize=0.0488)  ## $1000/metric_ton = £0.0388/mol
    # m.W_cost=Param(initialize=7.74e-5)## $4.3/m3 = £7.74e-5/mol
    return m


def PFR(m, inlet, outlet, res_time):
    m.old_constr_cnt.value = m.constr_cnt.value
    m.t = ContinuousSet(bounds=(0.0, res_time))
    m.T = Var(m.t, bounds=(400, 530))
    m.P = Var(m.t, bounds=(1e5, 2e6))
    m.n = Var(m.components, m.t, bounds=(0, 200))
    m.r = Var(m.rxns, m.t)  ##mol/kg_cat-s
    m.q = Var(m.t, bounds=(-0.2, 0.2))  ##Mj/s-m2
    m.Vg = Var(m.t, bounds=(0, 10))
    m.h = Var(m.components, m.t, within=Reals, bounds=hpfr_bnds)
    m.cp = Var(m.components, m.t, within=NonNegativeReals)
    m.jE = Var(m.t, bounds=(0, 1))  ## upper bound affects convergence very much.
    m.dn = DerivativeVar(m.n, wrt=m.t)
    m.dT = DerivativeVar(m.T, wrt=m.t)
    m.dP = DerivativeVar(m.P, wrt=m.t)

    def inT(m):
        return m.T[0] == inlet[1]

    def inP(m):
        return m.P[0] == inlet[2]

    def inn(m, comp):
        return m.n[comp, 0] == inlet[3] * inlet[3 + comp]

    def molbal(m, comp, t):
        return m.dn[comp, t] == m.j_coef[comp] * m.jE[t] + m.rho_cat * (
            1 - m.eps
        ) / m.eps * m.Vg[t] * sum(m.nu[rxn, comp] * m.r[rxn, t] for rxn in m.rxns)

    def ig(m, t):
        return m.Vg[t] == sum(m.n[comp, t] for comp in m.components) * m.R * m.T[t] / (
            m.P[t]
        )

    def h_i(m, comp, t):
        return (
            m.h[comp, t] / 1000
            == m.cp_coef[comp, 1] * (m.T[t] / 1000)
            + m.cp_coef[comp, 2] / 2 * (m.T[t] / 1000) ** 2
            + m.cp_coef[comp, 3] / 3 * (m.T[t] / 1000) ** 3
            + m.cp_coef[comp, 4] / 4 * (m.T[t] / 1000) ** 4
            - m.cp_coef[comp, 5] / (m.T[t] / 1000)
            + m.cp_coef[comp, 6]
        )

    def cp_i(m, comp, t):
        return (
            m.cp[comp, t]
            == m.cp_coef[comp, 1]
            + m.cp_coef[comp, 2] * (m.T[t] / 1000)
            + m.cp_coef[comp, 3] * (m.T[t] / 1000) ** 2
            + m.cp_coef[comp, 4] * (m.T[t] / 1000) ** 3
            + m.cp_coef[comp, 5] / (m.T[t] / 1000) ** 2
        )

    def Hbal(m, t):
        return sum(m.n[comp, t] * m.cp[comp, t] for comp in m.components) / m.Vg[
            t
        ] * m.dT[t] == -(
            m.q[t] * 1e6
            + m.rho_cat
            * (1 - m.eps)
            / m.eps
            * sum(
                m.h[comp, t] * sum(m.nu[rxn, comp] * m.r[rxn, t] for rxn in m.rxns)
                for comp in m.components
            )
        )

    def rate(m, rxn, t):
        return m.r[rxn, t] == m.k0[rxn] * exp(-m.EA0[rxn] / (m.R * m.T[t])) * (
            m.n[1, t] * (m.P[t]) / sum(m.n[comp, t] for comp in m.components)
        ) ** m.exp_e[rxn] * (
            m.n[2, t] * (m.P[t]) / sum(m.n[comp, t] for comp in m.components)
        ) ** m.exp_o[
            rxn
        ] / (
            1
            + m.K0[rxn]
            * exp(m.Tads[rxn] / m.T[t])
            * m.n[4, t]
            * (m.P[t])
            / sum(m.n[comp, t] for comp in m.components)
        )

    def pres(m, t):
        return m.dP[t] == -(
            150 * 2.52e-5 * (1 - 0.4) ** 2 / (0.2e-3 * 0.4**3)
        )  ## -2000#*m.P[t]

    def Elim(m, t):
        return sum(m.n[comp, t] for comp in m.components) >= 10 * m.n[1, t]

    def outT(m):
        return outlet[1] == m.T[res_time]

    def outP(m):
        return outlet[2] == m.P[res_time]

    def outn(m):
        return outlet[3] == sum(m.n[comp, res_time] for comp in m.components)

    def outz(m, comp):
        return outlet[3 + comp] == m.n[comp, res_time] / sum(
            m.n[comp, res_time] for comp in m.components
        )

    def outH(m):
        return outlet[10] == sum(
            m.n[comp, res_time] * m.h[comp, res_time] for comp in m.components
        )

    def integ_j(m, t):
        return m.jE[t]

    m.integ_j = Integral(m.t, wrt=m.t, rule=integ_j)

    def integ_q(m, t):
        return m.q[t]

    m.integ_q = Integral(m.t, wrt=m.t, rule=integ_q)

    def j_max(m, t):
        return m.integ_j <= 1

    eqs = [
        ["", "inT", ""],
        ["", "inP", ""],
        ["m.components,", "inn", ""],
        ["m.components,m.t,", "molbal", ""],
        ["m.t,", "ig", ""],
        ["m.components,m.t,", "h_i", "HB"],
        ["m.rxns,m.t,", "rate", ""],
        ["m.components,m.t,", "cp_i", ""],
        ["m.t,", "Hbal", "HB"],
        ["m.t,", "pres", ""],
        ["", "outT", ""],
        ["", "outP", ""],
        ["", "outn", ""],
        ["m.components,", "outz", ""],
        ["", "outH", "HB"],
        ["m.t,", "j_max", ""],
    ]
    for eq in eqs:
        const = compile(
            "m.c_"
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
        if eq[2] == "HB":  # heat balance
            m.heat_constr[m.heat_constr_cnt.value] = str(
                "m.c_" + str(m.constr_cnt.value)
            )
            m.heat_constr_cnt.value += 1
        m.constr_cnt.value += 1
    discretizer = TransformationFactory("dae.collocation")
    discretizer.apply_to(m, scheme="LAGRANGE-RADAU", nfe=3, ncp=5)
    for i in m.T:
        m.q[i].fix(0)
        m.jE[i].fix(0)
        m.T[i] = inlet[1].value
        m.P[i] = inlet[2].value
        for j in m.components:
            m.n[j, i] = (
                inlet[3].value * inlet[j + 3].value
                + m.nu[1, j] * i / (16 * res_time)
                + m.nu[2, j] * i / (16 * res_time)
            )
            m.h[j, i] = 1000 * (
                m.cp_coef[j, 1] * (m.T[i].value / 1000)
                + m.cp_coef[j, 2] / 2 * (m.T[i].value / 1000) ** 2
                + m.cp_coef[j, 3] / 3 * (m.T[i].value / 1000) ** 3
                + m.cp_coef[j, 4] / 4 * (m.T[i].value / 1000) ** 4
                - m.cp_coef[j, 5] / (m.T[i].value / 1000)
                + m.cp_coef[j, 6]
            )
    control_lst = [*[str(m.q[i]) for i in m.t], *[str(m.jE[i]) for i in m.t]]
    for contr in control_lst:
        contrdict = compile(
            "m.control_dict[value(m.control_cnt)]=str(m.{})".format(contr),
            "<string>",
            "exec",
        )
        exec(contrdict)
        m.control_cnt.value += 1
    m.cost_dict[value(m.cost_cnt)] = 0.5 * (
        ((2 * m.q_cost.value * value(m.integ_q)) ** 2 + 0.01**2) ** 0.5
    )

    def cost1(m):
        return m.cost_dict[value(m.cost_cnt)] == 0.5 * (
            ((2 * m.q_cost * m.integ_q) ** 2 + 0.01**2) ** 0.5
        )

    m.c_cost1 = Constraint(expr=cost1)
    m.cost_cnt.value += 1
    return m


def VLE(m, inlet, outlets, vf):
    m.old_constr_cnt.value = m.constr_cnt.value
    try:  ##predict outcomes and set variables
        [To, q, Ho, y, hE, hO, hEO, hC, hW, hN] = RF_VLE.predict(
            [[*[inlet[i].value for i in m.stream_props], vf]]
        )[0]
        # [q,nx,xE,xO,xEO,xC,xW,xN,Hx,ny,yE,yO,yEO,yC,yW,yN,Hy]=np.array([max(i, 0) for i in NNhxt(Tensor([*[(inlet[i].value-intcp[i-1])/slope[i-1] for i in m.stream_props],(T0-intcp[0])/slope[0]])).detach().numpy()])
        # Ho=Ho*slope[-1]+intcp[-1]; q=q*10-5 ##needed only for NN
        for i in m.props:
            outlets[0][i].value = [To, inlet[2].value, ny, yE, yO, yEO, yC, yW, yN, Hy][
                i - 1
            ]
            outlets[1][i].value = [To, inlet[2].value, nx, xE, xO, xEO, xC, xW, xN, Hx][
                i - 1
            ]
        line1 = compile(
            "m.q{}=Var(initialize=q, within=Reals, bounds=(-5,10))".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line1)
        line4 = compile(
            "m.h{}=Var(m.components,initialize=dict(zip([1,2,3,4,5,6],[hE,hO,hEO,hC,hW,hN])),bounds=h_bnds,within=Reals)".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line4)
    except:  ###set variables without prediction
        for i in m.props:
            outlets[0][i].value = [
                inlet[1].value,
                inlet[2].value,
                *[max(1e-5, inlet[j].value / 2.2) for j in [3, 4, 5, 6, 7, 8, 9, 10]],
            ][i - 1]
            outlets[1][i].value = [
                inlet[1].value,
                inlet[2].value,
                *[max(1e-5, inlet[j].value / 2.2) for j in [3, 4, 5, 6, 7, 8, 9, 10]],
            ][i - 1]
        line1 = compile(
            "m.q{}=Var(initialize=0, within=Reals, bounds=(-5,10))".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line1)
        line2 = compile(
            "m.h{}=Var(m.components,initialize=dict(zip([1,2,3,4,5,6],[0,0,0,0,0,0])),bounds=h_bnds,within=Reals)".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line2)
    line0 = compile(
        "m.cont_var{}=Var(initialize=vf, within=NonNegativeReals, bounds=(0,1));m.cont_var{}.fix()".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line0)

    def psat(T):
        return 10 ** (m.psat_coef[1] - (m.psat_coef[2] / (T + m.psat_coef[3])))

    def henry(comp, T):
        return (
            (m.he_coef[comp, 1] * exp(m.he_coef[comp, 2] * (1 / T - 1 / 298)))
            * (1.83)
            * (1 / 101325)
        )  ## Hcp* (Hxp/Hcp) * (atm/Pa)

    def totmol(m):
        return outlets[0][3] + outlets[1][3] == inlet[3]

    def masbal(m, comp):
        return (
            outlets[0][comp + 3] * outlets[0][3] + outlets[1][comp + 3] * outlets[1][3]
            == inlet[comp + 3] * inlet[3]
        )

    eq1 = compile(
        "def Htot(m): return inlet[10]+m.q{}*1000000==outlets[0][10]+outlets[1][10]".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq1, locals())
    eq4 = compile(
        "def h_i(m,comp): return m.h{}[comp]/1000==m.cp_coef[comp,1]*(outlets[0][1]/1000)+m.cp_coef[comp,2]/2*(outlets[0][1]/1000)**2+m.cp_coef[comp,3]/3*(outlets[0][1]/1000)**3+m.cp_coef[comp,4]/4*(outlets[0][1]/1000)**4-m.cp_coef[comp,5]/(outlets[0][1]/1000)+m.cp_coef[comp,6]".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq4, locals())
    eq5 = compile(
        "def Hl(m): return outlets[1][10] == outlets[1][3]*sum(outlets[1][3+comp]*m.h{}[comp] for comp in m.components)".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq5, locals())
    eq6 = compile(
        "def Hv(m): return outlets[0][10] == outlets[0][3]*outlets[0][8]*m.Hvap_coef+outlets[0][3]*sum(outlets[0][3+comp]*m.h{}[comp] for comp in m.components)".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq6, locals())

    def vlebal(m):
        return (
            sum(outlets[1][comp + 3] - outlets[0][comp + 3] for comp in m.components)
            == 0
        )

    def vle(m, comp):
        if comp == 5:
            return outlets[0][comp + 3] * outlets[0][2] == outlets[1][comp + 3] * psat(
                outlets[0][1]
            )  ## y * P = x * Psat
        else:
            return (henry(comp, outlets[0][1])) * outlets[0][comp + 3] * outlets[0][
                2
            ] == outlets[1][
                comp + 3
            ]  ## Hxp * y * P = x

    eq7 = compile(
        "def TT1(m): return outlets[1][1]==outlets[0][1]".format(str(inlet)[-2::]),
        "<string>",
        "exec",
    )
    exec(eq7, locals())
    eq8 = compile(
        "def cont_var_eq(m): return outlets[0][3]*outlets[0][8]/(inlet[3]*inlet[8])==m.cont_var{}".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq8, locals())

    def PP1(m):
        return inlet[2] == outlets[0][2]

    def PP2(m):
        return inlet[2] == outlets[1][2]

    eqs = [
        ["", "totmol", ""],
        ["m.components,", "masbal", ""],
        ["m.components,", "h_i", "HB"],
        ["", "Hl", "HB"],
        ["", "Hv", "HB"],
        ["", "Htot", "HB"],
        ["", "vlebal", ""],
        ["m.components,", "vle", ""],
        ["", "TT1", ""],
        ["", "cont_var_eq", ""],
        ["", "PP1", ""],
        ["", "PP2", ""],
    ]
    for eq in eqs:
        const = compile(
            "m.c_"
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
        if eq[2] == "HB":  # heat balance
            m.heat_constr[m.heat_constr_cnt.value] = str(
                "m.c_" + str(m.constr_cnt.value)
            )
            m.heat_constr_cnt.value += 1
        m.constr_cnt.value += 1
    contrdict = compile(
        "m.control_dict[value(m.control_cnt)]=str(m.cont_var{})".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(contrdict)
    m.control_cnt.value += 1
    costdict = compile(
        "m.cost_dict[value(m.cost_cnt)] = 0.5*(((2*m.q_cost.value*value(m.q{}))**2+0.01**2)**0.5)".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(costdict)
    costdict = compile(
        "def cost_vle(m): return m.cost_dict[value(m.cost_cnt)] == 0.5*(((2*m.q_cost*m.q{})**2+0.01**2)**0.5)".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(costdict)
    costdict = compile("m.c_vle=Constraint(expr=cost_vle)", "<string>", "exec")
    exec(costdict)
    m.cost_cnt.value += 1
    # print('VLE:', [m.cost_dict[i].value for i in range(1,value(m.cost_cnt))])
    return m


def HX(m, inlet, outlet, T0):  ### Done
    m.old_constr_cnt.value = m.constr_cnt.value
    for i in m.props:
        outlet[i].value = [T0, *[inlet[j].value for j in [2, 3, 4, 5, 6, 7, 8, 9, 10]]][
            i - 1
        ]
    line1 = compile(
        "m.q{}=Var(initialize=0.01, within=Reals, bounds=(-5,5))".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line1)
    line2 = compile(
        "m.vley{}=Var(initialize=inlet[8], bounds=(0,1), within=NonNegativeReals)".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line2)
    line4 = compile(
        "m.h{}=Var(m.components,initialize=dict(zip([1,2,3,4,5,6],[0,0,0,0,0,0])),bounds=h_bnds,within=Reals)".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line4)
    line0 = compile(
        "m.cont_var{}=Var(initialize=T0, within=NonNegativeReals, bounds=(intcp[0],slope[0]+intcp[0]));m.cont_var{}.fix()".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line0)

    def psat(T):
        return 10 ** (m.psat_coef[1] - (m.psat_coef[2] / (T + m.psat_coef[3])))

    def miny(T, P):
        return 0.5 * (
            psat(T) / P
            + outlet[8]
            - ((psat(T) / P - outlet[8]) ** 2 + 0.001**2) ** 0.5
        )

    def moltot(m):
        return outlet[3] == inlet[3]

    def molbal(m, i):
        return outlet[i + 3] == inlet[i + 3]

    eq1 = compile(
        "def Htot(m): return inlet[10]+m.q{}*1000000==outlet[10]".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq1, locals())
    eq2 = compile(
        "def antoine(m): return m.vley{}==0.5*(miny(outlet[1],outlet[2])+(miny(outlet[1],outlet[2])**2+0.00005**2)**0.5)".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq2, locals())
    eq4 = compile(
        "def h_i(m,comp): return m.h{}[comp]/1000==m.cp_coef[comp,1]*(outlet[1]/1000)+m.cp_coef[comp,2]/2*(outlet[1]/1000)**2+m.cp_coef[comp,3]/3*(outlet[1]/1000)**3+m.cp_coef[comp,4]/4*(outlet[1]/1000)**4-m.cp_coef[comp,5]/(outlet[1]/1000)+m.cp_coef[comp,6]".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq4, locals())
    eq5 = compile(
        "def Hbal(m): return outlet[10] == outlet[3]*outlet[8]*m.vley{}*m.Hvap_coef+outlet[3]*sum(outlet[3+comp]*m.h{}[comp] for comp in m.components)".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq5, locals())
    eq6 = compile(
        "def TT(m): return outlet[1]==m.cont_var{}".format(str(inlet)[-2::]),
        "<string>",
        "exec",
    )
    exec(eq6, locals())

    def PP(m):
        return outlet[2] == inlet[2]

    eqs = [
        ["", "moltot", ""],
        ["m.components,", "molbal", ""],
        ["", "Htot", "HB"],
        ["", "antoine", ""],
        ["m.components,", "h_i", "HB"],
        ["", "Hbal", "HB"],
        ["", "TT", ""],
        ["", "PP", ""],
    ]  # ['','phasemasbal']
    for eq in eqs:
        const = compile(
            "m.c_{}=Constraint({}expr={})".format(
                str(value(m.constr_cnt)), eq[0], eq[1]
            ),
            "<string>",
            "exec",
        )
        exec(const)
        if eq[2] == "HB":  # heat balance
            m.heat_constr[m.heat_constr_cnt.value] = str(
                "m.c_" + str(m.constr_cnt.value)
            )
            m.heat_constr_cnt.value += 1
        m.constr_cnt.value += 1
    contrdict = compile(
        "m.control_dict[value(m.control_cnt)]=str(m.cont_var{})".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(contrdict)
    m.control_cnt.value += 1
    # costdict=compile('m.cost_dict[value(m.cost_cnt)]=str("m.q_cost*m.q{}")'.format(str(outlet)[-2::]),'<string>','exec');exec(costdict)
    # m.cost_cnt.value+=1
    costdict = compile(
        "def cost_hx(m): return m.cost_dict[value(m.cost_cnt)] == 0.5*(((2*m.q_cost*m.q{})**2+0.01**2)**0.5)".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(costdict)
    costdict = compile(
        "m.cost_dict[value(m.cost_cnt)] = 0.5*(((2*m.q_cost.value*value(m.q{}))**2+0.01**2)**0.5)".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(costdict)
    costdict = compile("m.c_hx=Constraint(expr=cost_hx)", "<string>", "exec")
    exec(costdict)
    m.cost_cnt.value += 1
    return m


def air(m, inlet, outlet, amtAir):  ## max amtAir=20 mol/s
    constr_cnt = m.constr_cnt.value
    air = [
        298,
        inlet[2].value,
        amtAir,
        1e-5,
        2.099974e-01,
        1e-5,
        1.0e-5,
        1.0e-05,
        7.899626e-01,
        amtAir * -7.366384194e-01,
    ]
    line0 = compile(
        "m.cont_var{}=Var(initialize=amtAir, bounds=(0,100), within=NonNegativeReals); m.cont_var{}.fix()".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line0)
    line1 = compile(
        "m.air{}=Var(m.props,initialize=dict(zip(range(1,11),air)), bounds=air_bnds, within=Reals);m.air{}.fix();m.air{}[3].unfix();m.air{}[10].unfix() ".format(
            str(inlet)[-2::], str(inlet)[-2::], str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line1)
    line2 = compile(
        "def airflow(m): return m.air{}[3]==m.cont_var{}".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line2)
    const = compile(
        "m.c_{}=Constraint(expr=airflow)".format(str(value(m.constr_cnt))),
        "<string>",
        "exec",
    )
    exec(const)
    m.constr_cnt.value += 1
    line2 = compile(
        "def airH(m): return m.air{}[10]==m.cont_var{}*-7.36638419e-01".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line2)
    const = compile(
        "m.c_{}=Constraint(expr=airH)".format(str(value(m.constr_cnt))),
        "<string>",
        "exec",
    )
    exec(const)
    m.heat_constr[value(m.heat_constr_cnt)].value = "m.c_{}".format(
        str(value(m.constr_cnt))
    )
    m.heat_constr_cnt.value += 1
    m.constr_cnt.value += 1
    line3 = compile(
        "m=mix(m,[inlet,m.air{}],outlet)".format(str(inlet)[-2::]), "<string>", "exec"
    )
    exec(line3)
    contrdict = compile(
        "m.control_dict[value(m.control_cnt)]=str(m.cont_var{})".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(contrdict)
    m.control_cnt.value += 1
    m.old_constr_cnt.value = constr_cnt
    return m


def mix(m, inlets, outlet):
    m.old_constr_cnt.value = m.constr_cnt.value
    # try: [no,Ao,Bo,Co,Do,Ho,q,x1,x2,x3,x4,y1,y2,y3,y4,Vo]=np.array([max(i, 0) for i in NNhxt(Tensor([*s,T0])).detach().numpy()])*np.array([25,1,1,1,1,2e6,4,1,1,1,1,1,1,1,1,1])
    # except: pass
    line1 = compile(
        "m.vley{}=Var(initialize=0, bounds=(0,1), within=NonNegativeReals)".format(
            str(inlets[0])[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line1)
    line2 = compile(
        "m.h{}=Var(m.components,initialize=dict(zip([1,2,3,4,5,6],[0,0,0,0,0,0])),bounds=h_bnds,within=Reals)".format(
            str(inlets[0])[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line2)

    def psat(T):
        return 10 ** (m.psat_coef[1] - (m.psat_coef[2] / (T + m.psat_coef[3])))

    def miny(s):
        return 0.5 * (
            psat(s[1]) / s[2]
            + s[8]
            - ((psat(s[1]) / s[2] - s[8]) ** 2 + 0.1**2) ** 0.5
        )

    def totmol(m):
        return outlet[3] == sum(inlet[3] for inlet in inlets)

    def masbal(m, comp):
        return outlet[comp + 3] * outlet[3] == sum(
            inlet[comp + 3] * inlet[3] for inlet in inlets
        )

    def Htot(m):
        return outlet[10] == sum(inlet[10] for inlet in inlets)

    eq1 = compile(
        "def antoine(m): return m.vley{}==0.5*(miny(outlet)+(miny(outlet)**2+0.00005**2)**0.5)".format(
            str(inlets[0])[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq1, locals())
    eq2 = compile(
        "def h_i(m,comp): return m.h{}[comp]/1000==m.cp_coef[comp,1]*(outlet[1]/1000)+m.cp_coef[comp,2]/2*(outlet[1]/1000)**2+m.cp_coef[comp,3]/3*(outlet[1]/1000)**3+m.cp_coef[comp,4]/4*(outlet[1]/1000)**4-m.cp_coef[comp,5]/(outlet[1]/1000)+m.cp_coef[comp,6]".format(
            str(inlets[0])[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq2, locals())
    eq3 = compile(
        "def Hbal(m): return outlet[10] == outlet[3]*outlet[8]*m.vley{}*m.Hvap_coef+outlet[3]*sum(outlet[3+comp]*m.h{}[comp] for comp in m.components)".format(
            str(inlets[0])[-2::], str(inlets[0])[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq3, locals())

    def PP(m):
        return outlet[2] == inlets[0][2]

    eqs = [
        ["", "totmol", ""],
        ["m.components,", "masbal", ""],
        ["", "Htot", "HB"],
        ["", "antoine", ""],
        ["m.components,", "h_i", "HB"],
        ["", "Hbal", "HB"],
        ["", "PP", ""],
    ]
    for eq in eqs:
        const = compile(
            "m.c_{}=Constraint({}expr={})".format(
                str(value(m.constr_cnt)), eq[0], eq[1]
            ),
            "<string>",
            "exec",
        )
        exec(const)
        if eq[2] == "HB":  # heat balance
            m.heat_constr[value(m.heat_constr_cnt)].value = "m.c_{}".format(
                str(value(m.constr_cnt))
            )
            m.heat_constr_cnt.value += 1
        m.constr_cnt.value += 1
    return m


def splitter(m, inlet, outlets, purge0):
    m.old_constr_cnt.value = m.constr_cnt.value
    line1 = compile(
        "m.cont_var{}=Var(initialize=purge0, within=NonNegativeReals, bounds=(0,1));m.cont_var{}.fix()".format(
            str(inlet)[-2:], str(inlet)[-2:]
        ),
        "<string>",
        "exec",
    )
    exec(line1)
    for i in [3, 10]:
        outlets[0][i].value = purge0 * inlet[i].value
        outlets[1][i].value = (1 - purge0) * inlet[i].value
    for i in [1, 2, 4, 5, 6, 7, 8, 9]:
        outlets[0][i].value = inlet[i].value
        outlets[1][i].value = inlet[i].value

    def totmol(m):
        return sum(outlet[3] for outlet in outlets) == inlet[3]

    eq1 = compile(
        "def cont_varmol(m): return outlets[0][3]==inlet[3]*m.cont_var{}".format(
            str(inlet)[-2:]
        ),
        "<string>",
        "exec",
    )
    exec(eq1, locals())

    def masbal1(m, comp):
        return outlets[0][comp + 3] == inlet[comp + 3]

    def masbal2(m, comp):
        return outlets[1][comp + 3] == inlet[comp + 3]

    def Hbal(m):
        return sum(outlet[10] for outlet in outlets) == inlet[10]

    eq2 = compile(
        "def cont_varH(m): return outlets[0][10]==inlet[10]*m.cont_var{}".format(
            str(inlet)[-2:]
        ),
        "<string>",
        "exec",
    )
    exec(eq2, locals())

    def TT1(m):
        return inlet[1] == outlets[0][1]

    def TT2(m):
        return inlet[1] == outlets[1][1]

    def PP1(m):
        return inlet[2] == outlets[0][2]

    def PP2(m):
        return inlet[2] == outlets[1][2]

    eqs = [
        ["", "Hbal", "HB"],
        ["", "cont_varH", "HB"],
        ["", "totmol", ""],
        ["", "cont_varmol", ""],
        ["m.components,", "masbal1", ""],
        ["m.components,", "masbal2", ""],
        ["", "TT1", ""],
        ["", "TT2", ""],
        ["", "PP1", ""],
        ["", "PP2", ""],
    ]
    for eq in eqs:
        const = compile(
            "m.c_{}=Constraint({}expr={})".format(
                str(value(m.constr_cnt)), eq[0], eq[1]
            ),
            "<string>",
            "exec",
        )
        exec(const)
        if eq[2] == "HB":  # heat balance
            m.heat_constr[m.heat_constr_cnt.value] = str(
                "m.c_" + str(m.constr_cnt.value)
            )
            m.heat_constr_cnt.value += 1
        m.constr_cnt.value += 1
    contrdict = compile(
        "m.control_dict[value(m.control_cnt)]=str(m.cont_var{})".format(
            str(inlet)[-2:]
        ),
        "<string>",
        "exec",
    )
    exec(contrdict)
    m.control_cnt.value += 1
    return m


def dP(m, inlet, outlet, Po, rec):  ##isothermal expansion and compression
    m.old_constr_cnt.value = m.constr_cnt.value
    line0 = compile(
        "m.cont_var{}=Var(initialize=Po, within=NonNegativeReals, bounds=(intcp[1],slope[1]+intcp[1]));m.cont_var{}.fix()".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line0)
    for i in m.props:
        outlet[i].value = inlet[i].value
    line00 = compile(
        "outlet[2]=m.cont_var{}.value".format(str(inlet)[-2::]), "<string>", "exec"
    )
    exec(line00, locals())
    try:  ##preict outcomes and set variables
        [Pwo, Ho, y, vfo, hE, hO, hEO, hC, hW, hN] = np.array(
            [
                max(i, 0)
                for i in NNhxt(
                    Tensor(
                        [
                            *[
                                (inlet[i].value - intcp[i - 1]) / slope[i - 1]
                                for i in m.stream_props
                            ],
                            (T0 - intcp[0]) / slope[0],
                        ]
                    )
                )
                .detach()
                .numpy()
            ]
        )
        Ho = Ho * slope[-1] + intcp[-1]
        Pwo = Pwo * 1e8
        hE, hO, hEO, hC, hW, hN = np.array([hE, hO, hEO, hC, hW, hN]) * np.array(
            [2e5] * 6
        ) + np.array([-10] * 6)
        line1 = compile(
            "m.Pw{}=Var(initialize=Pwo, within=NonNegativeReals, bounds=(0,1e9))".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line1)
        line2 = compile(
            "m.vley{}=Var(initialize=y, bounds=(0,1), within=NonNegativeReals)".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line2)
        line3 = compile(
            "m.h{}=Var(m.components,initialize=dict(zip([1,2,3,4,5,6],[hE,hO,hEO,hC,hW,hN])),bounds=h_bnds,within=Reals)".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line3)
    except:  ###set variables without prediction
        line1 = compile(
            "m.Pw{}=Var(initialize=0, within=NonNegativeReals, bounds=(0,1e9))".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line1)
        line2 = compile(
            "m.vley{}=Var(initialize=inlet[8], bounds=(0,1), within=NonNegativeReals)".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line2)
        line3 = compile(
            "m.h{}=Var(m.components,initialize=dict(zip([1,2,3,4,5,6],[0,0,0,0,0,0])),bounds=h_bnds,within=Reals)".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line3)
    line4 = compile(
        "m.V{}=Var(initialize=inlet[3]*m.R*inlet[1]/inlet[2],bounds=(0,20),within=NonNegativeReals)".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line4)

    def psat(T):
        return 10 ** (m.psat_coef[1] - (m.psat_coef[2] / (T + m.psat_coef[3])))

    def miny(outlet):
        return 0.5 * (
            psat(outlet[1]) / outlet[2]
            + outlet[8]
            - ((psat(outlet[1]) / outlet[2] - outlet[8]) ** 2 + 0.01**2) ** 0.5
        )

    def moltot(m):
        return inlet[3] == outlet[3]

    def molbal(m, i):
        return outlet[i + 3] == inlet[i + 3]

    eq1 = compile(
        "def antoine(m): return m.vley{}==0.5*(miny(outlet)+(miny(outlet)**2+0.00005**2)**0.5)".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq1, locals())
    eq2 = compile(
        "def h_i(m,comp): return m.h{}[comp]/1000 == m.cp_coef[comp,1]*(outlet[1]/1000) + m.cp_coef[comp,2]/2*(outlet[1]/1000)**2+m.cp_coef[comp,3]/3*(outlet[1]/1000)**3+ m.cp_coef[comp,4]/4*(outlet[1]/1000)**4-m.cp_coef[comp,5]/(outlet[1]/1000)+m.cp_coef[comp,6]".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq2, locals())
    eq3 = compile(
        "def Htot(m): return outlet[10] == outlet[3]*outlet[8]*m.vley{}*m.Hvap_coef+outlet[3]*sum(outlet[3+comp]*m.h{}[comp] for comp in m.components)".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq3, locals())

    def TT(m):
        return outlet[1] == inlet[1]

    eq7 = compile(
        "def PP(m): return outlet[2] == m.cont_var{}".format(str(inlet)[-2::]),
        "<string>",
        "exec",
    )
    exec(eq7, locals())
    eq4 = compile(
        "def VV(m): return m.V{} == inlet[3]*m.R*inlet[1]/inlet[2]".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq4, locals())
    eq5 = compile(
        "def pump_work(): return m.V{}*inlet[2]*(outlet[2]/inlet[2]-1)".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq5, locals())
    eq6 = compile(
        "def Power(m): return m.Pw{} == 0.5*(pump_work()+((pump_work())**2+0.01**2)**0.5)".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq6, locals())
    eqs = [
        ["", "moltot", ""],
        ["m.components,", "molbal", ""],
        ["", "antoine", ""],
        ["m.components,", "h_i", "HB"],
        ["", "Htot", "HB"],
        ["", "TT", ""],
        ["", "PP", ""],
        ["", "VV", ""],
        ["", "Power", ""],
    ]
    for eq in eqs:
        const = compile(
            "m.c_"
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
        if eq[2] == "HB":  # heat balance
            m.heat_constr[m.heat_constr_cnt.value] = str(
                "m.c_" + str(m.constr_cnt.value)
            )
            m.heat_constr_cnt.value += 1
        m.constr_cnt.value += 1
    # costdict=compile('m.cost_dict[value(m.cost_cnt)]=str("m.Pw_cost*m.Pw{}")'.format(str(outlet)[-2::]),'<string>','exec');exec(costdict)
    # m.cost_cnt.value+=1
    if rec == False:
        contrdict = compile(
            "m.control_dict[value(m.control_cnt)]=str(m.cont_var{})".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(contrdict)
        m.control_cnt.value += 1
    costdict = compile(
        "def cost_dP(m): return m.cost_dict[value(m.cost_cnt)] == m.Pw_cost*m.Pw{}".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(costdict)
    costdict = compile(
        "m.cost_dict[value(m.cost_cnt)]  = m.Pw_cost.value*value(m.Pw{})".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(costdict)
    costdict = compile("m.c_dP=Constraint(expr=cost_dP)", "<string>", "exec")
    exec(costdict)
    m.cost_cnt.value += 1
    return m


def GLE(m, inlet, outlets, amt_water):  ### GLE w pure water
    m.old_constr_cnt.value = m.constr_cnt.value
    line1 = compile(
        "m.cont_var{}=Var(initialize=max(0,amt_water-inlet[3].value*inlet[8].value),within=NonNegativeReals,bounds=(0,50));m.cont_var{}.fix()".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line1)
    # line11=compile('cont_var=m.cont_var{}'.format(str(outlets[0])[-2::]),'<string>','eval');eval(line11)
    line2 = compile(
        "m.yhat{}=Var(m.components,initialize=dict(zip([1,2,3,4,5,6],[0,0,0,0,0,0])),within=NonNegativeReals)".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line2)
    line3 = compile(
        "m.N{}=Var(initialize=5, within=PositiveReals, bounds=(1,8));m.N{}.fix()".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(line3)
    try:  ##predict outcomes and set variables
        [
            q,
            nx,
            xE,
            xO,
            xEO,
            xC,
            xW,
            xN,
            Hx,
            ny,
            yE,
            yO,
            yEO,
            yC,
            yW,
            yN,
            Hy,
        ] = np.array(
            [
                max(i, 0)
                for i in NNhxt(
                    Tensor(
                        [
                            *[
                                (inlet[i].value - intcp[i - 1]) / slope[i - 1]
                                for i in m.stream_props
                            ],
                            (T0 - intcp[0]) / slope[0],
                        ]
                    )
                )
                .detach()
                .numpy()
            ]
        )
        Ho = Ho * slope[-1] + intcp[-1]
        q = q * 10 - 5  ##needed only for NN
        for i in m.props:
            outlets[0][i].value = [T0, inlet[2].value, ny, yE, yO, yEO, yC, yW, yN, Hy][
                i - 1
            ]
            outlets[1][i].value = [T0, inlet[2].value, nx, xE, xO, xEO, xC, xW, xN, Hx][
                i - 1
            ]
        line4 = compile(
            "m.h{}=Var(m.components,initialize=dict(zip([1,2,3,4,5,6],[hE,hO,hEO,hC,hW,hN])),bounds=h_bnds,within=Reals)".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line4)
    except:  ###set variables without prediction
        for i in m.props:
            outlets[0][i].value = inlet[i].value
            outlets[1][i].value = [
                inlet[1].value,
                inlet[2].value,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
            ][i - 1]
        outlets[0][3].value = inlet[3].value * (1 - inlet[8].value)
        outlets[0][8].value = 0
        line5 = compile(
            "outlets[1][3].value=inlet[3].value*inlet[8].value+m.cont_var{}.value".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line5)
        line6 = compile(
            "m.h{}=Var(m.components,initialize=dict(zip([1,2,3,4,5,6],[0,0,0,0,0,0])),bounds=h_bnds,within=Reals)".format(
                str(inlet)[-2::]
            ),
            "<string>",
            "exec",
        )
        exec(line6)

    def psat(T):
        return 10 ** (m.psat_coef[1] - (m.psat_coef[2] / (T + m.psat_coef[3])))

    def henry(T, comp):
        return (
            m.he_coef[comp, 1]
            * exp(m.he_coef[comp, 2] * (1 / T - 1 / 298))
            * (1.83)
            * (1 / 101325)
        )  ## Hcp* (Hxp/Hcp) * (atm/Pa)

    def K(comp):
        return 1 / (
            henry(outlets[0][1], comp) * outlets[0][2]
        )  ###  --> K = y/x = 1/(Hxp*P)

    eq1 = compile(
        "def totmol(m): return outlets[1][3]+outlets[0][3]==inlet[3]+m.cont_var{}".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq1, locals())
    eq2 = compile(
        "def masbal(m,comp): \n if comp==5: return (outlets[1][comp+3]*outlets[1][3]+outlets[0][comp+3]*outlets[0][3])==inlet[comp+3]*inlet[3]+m.cont_var{} \
                \n else: return outlets[1][comp+3]*outlets[1][3]+outlets[0][comp+3]*outlets[0][3]==inlet[comp+3]*inlet[3]".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq2, locals())
    eq3 = compile(
        "def h_i(m,comp): return m.h{}[comp]/1000==m.cp_coef[comp,1]*(outlets[0][1])/1000 + m.cp_coef[comp,2]/2*((outlets[0][1]/1000)**2)+m.cp_coef[comp,3]/3*(outlets[0][1]/1000)**3+m.cp_coef[comp,4]/4*(outlets[0][1]/1000)**4-m.cp_coef[comp,5]/(outlets[0][1]/1000)+m.cp_coef[comp,6]".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq3, locals())
    eq4 = compile(
        "def Hl(m): return outlets[1][10]==outlets[1][3]*sum(outlets[1][comp+3]*m.h{}[comp] for comp in m.components)+m.cont_var{}*m.h{}[5]".format(
            str(inlet)[-2::], str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq4, locals())
    eq5 = compile(
        "def Hv(m): return outlets[0][10] ==outlets[0][3]*(outlets[0][8]*m.Hvap_coef+sum(outlets[0][comp+3]*m.h{}[comp] for comp in m.components))".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq5, locals())
    eq6 = compile(
        "def vle(m,comp): \n if comp == 5: return outlets[0][comp+3]*outlets[0][2]==outlets[1][comp+3]*psat(outlets[0][1]) \
                \n else: return ((inlet[comp+3]-K(comp)*outlets[1][comp+3])/m.yhat{}[comp])==((m.cont_var{}+inlet[3]*inlet[8])/(inlet[3]*(1-inlet[8]))/K(comp))**m.N{}".format(
            str(inlet)[-2::], str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq6, locals())
    eq7 = compile(
        "def scale_y(m,comp): return outlets[0][comp+3]==m.yhat{}[comp]/sum(m.yhat{}[i] for i in m.components)".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq7, locals())
    eq8 = compile(
        "def scale_n(m):return outlets[0][3]==inlet[3]*(sum(m.yhat{}[i] for i in m.components)-inlet[8]-inlet[6])".format(
            str(inlet)[-2::], str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(eq8, locals())

    def TT1(m):
        return inlet[1] == outlets[0][1]

    def TT2(m):
        return inlet[1] == outlets[1][1]

    def PP1(m):
        return outlets[0][2] == inlet[2]

    def PP2(m):
        return outlets[1][2] == inlet[2]

    eqs = [
        ["", "totmol", ""],
        ["m.components,", "masbal", ""],
        ["m.components,", "h_i", "HB"],
        ["", "Hl", "HB"],
        ["", "Hv", "HB"],
        ["", "scale_n", ""],
        ["m.components,", "scale_y", ""],
        ["m.components,", "vle", ""],
        ["", "TT1", ""],
        ["", "TT2", ""],
        ["", "PP1", ""],
        ["", "PP2", ""],
    ]
    for eq in eqs:
        const = compile(
            "m.c_"
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
        if eq[2] == "HB":  # heat balance
            m.heat_constr[m.heat_constr_cnt.value] = str(
                "m.c_" + str(m.constr_cnt.value)
            )
            m.heat_constr_cnt.value += 1
        m.constr_cnt.value += 1
    contrdict = compile(
        "m.control_dict[value(m.control_cnt)]=str(m.cont_var{})".format(
            str(inlet)[-2::]
        ),
        "<string>",
        "exec",
    )
    exec(contrdict)
    m.control_cnt.value += 1
    # costdict=compile('def cost_gle(m): return m.cost_dict[value(m.cost_cnt)] == m.W_cost*m.cont_var{}'.format(str(outlets[0])[-2::]),'<string>','exec');exec(costdict)
    # costdict=compile('m.cost_dict[value(m.cost_cnt)] =  m.W_cost.value*value(m.cont_var{})'.format(str(outlets[0])[-2::]),'<string>','exec');exec(costdict)
    # costdict=compile('m.c_gle=Constraint(expr=cost_gle)','<string>','exec');exec(costdict)
    # m.cost_cnt.value+=1
    return m


def findH(
    s,
):  ## takes s with values between 0 and 1,and finds H which is mainly a function of H_vap of water
    m = ConcreteModel()
    m.props = RangeSet(10)
    m.components = RangeSet(6)
    m.psat_coef_set = RangeSet(6)
    s = np.array(s) * slope[:-1] + intcp[:-1]
    m.psat_coef = Param(
        m.psat_coef_set,
        initialize=dict(zip([1, 2, 3], [11.21, 2345.731, 7.559])),
        within=Reals,
    )
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
    m.T = Param(initialize=s[0])
    m.P = Param(initialize=s[1])
    m.n = Param(initialize=s[2])
    m.z = Param(
        m.components, initialize={1: s[3], 2: s[4], 3: s[5], 4: s[6], 5: s[7], 6: s[8]}
    )
    m.vley = Var(initialize=0, bounds=(0, 1), within=NonNegativeReals)  ## water vapor
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
            psat(m.T) / s[1]
            + m.z[5]
            - ((psat(m.T) / s[1] - m.z[5]) ** 2 + 0.1**2) ** 0.5
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

    m.c1 = Constraint(
        expr=antoine
    )  # ; m.c2=Constraint(expr=phasemasbal)# ;m.c1=Constraint(expr=vle)
    m.c9 = Constraint(m.components, expr=h_i)
    m.c4 = Constraint(expr=Hbal)
    solver = SolverFactory("ipopt")
    solver.options = {"tol": 1e-8, "max_iter": 200}
    results = solver.solve(m, tee=False)
    if results.Solver.status == "ok":
        return np.array(
            [
                m.T.value,
                m.P.value,
                m.n.value,
                *[value(m.z[i]) for i in m.components],
                m.H.value,
            ]
        )  # , value(m.vley), m.vf.value
    else:
        print(results.Solver.status)
        return 0


def solve_PFR(inlet, res_time):
    try:
        t0 = time.time()
        m = start_episode()
        m.s01 = Var(
            m.props,
            initialize=dict(
                zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inlet * slope + intcp)
            ),
            bounds=strm_bnds,
            within=Reals,
        )
        m.s01.fix()
        m.s02 = Var(m.props, initialize=emptval, bounds=strm_bnds, within=Reals)
        res_time = res_time * 90  ##scaling factor
        m = PFR(m, m.s01, m.s02, res_time)
        solver = SolverFactory("ipopt")
        solver.options = {"tol": 1e-8, "max_iter": 500}
        results = solver.solve(m, tee=False)
        if results.Solver.status == "ok":
            return [
                [(m.s02[i].value - intcp[i - 1]) / slope[i - 1] for i in m.props]
            ], res_time
        else:
            # print('pfr bad')
            return [inlet], res_time
    except:
        # print('pfr error')
        return [inlet], res_time


def solve_HX(inlet, To):
    try:
        m = start_episode()
        m.s01 = Var(
            m.props,
            initialize=dict(
                zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inlet * slope + intcp)
            ),
            bounds=strm_bnds,
            within=Reals,
        )
        m.s01.fix()
        m.s02 = Var(m.props, initialize=emptval, bounds=strm_bnds, within=Reals)
        t0 = time.time()
        To = To * 232 + 298
        m = HX(m, m.s01, m.s02, To)
        solver = SolverFactory("ipopt")
        solver.options = {"tol": 1e-8, "max_iter": 400}
        results = solver.solve(m, tee=False)
        if results.Solver.status == "ok":
            return [
                [(m.s02[i].value - intcp[i - 1]) / slope[i - 1] for i in m.props]
            ], To
        else:
            # print('HX bad')
            return [inlet], To
    except:
        return [inlet], To


def solve_dP(inlet, Po):
    try:
        m = start_episode()
        m.s01 = Var(
            m.props,
            initialize=dict(
                zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inlet * slope + intcp)
            ),
            bounds=strm_bnds,
            within=Reals,
        )
        m.s01.fix()
        m.s02 = Var(m.props, initialize=emptval, bounds=strm_bnds, within=Reals)
        t0 = time.time()
        Po = Po * 1.9e6 + 1e5
        m = dP(m, m.s01, m.s02, Po, False)
        solver = SolverFactory("ipopt")
        solver.options = {"tol": 1e-8, "max_iter": 200}
        results = solver.solve(m, tee=False)
        if results.Solver.status == "ok":
            return [
                [(m.s02[i].value - intcp[i - 1]) / slope[i - 1] for i in m.props]
            ], Po
        else:
            # print('dP bad')
            return [inlet], Po
    except:
        return [inlet], Po


def solve_air(inlet, amtAir):
    try:
        m = start_episode()
        m.s01 = Var(
            m.props,
            initialize=dict(
                zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inlet * slope + intcp)
            ),
            bounds=strm_bnds,
            within=Reals,
        )
        m.s01.fix()
        # print(inlet,'\n s01:',amtAir)
        m.s02 = Var(m.props, initialize=emptval, bounds=strm_bnds, within=Reals)
        t0 = time.time()
        amtAir = amtAir * 90  # scaling factor
        m = air(m, m.s01, m.s02, amtAir)
        # m.pprint()
        solver = SolverFactory("ipopt")
        solver.options = {"tol": 1e-8, "max_iter": 200}
        results = solver.solve(m, tee=False)
        if results.Solver.status == "ok":
            return [[m.s02[i].value for i in m.props]]
        else:
            # print('add Air bad')
            return [inlet]
    except:
        return [inlet]


def solve_VLE(inlet, vf):
    try:
        m = start_episode()
        m.s01 = Var(
            m.props,
            initialize=dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inlet)),
            bounds=strm_bnds,
            within=Reals,
        )
        m.s01.fix()
        m.s02 = Var(m.props, initialize=emptval, bounds=strm_bnds, within=Reals)
        m.s03 = Var(m.props, initialize=emptval, bounds=strm_bnds, within=Reals)
        t0 = time.time()
        vf = vf  # *0.19+0.01
        m = VLE(m, m.s01, [m.s02, m.s03], vf)
        solver = SolverFactory("ipopt")
        solver.options = {"tol": 1e-5, "max_iter": 200}
        results = solver.solve(m, tee=False)
        if results.Solver.status == "ok":
            return [
                [m.s02[i].value for i in m.props],
                [m.s03[i].value for i in m.props],
            ]
        else:
            # print('VLE bad')
            return [inlet, inlet]
    except:
        # print('error')
        return [inlet, inlet]


def solve_GLE(inlet, water):
    try:
        m = start_episode()
        m.s01 = Var(
            m.props,
            initialize=dict(
                zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inlet * slope + intcp)
            ),
            bounds=strm_bnds,
            within=Reals,
        )
        m.s01.fix()
        m.s02 = Var(m.props, initialize=emptval, bounds=strm_bnds, within=Reals)  ## gas
        m.s03 = Var(
            m.props, initialize=emptval, bounds=strm_bnds, within=Reals
        )  ## liquid
        t0 = time.time()
        water = water * 20 + 10
        m = GLE(m, m.s01, [m.s02, m.s03], water)
        solver = SolverFactory("ipopt")
        solver.options = {"tol": 1e-8, "max_iter": 300}
        results = solver.solve(m, tee=False)
        # m.pprint()
        if results.Solver.status == "ok":
            return [
                [(m.s02[i].value - intcp[i - 1]) / slope[i - 1] for i in m.props],
                [(m.s03[i].value - intcp[i - 1]) / slope[i - 1] for i in m.props],
            ], water
        else:
            # print('GLE bad')
            return [inlet, inlet], water
    except:
        # print('GLE error')
        return [inlet, inlet], water


def solve_rec(inlet, purge_ratio, mix_stream):
    try:
        m = start_episode()
        m.s01 = Var(
            m.props,
            initialize=dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inlet)),
            bounds=strm_bnds,
            within=Reals,
        )
        m.s01.fix()
        m.s02 = Var(
            m.props,
            initialize=dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inlet)),
            bounds=strm_bnds,
            within=Reals,
        )  ##purge
        m.s03 = Var(
            m.props,
            initialize=dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inlet)),
            bounds=strm_bnds,
            within=Reals,
        )  ##recycle stream
        m.s02[3].value = m.s01[3].value * purge_ratio
        m.s03[3].value = m.s01[3].value * (1 - purge_ratio)
        m.s04 = Var(
            m.props,
            initialize=dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inlet)),
            bounds=strm_bnds,
            within=Reals,
        )  ##mix stream
        m.s05 = Var(
            m.props,
            initialize=dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], mix_stream)),
            bounds=strm_bnds,
            within=Reals,
        )  ##mix stream
        m.s05.fix()
        m.s04[2].value = m.s05[2].value
        m.s06 = Var(
            m.props,
            initialize=dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], mix_stream)),
            bounds=strm_bnds,
            within=Reals,
        )  ##mixed stream
        m = splitter(m, m.s01, [m.s02, m.s03], purge_ratio)
        m = dP(m, m.s03, m.s04, m.s05[2].value, True)
        m = mix(m, [m.s04, m.s05], m.s06)
        solver = SolverFactory("ipopt")
        solver.options = {"tol": 1e-7, "max_iter": 600}
        results = solver.solve(m, tee=False)
        # m.pprint()
        if results.Solver.status == "ok":
            # return [[(m.s02[i].value-intcp[i-1])/slope[i-1] for i in m.props],[(m.s03[i].value-intcp[i-1])/slope[i-1] for i in m.props],[(m.s04[i].value-intcp[i-1])/slope[i-1] for i in m.props],[(m.s06[i].value-intcp[i-1])/slope[i-1] for i in m.props]]
            return [
                [m.s02[i].value for i in m.props],
                [m.s03[i].value for i in m.props],
                [m.s04[i].value for i in m.props],
                [m.s06[i].value for i in m.props],
            ]
        else:
            print("rec bad")
            return [inlet, inlet, inlet, mix_stream]
    except:
        print("rec error")
        return [inlet, inlet, inlet, mix_stream]
