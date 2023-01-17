"""
Credit to my supervisor for the following code to solve an ethylene oxide process flowsheet.
"""


import time

import numpy as np
import pandas as pd

from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.network import *

from pyomo_models import *


feed = [298, 1e5, 5, 0.99995, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, -86.1876894]
empty = [298, 1e5, 0, 0.16, 0.16, 0.16, 0.16, 0.16, 0.17, 0]
fdval = {}
emptval = {}
for i in range(len(feed)):
    fdval[i + 1] = feed[i]
    emptval[i + 1] = empty[i]


def create_m(pdf, stream_cnt):
    m = start_episode()
    m.s01 = Var(m.props, initialize=fdval, bounds=strm_bnds, within=Reals)
    m.s01.fix()
    for s in range(1, stream_cnt):
        ln = compile(
            "m.s{:02d}=Var(m.props, initialize=emptval,bounds=strm_bnds, within=Reals)".format(
                s + 1
            ),
            "<string>",
            "exec",
        )
        exec(ln)
    for i in pdf.index:
        if pdf.iloc[i]["action"] == "addAir":
            ln = compile(
                "m=addAir(m,m.s{:02d},m.s{:02d}, {})".format(
                    pdf.iloc[i]["inlet"],
                    pdf.iloc[i]["outlets"][0],
                    pdf.iloc[i]["param"],
                ),
                "<string>",
                "exec",
            )
            exec(ln)
            solver = SolverFactory("ipopt")
            solver.options = {
                "tol": 1e-8,
                "max_iter": 200,
            }  # )bound_relax_factor':1e-4}
            results = solver.solve(m, tee=False)
        elif pdf.iloc[i]["action"] == "HX":
            ln = compile(
                "m=HX(m,m.s{:02d},m.s{:02d}, {})".format(
                    pdf.iloc[i]["inlet"],
                    pdf.iloc[i]["outlets"][0],
                    pdf.iloc[i]["param"],
                ),
                "<string>",
                "exec",
            )
            exec(ln)
        elif pdf.iloc[i]["action"] == "rxn":
            ln = compile(
                "m=PFR(m,m.s{:02d},m.s{:02d},{})".format(
                    pdf.iloc[i]["inlet"],
                    pdf.iloc[i]["outlets"][0],
                    pdf.iloc[i]["param"],
                ),
                "<string>",
                "exec",
            )
            exec(ln)
        elif pdf.iloc[i]["action"] == "VLE":
            ln = compile(
                "m=VLE(m,m.s{:02d},[m.s{:02d},m.s{:02d}],{})".format(
                    pdf.iloc[i]["inlet"],
                    pdf.iloc[i]["outlets"][0],
                    pdf.iloc[i]["outlets"][1],
                    pdf.iloc[i]["param"],
                ),
                "<string>",
                "exec",
            )
            exec(ln)
        elif pdf.iloc[i]["action"] == "GLE":
            ln = compile(
                "m=GLE(m,m.s{:02d},[m.s{:02d},m.s{:02d}],{})".format(
                    pdf.iloc[i]["inlet"],
                    pdf.iloc[i]["outlets"][0],
                    pdf.iloc[i]["outlets"][1],
                    pdf.iloc[i]["param"],
                ),
                "<string>",
                "exec",
            )
            exec(ln)
        elif pdf.iloc[i]["action"] == "dP":
            ln = compile(
                "m=dP(m,m.s{:02d},m.s{:02d},{})".format(
                    pdf.iloc[i]["inlet"],
                    pdf.iloc[i]["outlets"][0],
                    pdf.iloc[i]["param"],
                ),
                "<string>",
                "exec",
            )
            exec(ln)
        elif pdf.iloc[i]["action"] == "rec":
            ### run m=splitter(m,inlet, [purge, recycle]), purge_ratio)
            ln = compile(
                "m=splitter(m,m.s{:02d},[m.s{:02d},m.s{:02d}],{})".format(
                    pdf.iloc[i]["inlet"],
                    pdf.iloc[i]["outlets"][0],
                    pdf.iloc[i]["outlets"][1],
                    pdf.iloc[i]["param"][1],
                ),
                "<string>",
                "exec",
            )
            exec(ln)
            ## change recyclce stream pressure to be equal to mix stream
            ln = compile(
                "m=dP(m,m.s{:02d},m.s{:02d},m.s{:02d}[2])".format(
                    pdf.iloc[i]["outlets"][1],
                    pdf.iloc[i]["outlets"][2],
                    pdf.iloc[i]["param"][0],
                ),
                "<string>",
                "exec",
            )
            exec(ln)
            ### run m=mix(m,[mix_stream, recycle], mixed)
            ln = compile(
                "m=mix(m,[m.s{:02d},m.s{:02d}],m.s{:02d})".format(
                    pdf.iloc[i]["param"][0],
                    pdf.iloc[i]["outlets"][2],
                    pdf.iloc[i]["outlets"][3],
                ),
                "<string>",
                "exec",
            )
            exec(ln)
            ### this expects ann already updated pdf, with the recycle stream present.
        elif pdf.iloc[i]["action"] == "sink":
            pass
    return m


def solve_m(m, prod_stream):
    reward = -1  ## value if failed.
    t = 1
    try:  # m.s03[3].value=0.46; m.s03[4].value=0.54; m.s03[5].value=0; m.s03[6].value=0; m.s04[2].value=22;
        t0 = time.time()
        solver = SolverFactory("ipopt")
        solver.options = {"tol": 1e-5, "max_iter": 500}
        results = solver.solve(m, tee=True)
        print("convergence", time.time() - t0)
        # m.pprint() # m.s04.pprint()
        t0 = time.time()
        for flux in m.control_dict.values():  ###unfix all decision vars for opt,
            unfix_line = compile("m.{}.unfix()".format(flux.value), "<string>", "exec")
            exec(unfix_line)
        # print([i.value for i in m.control_dict.values()])
        # m.c=Constraint(expr=prod_stream[6]>=0.95)
        m.obj = Objective(expr=prod_stream[6], sense=maximize)
        # def Elim(m,t): return  sum(m.n[comp,t] for comp in m.components) >= 10 * m.n[1,t]
        # m.c_explsv=Constraint(m.t,rule=Elim)
        # m.obj=Objective(expr=prod_stream[6]/prod_stream[8]/(m.s04[8]/m.s04[6]), sense=maximize)
        solver = SolverFactory("ipopt")
        solver.options = {"tol": 1e-5, "max_iter": 700, "bound_relax_factor": 1e-5}
        results = solver.solve(m, tee=True)
        print("optimization", time.time() - t0)
        # print(value(m.obj), results.Solver.status)
        if results.Solver.status == "ok":
            reward = value(m.obj)
        t = np.round(results.Solver.Time, 2)
    except:
        pass
    return [m, reward, time]


#### START HERE: ####

### solving purification section
t0 = time.time()
units = 6  ##number of units involved
s_cnt = 17  ## number of streams involved
### parameters for each unit:
GLE_w = 0.2
Po = 2e5
VLE1_vf = 0.2
VLE2_vf = 0.0
recy = [5, 0.9]  # recyle stream num & purge ratio
### build process dataframe (pdf), and control connections
pdf = pd.DataFrame(
    index=range(0, units), columns=("action", "inlet", "outlets", "param")
)
## outlet[0] is vapor, outlet[1] is liq... for recycling: outlet[0] is purge, outlet[1] is recy
pdf.iloc[0][:] = np.array(["GLE", 17, [3, 4], GLE_w], dtype=object)
pdf.iloc[1][:] = np.array(["dP", 4, [5], Po], dtype=object)
pdf.iloc[2][:] = np.array(["VLE", 13, [6, 7], VLE1_vf], dtype=object)
pdf.iloc[3][:] = np.array(["VLE", 6, [8, 9], VLE2_vf], dtype=object)
pdf.iloc[4][:] = np.array(["rec", 9, [10, 11, 12, 13], recy], dtype=object)
pdf.iloc[5][:] = np.array(["rec", 7, [14, 15, 16, 17], [2, 0.9]], dtype=object)
m = create_m(pdf, s_cnt)
## initialize values to either by solving or guess... here we use solving
feed_props = np.array([298, 2e6, 20, 0.482, 0.071, 0.024, 0.059, 0.013, 0.35])
feed_props = (findH((feed_props - intcp[:-1]) / slope[:-1]) - intcp) / slope
# feed_props=(np.array([4.836e+02,1.0e+05,5,8.00052e-02,1e-05,1.7e-01,1e-05,7.49965e-01,1e-05,2.19638326e+05])-intcp)/slope
s3, s4 = solve_GLE(feed_props, GLE_w)
s5 = solve_dP(s4, Po)[0]
outv1, outl1 = solve_VLE(s4, VLE1_vf)
outv2, outl2 = solve_VLE(outv1, VLE2_vf)
# purge1,recycle1,recycle_p1,mixed1=solve_rec(outl2,s5,0.99)
# purge2,recycle2,recycle_p2,mixed2=solve_rec(s16,feed_props,0.99)
# print('{}\n{}\n{}\n{}\n'.format(outl2,purge,recycle,mixed))
for i in m.props:
    m.s02[i].value = feed_props[i - 1] * slope[i - 1] + intcp[i - 1]
    m.s03[i].value = s3[i - 1] * slope[i - 1] + intcp[i - 1]
    m.s04[i].value = s4[i - 1] * slope[i - 1] + intcp[i - 1]
    m.s05[i].value = s5[i - 1] * slope[i - 1] + intcp[i - 1]
    m.s06[i].value = outv1[i - 1] * slope[i - 1] + intcp[i - 1]
    m.s07[i].value = outl1[i - 1] * slope[i - 1] + intcp[i - 1]
    m.s08[i].value = outv2[i - 1] * slope[i - 1] + intcp[i - 1]
    m.s09[i].value = outl2[i - 1] * slope[i - 1] + intcp[i - 1]
    # m.s10[i].value=purge1[i-1]*slope[i-1]+intcp[i-1]
    # m.s11[i].value=recycle1[i-1]*slope[i-1]+intcp[i-1]
    # m.s12[i].value=mixed1[i-1]*slope[i-1]+intcp[i-1]
    # m.s13[i].value=purge2[i-1]*slope[i-1]+intcp[i-1]
    # m.s14[i].value=recycle2[i-1]*slope[i-1]+intcp[i-1]
    # m.s15[i].value=mixed2[i-1]*slope[i-1]+intcp[i-1]
    # m.s16[i].value=s16[i-1]*slope[i-1]+intcp[i-1]
m.s02.fix()
## solve model & choose stream for which to optimize EO fraction
t00 = time.time()
m, profit, solve_time = solve_m(m, m.s08)
print("solve_m time:", time.time() - t00)
# m.pprint()
# for v in m.component_data_objects(Var, active=True):
#     print(v, value(v))  # doctest: +SKIP
print("optimal EO produced: ", profit)
print(
    "Simulation time: {}s, optimizer time: {}s".format(
        np.round(time.time() - t0, 2), solve_time
    )
)
# # for v in m.s02:
# #     print(['T','P','n','E','O','EO','C','W','N','H'][v-1], value(m.s05[v]),'      ', value(m.s04[v]))  # doctest: +SKIP

print(pdf)
