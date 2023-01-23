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
        t0 = time.time()
        for flux in m.control_dict.values():  ###unfix all decision vars for opt,
            unfix_line = compile("m.{}.unfix()".format(flux.value), "<string>", "exec")
            exec(unfix_line)
        m.obj = Objective(expr=prod_stream[6], sense=maximize)
        solver = SolverFactory("ipopt")
        solver.options = {"tol": 1e-5, "max_iter": 700, "bound_relax_factor": 1e-5}
        results = solver.solve(m, tee=True)
        print("optimization", time.time() - t0)
        if results.Solver.status == "ok":
            reward = value(m.obj)
        t = np.round(results.Solver.Time, 2)
    except:
        pass
    return [m, reward, time]

