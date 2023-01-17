"""
Credit to my supervisor for the following code to optimise an ethylene oxide process flowsheet.

This is the env where LL makes decisions, it should keep adding units to the same m model, 
this requires we tell it how to attached each unit's constraints to m
"""

import gym
import numpy as np
import pandas as pd
from gym import error, spaces, utils
from gym.utils import seeding
import time
import random
from pyomo.environ import *
from pyomo.opt import SolverFactory

from pyomo_models import *
import logging

logging.getLogger("pyomo.core").setLevel(logging.CRITICAL)
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

## this slope and intercept are used to normalize stream,m properties.
def strm_bnds(m, i):
    lb = {1: 298, 2: 1e5, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: -1e5}
    ub = {1: 530, 2: 2e6, 3: 200, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1e7}
    return (lb[i], ub[i])


def h_bnds(m, i):  ## compnent enthalpy bounds
    lb = {1: -100, 2: -100, 3: -100, 4: -100, 5: -100, 6: -100}
    ub = {1: 2e5, 2: 2e5, 3: 2e5, 4: 2e5, 5: 2e5, 6: 2e5}
    return (lb[i], ub[i])


def hpfr_bnds(m, i, t):  ## compnent enthalpy bounds
    lb = {1: -100, 2: -100, 3: -100, 4: -100, 5: -100, 6: -100}
    ub = {1: 2e5, 2: 2e5, 3: 2e5, 4: 2e5, 5: 2e5, 6: 2e5}
    return (lb[i], ub[i])


## this slope and intercept are used to normalize stream,m properties.
# obs_slope=np.array([232,1.9e6, 200,.99994,.99994,.99994,.99994,.99994,.99994,1.01e7])
# obs_intcp=np.array([298,  1e5,   0,  1e-5,  1e-5,  1e-5,  1e-5,  1e-5,  1e-5,  -1e5])
obs_slope = np.array([232, 1.9e6, 200, 1, 1, 1, 1, 1, 1, 1.01e7])
obs_intcp = np.array([298, 1e5, 0, 0, 0, 0, 0, 0, 0, -1e5])

# feed =[298,2e6,1,0.9995,1e-5,1e-5,1e-5,1e-5,1e-5,-43.094768]
# empty=[298,1e6,0,  0.16,0.16,0.16,0.16,0.16,0.17, 0]
# fdval={}
# emptval={}
# for i in range(len(feed)):
#     fdval[i+1]= feed[i]
#     emptval[i+1]= empty[i]


class LLEOEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        # make action space discrete: [HX, rxn, VLE, GLE, Add(O2), dP, sink]
        self.param_segments = 10
        self.action_space = spaces.MultiDiscrete([7, self.param_segments + 1])
        self.observation_space = spaces.Box(
            low=np.float32(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
            high=np.float32(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])),
            dtype=np.float32,
        )
        self.obs_slope = np.array([232, 1.9e6, 200, 1, 1, 1, 1, 1, 1, 1.01e7, 1])
        self.obs_intcp = np.array(
            [298, 1e5, 0, 0, 0, 0, 0, 0, 0, -1e5, 0]
        )  # np.array([298,  1e5,  0,0,0,0,0,0,0,  -1e5,0])
        self.max_actions = 10  # number of sections
        self.seed()
        self.feed_method = "set"
        # self.sec_feed=self.get_feed(self.feed_method) ### we use this in HLenv to define the section feed
        self.feed = self.get_feed(self.feed_method)
        self.state = self.feed  # current state
        self.sg = random.choice([0])
        self.state[10] = self.sg
        self.stream_cnt = 1  # number of streams so far
        self.stream_cnt_checkpoint = 1  ##checkpoint to connect sections together
        self.key_path = [1]  # list of key_path streams for a section
        self.key_props = self.feed[
            :-1
        ]  # list of key_path stream properties for a section
        self.sf_num = 1  ##identity of new section feed
        self.sf = self.feed  ##define section feed
        self.product_stream = None  ##identified product stream to be optimized
        self.CO2_stream = None  ##identified CO2 stream to be optimized
        self.open_streams = []  ##streams sent to HL for sg selection
        self.open_props = []  ##open_streams properties
        self.act_cnt = 0  # action counter per section
        self.action_cnt_checkpoint = 0  ##action count checkpoint to connect sections
        self.last_conc = 0
        # self.HL=HighLevel(50)
        self.m = start_episode()
        self.m.s01 = Var(
            self.m.props,
            initialize=dict(
                zip(
                    range(1, 11),
                    self.feed[:-1] * self.obs_slope[:-1] + self.obs_intcp[:-1],
                )
            ),
            bounds=strm_bnds,
            within=Reals,
        )
        self.m.s01.fix()
        self.secm = self.m
        self.recm = self.m
        self.reward = 0
        self.pdf_state = None
        ##reporting counters
        self.max_r = -1
        self.profitable_times = 0
        self.profitable_secs = 0
        self.profit = -1
        self.n_steps = 0  ##step counter for agent to train
        self.rep = 2
        self.r_scale = [1, 5, 10, 20, 50, 100]
        self.done = False
        self.episodes = 0
        self.sections = 0
        self.avg_r100 = 0
        self.counter = 0
        self.max_steps = 50000
        self.time = time.time()
        self.pdf = pd.DataFrame(
            index=range(0, 0),
            columns=(
                "action",
                "inlet",
                "outlets",
                "param",
                "c_in",
                "c_out",
                "inprop",
                "outprop",
            ),
            dtype=object,
        )
        self.pdf_len = 0
        self.sdf = pd.DataFrame(
            index=range(self.pdf_len, self.pdf_len + self.max_actions - 1),
            columns=(
                "action",
                "inlet",
                "outlets",
                "param",
                "c_in",
                "c_out",
                "inprop",
                "outprop",
            ),
        )
        # print(self.HL.update_sgdict())
        # a=self.HL.get_next_sg()
        # print(a)
        # self.revise_sg()

    def get_feed(self, method):
        if method == "set":
            # return (np.array([298,2e6,2,0.99995,1e-5,1e-5,1e-5,1e-5,1e-5,-43.0948126793,0])-self.obs_intcp)/self.obs_slope
            ### below is if we want 0.05 CO2 in feed cuz of PFR model
            return (
                np.array(
                    [
                        298,
                        2e6,
                        1,
                        0.94996,
                        1e-5,
                        1e-5,
                        0.05,
                        1e-5,
                        1e-5,
                        -40.8164241348 / 5,
                        0,
                    ]
                )
                - self.obs_intcp
            ) / self.obs_slope
            # return np.array([1.   , 0.998 ,0.079 ,0.083, 0.148, 0.037, 0.017, 0.014, 0.702, 0.022, 0.   ])
            # return np.array([0.99993, 0.99832, 0.07861, 0.08328, 0.14772, 0.03691, 0.01677, 0.01359, 0.70173, 0.02207, 1.     ])

    def stream_dict(self, stream):
        stmdct = {}
        for i in range(len(stream)):
            stmdct[i + 1] = stream[i] * obs_slope[i] + obs_intcp[i]
        return stmdct

    def report_stream(self, stream):
        rep = stream * self.obs_slope + self.obs_intcp
        return [np.round(v, 1) for v in rep]

    def test_reset(self):
        self.state = self.reset()
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # self.feed=self.get_feed(self.feed_method)#,self.rand_feed()
        self.state = self.sf  # current state sf
        self.stream_cnt = self.stream_cnt_checkpoint  # number of streams so far
        self.key_path = [self.sf_num]  # list of key_path streams
        self.key_props = self.sf[:-1]
        self.open_streams = []
        self.open_props = []
        self.act_cnt = 0  # action counter, needs to reset after every section
        self.done = False
        self.secm = self.m.clone()
        self.sdf = pd.DataFrame(
            index=range(self.pdf_len, self.pdf_len + self.max_actions - 1),
            columns=(
                "action",
                "inlet",
                "outlets",
                "param",
                "c_in",
                "c_out",
                "inprop",
                "outprop",
            ),
        )
        return self.state

    def reset_process(self):
        self.stream_cnt_checkpoint = 1
        self.sf_num = 1  ##identity of new section feed
        self.sf = self.feed
        self.HL.product_stream = None
        self.product_stream = None  ## should reset when full process is done
        self.CO2_stream = None
        self.action_cnt_checkpoint = 0
        self.profit = -1
        self.act_cnt = 0  # action counter
        self.sections = 0
        self.pdf = pd.DataFrame(
            index=range(0, 0),
            columns=(
                "action",
                "inlet",
                "outlets",
                "param",
                "c_in",
                "c_out",
                "inprop",
                "outprop",
            ),
            dtype=object,
        )
        self.pdf_len = 0
        self.HL.open_streams = {1: self.feed[:-1]}
        self.m = start_episode()
        self.m.s01 = Var(
            self.m.props,
            initialize=dict(
                zip(
                    range(1, 11),
                    self.feed[:-1] * self.obs_slope[:-1] + self.obs_intcp[:-1],
                )
            ),
            bounds=strm_bnds,
            within=Reals,
        )
        self.m.s01.fix()
        self.secm = self.m
        self.recm = self.m

        # self.reset()

    def revise_sg(self):
        # profit = solve_process() ## if all sgs have been solved, do we need to solve full process again? maybe just update TDG when finished.
        ##reset HL.reset: should update HL.sf_num and HL.free_sf to feed only
        # print('self.sf in revise_sg',self.sf[:-1], 'open streams', self.open_props)
        self.HL.update_TDG(
            self.sf_num,
            self.sf[:-1],
            self.sg,
            self.profit,
            self.key_path[-1],
            self.key_props,
            self.open_streams,
            self.open_props,
            self.product_stream,
        )
        if [
            item for sublist in self.HL.sg_dict.values() for item in sublist
        ] == []:  ##process finished
            print("empty sg_dict so reset process, ", self.HL.sg_dict.values())
            self.reset_process()  ##gotta make sure having the feed as s' of the last sf_num is not an issue. If it is then just use same stream as s', then reset process
            self.HL.update_sgdict()
            # self.HL.get_next_sg()

        # print('revise_sg, hl.sg_dict:',[item for sublist in self.HL.sg_dict.values() for item in sublist])
        self.sf_num, self.sg, info = self.HL.get_next_sg()
        # if len(self.pdf.index)>0: ## if not new process
        #     sf=self.sf*self.obs_slope[:-1]+self.obs_intcp[:-1]
        #     for i in self.pdf.index: ##specify section feed number from prop
        #         for j in range(len(self.pdf.loc[i,'outprop'])):
        #             print('191:',sf, self.pdf.loc[i]['outprop'][j])
        #             if all(sf == self.pdf.loc[i]['outprop'][j]): self.sf_num=self.pdf.loc[i]['outlets'][j]
        #             else:
        #                 print('issue w finding new sf location', sf)
        #                 print([self.pdf[i]['outprop'][j]]); raise ValueError
        if len(self.pdf.index) > 0:  ## if not new process
            # sf=self.sf*self.obs_slope[:-1]+self.obs_intcp[:-1]
            sf = [
                self.pdf.iloc[i]["outprop"][j]
                for i in self.pdf.index
                for j in range(len(self.pdf.iloc[i]["outlets"]))
                if self.sf_num == self.pdf.iloc[i]["outlets"][j]
            ][0]
            print("revisesg sf:", sf)
            self.sf = (sf - self.obs_intcp[:-1]) / self.obs_slope[:-1]
            # self.sf_num=[self.pdf.iloc[i]['outlets'][j] for i in self.pdf.index for j in range(len(self.pdf.iloc[i]['outlets'])) if abs(sum(self.sf*self.obs_slope[:-1]+self.obs_intcp[:-1]-np.array(self.pdf.iloc[i]['outprop'][j])))<=1e-3]
            # if self.sf_num>max([self.pdf.loc[i]['outlets'][j] for i in self.pdf.index for j in range(len(self.pdf.loc[i,'outprop']))]): print('problem finding sf_num', sf, 'found sf_num=',self.sf_num,'outlets:',[self.pdf.loc[i]['outlets'][j] for i in self.pdf.index for j in range(len(self.pdf.loc[i,'outprop']))]); raise ValueError
        # print('sg:',self.sg,'sf_num:',self.sf_num,'sf:',self.sf)
        self.sg = [0, 1, 3, 4, 2][self.sections]
        if self.sg in [2, 3]:
            self.key_props = None
            self.open_props = None
            self.recycle(self.sf_num, self.sf, info)
            self.revise_sg()
        else:  ## sg == rxn, separation, CC
            self.sf = np.append(self.sf, self.sg)
            # self.state=self.sf  # gotta make sure this doesn't affect agent's learning, as flat agent gets done with s'=s not return to feed
        # print('## rvised sg', self.sg,'sf:', self.sf)

    def recycle(self, rec_stream, rec_prop, component):
        pdf = self.pdf.copy()
        self.recm = self.m.clone()
        stream_cnt = self.stream_cnt
        ### identifying units
        if component == "E":
            unit = "air"
            purge = 0.4  # 1-rec_prop[3]
        elif component == "EO":
            unit = "VLE"
            purge = 0.01  # 1-rec_prop[5]
        # elif component=='W': unit='GLE' ; purge=1-rec_prop[7]
        # print(pdf.index[pdf.action==str(unit)], unit)
        ## identifying mixing stream
        mix_stream_loc = pdf.index[pdf.action == str(unit)][
            0
        ]  # get stream from pdf to mix with, given unit
        mix_stream = pdf.loc[mix_stream_loc]["inlet"]
        mix_prop = pdf.loc[mix_stream_loc]["inprop"]
        pdf.at[mix_stream_loc, "inlet"] = stream_cnt + 4
        ### solving the rec alone
        out_streams = [stream_cnt + 1, stream_cnt + 2, stream_cnt + 3, stream_cnt + 4]
        # print('rec_prop:',rec_prop,'\n purge:',purge, '\n mix_prop:',mix_prop)
        out_prop = solve_rec(
            rec_prop, purge, mix_prop
        )  ##outprop = [purge, rec_non_pump, rec_pump, mixed]
        c_in = [0, 0, 0]
        c_out = [0, 0, 0]
        # print('rec stream:', rec_prop,'out \n',out_prop)
        # print(pdf)
        #### working with m
        # self.recm.conc_const.deactivate(); self.recm.c_explsv.deactivate()
        ## adding rec streams
        # solver = SolverFactory('ipopt'); solver.options = {'tol': 1e-6,'max_iter': 600,'bound_relax_factor':1e-6}; results = solver.solve(self.secm, tee=False)
        # print('##### before open loop :',results.Solver.status)
        for i in np.arange(
            pdf.loc[mix_stream_loc]["c_in"], pdf.loc[mix_stream_loc]["c_out"]
        ):
            ln = compile("self.recm.c_{}.deactivate".format(i), "<string>", "exec")
            exec(ln)
        for j in range(len(out_streams)):
            ln1 = compile(
                "self.recm.s{:02d}=Var(self.recm.props, initialize=dict(zip(range(1,11),out_prop[j])),bounds=strm_bnds, within=Reals)".format(
                    out_streams[j]
                ),
                "<string>",
                "exec",
            )
            exec(ln1)
        ln = compile(
            "self.recm=splitter(self.recm,self.recm.s{:02d},[self.recm.s{:02d},self.recm.s{:02d}],{})".format(
                rec_stream, out_streams[0], out_streams[1], purge
            ),
            "<string>",
            "exec",
        )
        exec(ln)
        # solver = SolverFactory('ipopt'); solver.options = {'tol': 1e-6,'max_iter': 600, 'bound_relax_factor':1e-6}; results = solver.solve(self.secm, tee=False)
        # print('##### open loop splitter:',results.Solver.status)
        # self.recm.pprint()
        c_in[0] = self.recm.old_constr_cnt.value
        c_out[0] = self.recm.constr_cnt.value
        ln = compile(
            "self.recm=dP(self.recm,self.recm.s{:02d},self.recm.s{:02d},self.recm.s{:02d}[2], True)".format(
                out_streams[1], out_streams[2], mix_stream
            ),
            "<string>",
            "exec",
        )
        exec(ln)
        # solver = SolverFactory('ipopt'); solver.options = {'tol': 1e-6,'max_iter': 600,'bound_relax_factor':1e-6}; results = solver.solve(self.secm, tee=False)
        # print('##### open loop dP:',results.Solver.status)
        c_in[1] = self.recm.old_constr_cnt.value
        c_out[1] = self.recm.constr_cnt.value
        ln = compile(
            "self.recm=mix(self.recm,[self.recm.s{:02d},self.recm.s{:02d}],self.recm.s{:02d})".format(
                out_streams[2], mix_stream, out_streams[3]
            ),
            "<string>",
            "exec",
        )
        exec(ln)
        # solver = SolverFactory('ipopt'); solver.options = {'tol': 1e-6,'max_iter': 600,'bound_relax_factor':1e-6}; results = solver.solve(self.secm, tee=False)
        # print('##### open loop mix:',results.Solver.status)
        c_in[2] = self.recm.old_constr_cnt.value
        c_out[2] = self.recm.constr_cnt.value
        pdf.loc[self.pdf_len] = [
            "rec",
            rec_stream,
            out_streams,
            [purge, mix_stream],
            c_in,
            c_out,
            rec_prop,
            out_prop,
        ]
        # solver = SolverFactory('ipopt'); solver.options = {'tol': 1e-7, 'max_iter': 300}; results = solver.solve(self.secm, tee=False)
        # print('##### open loop:',results.Solver.status)
        # if unit=='VLE':
        #     self.recm.pprint()
        ## updating mix_stream unit
        if unit == "air":
            ln2 = compile(
                "self.recm=air(self.recm,self.recm.s{:02d},self.recm.s{:02d},{})".format(
                    stream_cnt + 4,
                    pdf.loc[mix_stream_loc]["outlets"][0],
                    pdf.loc[mix_stream_loc]["param"][0],
                ),
                "<string>",
                "exec",
            )
            exec(ln2)
            solved_prop = solve_air(out_prop[3], pdf.at[mix_stream_loc, "param"][0])
        elif unit == "VLE":
            ln2 = compile(
                "self.recm=VLE(self.recm,self.recm.s{:02d},[self.recm.s{:02d},self.recm.s{:02d}],{})".format(
                    stream_cnt + 4,
                    pdf.loc[mix_stream_loc]["outlets"][0],
                    pdf.loc[mix_stream_loc]["outlets"][1],
                    pdf.loc[mix_stream_loc]["param"][0],
                ),
                "<string>",
                "exec",
            )
            exec(ln2)
            solved_prop = solve_VLE(out_prop[3], pdf.at[mix_stream_loc, "param"][0])
        for out in range(len(solved_prop)):  ##num of streams
            for i in range(len(solved_prop[0])):  ## props
                ln2 = compile(
                    "self.recm.s{:02d}[i+1].value=(solved_prop[out][i]*2)/2".format(
                        pdf.at[mix_stream_loc, "outlets"][out]
                    ),
                    "<string>",
                    "exec",
                )
                exec(ln2)
        pdf.at[mix_stream_loc, "c_in"] = self.recm.old_constr_cnt.value
        pdf.at[mix_stream_loc, "c_out"] = self.recm.constr_cnt.value
        # try: ##solving m
        t = time.time()
        # for cnt in range(self.recm.heat_constr_cnt.value-1):
        #     ln=compile("self.rec{}.deactivate()".format(self.recm.heat_constr[cnt+1].value),'<string>','exec');exec(ln)
        # self.recm.obj.deactivate()
        # self.recm.conc_const.deactivate(); self.recm.c_explsv.deactivate()
        solver = SolverFactory("ipopt")
        # solver.options = {'tol': 1e-2, 'max_iter': 1, 'bound_relax_factor':1e-2}
        # results = solver.solve(self.recm, tee=False)
        solver.options = {"tol": 1e-7, "max_iter": 200, "bound_relax_factor": 1e-7}
        results = solver.solve(self.recm, tee=False)
        print("##### lvl 1:", results.Solver.status)
        # for cnt in range(self.recm.heat_constr_cnt.value-1):
        #     ln=compile("self.rec{}.activate()".format(self.recm.heat_constr[cnt+1].value),'<string>','exec');exec(ln)
        # solver = SolverFactory('ipopt')
        # solver.options = {'tol': 1e-7, 'max_iter': 180, 'bound_relax_factor':1e-7}
        # results = solver.solve(self.recm, tee=False)
        for (
            flux
        ) in self.recm.control_dict.values():  ###unfix all decision vars for opt,
            unfix_line = compile(
                "self.recm.{}.unfix()".format(flux.value), "<string>", "exec"
            )
            exec(unfix_line)
        # for cnt in range(self.recm.heat_constr_cnt.value-1):
        #     ln=compile("self.rec{}.deactivate()".format(self.recm.heat_constr[cnt+1].value),'<string>','exec');exec(ln)
        ### define objective as total profit,,, this is a function of product and C2 streams, and opex
        self.recm.obj.deactivate()
        self.recm.conc_const.deactivate()
        self.recm.c_explsv.deactivate()
        self.recm.tot_costs = Var(
            initialize=sum(
                self.recm.cost_dict[i] for i in range(1, value(self.recm.cost_cnt))
            ),
            within=NonNegativeReals,
        )

        def sum_costs(m):
            return m.tot_costs == sum(
                m.cost_dict[i] for i in range(1, value(m.cost_cnt))
            )

        self.recm.C_sum_costs = Constraint(expr=sum_costs)
        obj_ln = compile(
            "self.recm.obj=Objective(expr=self.recm.s{:02d}[3]*self.recm.EO_cost-self.recm.tot_costs, sense=maximize)".format(
                self.product_stream
            ),
            "<string>",
            "exec",
        )
        exec(obj_ln)
        obj_ln1 = compile(
            "self.recm.conc_const=Constraint(expr=self.recm.s{:02d}[6]>=0.95)".format(
                self.product_stream
            ),
            "<string>",
            "exec",
        )
        exec(obj_ln1)
        try:

            def Elim(m, t):
                return (
                    sum(m.n[comp, t] for comp in m.components) >= 10 * m.n[1, t]
                )  # + 10 * m.n[2,t]

            self.recm.c_explsv = Constraint(self.recm.t, rule=Elim)
        except:
            pass
        # self.recm.obj.deactivate(); self.recm.conc_const.deactivate(); self.recm.c_explsv.deactivate()
        # solver = SolverFactory('ipopt')
        # solver.options = {'tol': 1e-5, 'max_iter': 180, 'bound_relax_factor':1e-5}
        # results = solver.solve(self.recm, tee=False)
        # print('##### lvl 2:',results.Solver.status)
        # for cnt in range(self.recm.heat_constr_cnt.value-1):
        #     ln=compile("self.rec{}.activate()".format(self.recm.heat_constr[cnt+1].value),'<string>','exec');exec(ln)
        # self.recm.obj.activate(); self.recm.conc_const.activate(); self.recm.c_explsv.activate()
        solver = SolverFactory("ipopt")
        solver.options = {"tol": 1e-3, "max_iter": 2, "bound_relax_factor": 1e-3}
        results = solver.solve(self.recm, tee=False)
        solver.options = {"tol": 1e-6, "max_iter": 1100, "bound_relax_factor": 1e-6}
        results = solver.solve(self.recm, tee=False)
        # self.recm.pprint()
        # self.recm.obj.deactivate();
        # for flux in self.recm.control_dict.values(): ###unfix all decision vars for opt,
        #     unfix_line=compile('self.recm.{}.fix()'.format(flux.value),'<string>','exec');exec(unfix_line)
        print("#### done rec", results.Solver.status, "####")  #
        print(
            results.Solver.status,
            value(self.recm.obj) * self.r_scale[self.rep],
            "t:",
            round(time.time() - t, 2),
        )
        obj_ln = compile(
            "print('obj: ',round(value(self.recm.obj),2),' n: ',round(self.recm.s{:02d}[3].value/200,3),' EO: ',round(self.recm.s{:02d}[6].value,2),'costs: ',round(self.recm.tot_costs.value,2))".format(
                self.product_stream, self.product_stream
            ),
            "<string>",
            "exec",
        )
        exec(obj_ln)
        # self.recm.s02[3].value=10
        # solver = SolverFactory('ipopt')
        # solver.options = {'tol': 1e-6,'max_iter': 10,'bound_relax_factor':1e-6}
        # results = solver.solve(self.recm, tee=False)
        # print('#### double check rec',results.Solver.status,'####')  #
        # solver.options = {'tol': 1e-6,'max_iter': 600,'bound_relax_factor':1e-6}
        # results = solver.solve(self.recm, tee=False)
        # print('#### triple check rec',results.Solver.status,'####')  #
        # print(results.Solver.status,value(self.recm.obj)*self.r_scale[self.rep],'t:',round(time.time()-t,2))
        if results.Solver.status == "ok" and value(self.recm.obj) > 5e-3:
            reward = value(self.recm.obj) * self.r_scale[self.rep]
            for i in pdf.index:  # update unit inprop and outprops
                try:
                    ln = compile(
                        "pdf.at[i,'param'][0]=self.recm.cont_var{:02d}.value".format(
                            pdf.at[i, "inlet"]
                        ),
                        "<string>",
                        "exec",
                    )
                    exec(ln, locals())
                except:
                    pass
                ln = compile(
                    "pdf.at[i,'inprop']=[self.recm.s{:02d}[prop].value for prop in self.recm.props]".format(
                        pdf.at[i, "inlet"]
                    ),
                    "<string>",
                    "exec",
                )
                exec(ln, locals())
                # ln=compile("df.at[i,'inprop']=[(df.at[i,'inprop'][prop-1]+m.s{:02d}[prop].value)/2 for prop in m.props]".format(df.at[i,'inlet']),'<string>','exec');exec(ln,locals())
                outproplst = []
                for j in pdf.at[i, "outlets"]:
                    ln = compile(
                        "outproplst.append([self.recm.s{:02d}[prop].value for prop in self.recm.props])".format(
                            j
                        ),
                        "<string>",
                        "exec",
                    )
                    exec(ln, locals())
                    # ln=compile("outproplst.append([(df.at[i,'outprop'][df.at[i,'outlets'].index(j)][prop-1]+m.s{:02d}[prop].value)/2 for prop in m.props])".format(j),'<string>','exec');exec(ln,locals())
                pdf.at[i, "outprop"] = outproplst
        else:
            reward = -1
        # except: reward=-0.5; print('recycle error'); pass
        ## update pdf
        if reward > self.profit:
            self.m = self.recm.clone()
            self.sections += 1
            self.pdf_len += 1
            self.stream_cnt_checkpoint += 4
            self.stream_cnt += 4
            self.action_cnt_checkpoint += 1
            self.profit = reward
            self.reward += reward
            self.pdf = pdf
        return value(self.recm.obj)

    def step(self, actions):
        # print(actions,self.state, self.episodes, self.counter)
        discrete = actions[0]
        continuous = actions[1]
        if discrete == 6 or self.act_cnt == self.max_actions:  #'sink':
            # print('action:', discrete, 'act_cnt', self.act_cnt)
            discrete == 6
            sg_stream = self.key_path[-1]
            if self.sg == 1:
                self.product_stream = sg_stream
            elif self.sg == 0.5:
                self.CO2_stream = sg_stream
            self.process_step(sg_stream, self.state[:-1], self.state[-1], discrete, 0)
            self.sdf.dropna(subset=["action"], inplace=True)
            self.secm, self.sdf, self.reward = self.solve_m(
                self.secm, self.sdf, sg_stream
            )
            self.profit = self.reward / self.r_scale[self.rep]
            if self.reward > 0.001:
                self.sdf = self.sdf[self.sdf.action != "sink"]
                self.pdf_len += len(self.sdf)  ## looks iffy
                self.stream_cnt_checkpoint = self.stream_cnt
                self.action_cnt_checkpoint += self.act_cnt
                self.sections += 1
                self.m = self.secm.clone()
                self.pdf = pd.concat([self.pdf, self.sdf])
                self.profitable_secs += 1
                # self.open_props.remove(self.sf)
                # self.open_streams.remove(self.sf_num)
            ### add section to secdf
            # self.revise_sg()
            self.sf = self.state  # for manual trial
            self.sf_num = self.stream_cnt
            self.done = True  ### True after a full process is finished, no? or after all sections are finished
            ###make report:
            self.avg_r100 += self.reward
            logboard = 400
            # if self.episodes%logboard==0:
            #     print(self.profitable_times, '    ',self.episodes,\
            #           '    ', int(time.time()-self.time),'    ', np.round(self.avg_r100/logboard,3),'    ', self.counter,'    ',[self.pdf.loc[i]['action'] for i in self.pdf.index])
            #     self.avg_r100=0
        else:  # other actions....
            inlet = self.key_path[-1]
            if self.logical(discrete) == True:
                self.process_step(
                    inlet, self.state[:-1], self.state[-1], discrete, continuous
                )
                self.act_cnt += 1
                self.counter += 1
            else:
                self.reward = -0.1
        return self.state, self.reward, self.done, {}  # {self.max_r, self.episodes}

    def logical(self, a):
        if self.state[-1] == 0:  ## sg == rxn
            if a in [0, 1, 4, 5, 6]:
                return True
        elif self.state[-1] == 0.5:  ## sg == CO2 removal
            if a in [0, 2, 3, 5, 6]:
                return True
        elif self.state[-1] == 1:  ## sg == sep
            if a in [0, 2, 3, 5, 6]:
                return True
        return False

    def process_step(
        self, inlet, inprop, sg, discrete, param
    ):  ##[HX, rxn, VLE, GLE, Add(O2), dP, sink]
        # print('\n action taken:',discrete, 'param:',np.round(param,2))
        param = (
            param / self.param_segments
        )  ## go from agent's decisions base 20, to parameter base 1
        if discrete == 6:  ## sink
            self.sdf.loc[self.act_cnt + self.pdf_len] = [
                "sink",
                inlet,
                "-",
                0,
                "-",
                "-",
                "-",
                "-",
            ]
        elif discrete == 2 or discrete == 3:
            if discrete == 2:
                action = "VLE"
                param = param * 0.19 + 0.01
            if discrete == 3:
                action = "GLE"
                param = param * 20 + 5
            outlets = [self.stream_cnt + 1, self.stream_cnt + 2]
            # m.pprint()
            ln1a = compile(
                "self.secm.s{:02d}=Var(self.secm.props, initialize=dict(zip(range(1,11),empty*self.obs_slope[:-1]+self.obs_intcp[:-1])),bounds=strm_bnds, within=Reals)".format(
                    self.stream_cnt + 1
                ),
                "<string>",
                "exec",
            )
            exec(ln1a)
            ln1b = compile(
                "self.secm.s{:02d}=Var(self.secm.props, initialize=dict(zip(range(1,11),empty*self.obs_slope[:-1]+self.obs_intcp[:-1])),bounds=strm_bnds, within=Reals)".format(
                    self.stream_cnt + 2
                ),
                "<string>",
                "exec",
            )
            exec(ln1b)
            ln2 = compile(
                "self.secm={}(self.secm,self.secm.s{:02d},[self.secm.s{:02d},self.secm.s{:02d}],{})".format(
                    action, inlet, self.stream_cnt + 1, self.stream_cnt + 2, param
                ),
                "<string>",
                "exec",
            )
            exec(ln2)
            # try:
            solver = SolverFactory("ipopt")
            solver.options = {"tol": 1e-7, "max_iter": 600}
            results = solver.solve(self.secm, tee=False)
            out = [np.zeros(10), np.zeros(10)]
            for i in self.secm.props:
                ln = compile(
                    "out[0][i-1]=(self.secm.s{:02d}[i].value-self.obs_intcp[i-1])/self.obs_slope[i-1]".format(
                        self.stream_cnt + 1
                    ),
                    "<string>",
                    "exec",
                )
                exec(ln)
                ln = compile(
                    "out[1][i-1]=(self.secm.s{:02d}[i].value-self.obs_intcp[i-1])/self.obs_slope[i-1]".format(
                        self.stream_cnt + 2
                    ),
                    "<string>",
                    "exec",
                )
                exec(ln)
            # ln=compile("out=[[(self.secm.s{:02d}[i].value-self.obs_intcp[i-1])/self.obs_slope[i-1] for i in self.secm.props]]".format(self.stream_cnt+1),'<string>','exec');exec(ln)
            try:
                outprop = out
            except:
                outprop = [inprop, inprop]
            if results.Solver.status != "ok":
                self.reward -= 0.05
            # except: return m,[(inlet-self.obs_intcp)/self.obs_slope,(inlet-self.obs_intcp)/self.obs_slope]
            # print('VLE-GLE:',results.Solver.status, 'outprop', np.round(np.array(outprop),2))
            if outprop[0][2] * outprop[0][5] >= outprop[1][2] * outprop[1][5]:
                self.key_path.append(outlets[0])
                self.key_props = outprop[0]
                self.open_streams.append(outlets[1])
                self.open_props.append(outprop[1])
            else:
                self.key_path.append(outlets[1])
                self.key_props = outprop[1]
                self.open_streams.append(outlets[0])
                self.open_props.append(outprop[0])
            inprop = inprop * self.obs_slope[:-1] + self.obs_intcp[:-1]
            outprop = [
                outp * self.obs_slope[:-1] + self.obs_intcp[:-1] for outp in outprop
            ]
            self.sdf.loc[self.act_cnt + self.pdf_len] = [
                action,
                inlet,
                outlets,
                [param],
                self.secm.old_constr_cnt.value,
                self.secm.constr_cnt.value,
                inprop,
                outprop,
            ]
            self.stream_cnt += 2
            self.pdf_state = self.key_path[-1]
        else:  # this is for HX, air, and rxn
            outlets = [self.stream_cnt + 1]
            extra = ""
            if discrete == 0:
                action = "HX"
                param = param * 232 + 298
            elif discrete == 1:
                action = "PFR"
                param = param * 30
            elif discrete == 4:
                action = "air"
                param = param * 10
            elif discrete == 5:
                action = "dP"
                extra = ", False"
                param = param * 1.9e6 + 1e5
            ln1 = compile(
                "self.secm.s{:02d}=Var(self.secm.props, initialize=dict(zip(range(1,11),inprop*self.obs_slope[:-1]+self.obs_intcp[:-1])),bounds=strm_bnds, within=Reals)".format(
                    self.stream_cnt + 1
                ),
                "<string>",
                "exec",
            )
            exec(ln1)
            ln2 = compile(
                "self.secm={}(self.secm,self.secm.s{:02d},self.secm.s{:02d},{} {})".format(
                    action, inlet, self.stream_cnt + 1, param, extra
                ),
                "<string>",
                "exec",
            )
            exec(ln2)
            # if action=='PFR':
            # for cnt in range(self.secm.heat_constr_cnt.value-1):
            #     ln=compile("self.sec{}.deactivate()".format(self.secm.heat_constr[cnt+1].value),'<string>','exec');exec(ln)
            # solver = SolverFactory('ipopt'); solver.options = {'tol': 1e-8, 'max_iter': 450}; results = solver.solve(self.secm, tee=False)
            # for cnt in range(self.secm.heat_constr_cnt.value-1):
            #     ln=compile("self.sec{}.activate()".format(self.secm.heat_constr[cnt+1].value),'<string>','exec');exec(ln)
            solver = SolverFactory("ipopt")
            solver.options = {"tol": 1e-8, "max_iter": 450}
            results = solver.solve(self.secm, tee=False)
            # self.secm.pprint()
            out = np.zeros(10)
            for i in self.secm.props:
                ln = compile(
                    "out[i-1]=(self.secm.s{:02d}[i].value-self.obs_intcp[i-1])/self.obs_slope[i-1]".format(
                        self.stream_cnt + 1
                    ),
                    "<string>",
                    "exec",
                )
                exec(ln)
            try:
                outprop = [out]
            except:
                outprop = [inprop, inprop]
            # print('other units:', results.Solver.status)
            self.key_path.append(outlets[0])
            self.key_props = outprop[0]
            outprop = [
                outp * self.obs_slope[:-1] + self.obs_intcp[:-1] for outp in outprop
            ]
            inprop = inprop * self.obs_slope[:-1] + self.obs_intcp[:-1]
            self.sdf.loc[self.act_cnt + self.pdf_len] = [
                action,
                inlet,
                outlets,
                [param],
                self.secm.old_constr_cnt.value,
                self.secm.constr_cnt.value,
                inprop,
                outprop,
            ]
            self.stream_cnt += 1
            self.pdf_state = self.act_cnt + self.pdf_len
        # print('in',inprop,'\n',action,param,'\n out',outprop)
        # print(np.round(np.multiply(self.state,np.array([227,50,1,1,1,1,5e7]))+np.array([273,0,0,0,0,0,-2.5e-7]),1), action, param)
        # print('outprop:',np.round(outprop,2),'\n inprop',np.round(self.state,2)
        # if self.state[-1]==0:
        self.state = [*self.key_props, self.sg]
        self.reward = self.r_scale[self.rep] * (
            self.state[5] - self.last_conc
            if self.state[5] - self.last_conc > 0.005
            else 0
        )  ##only reward steps where change in EO is considerable.
        try:
            if results.Solver.status != "ok":
                self.reward -= 0.05
        except:
            pass
        # else: reward=-0.04  ## should get updated value from EOYieldEnv
        self.last_conc = self.state[5]
        return

    def render(self, mode="human", close=False):
        pass

    def solve_m(self, m, df, prod_stream):
        reward = 0  # -0.05*len(df)#-fdct[2]*fdct[5]#0 #sol_time=0; ## value if failed.
        # print('solve_m sg:',self.state[-1], 'section:',  df['action'].tolist())
        if self.stream_cnt == self.stream_cnt_checkpoint:
            reward -= 2
        else:  ##solving the model:
            # try:
            # solver = SolverFactory('ipopt')
            # solver.options = {'tol': 1e-7, 'max_iter': 150, 'bound_relax_factor':1e-7}
            # results = solver.solve(m, tee=False)
            ### unfix control variables
            for flux in m.control_dict.values():  ###unfix all decision vars for opt,
                unfix_line = compile(
                    "m.{}.unfix()".format(flux.value), "<string>", "exec"
                )
                exec(unfix_line)
            ### chooose objective
            if self.state[-1] == 0:  ## sg==rxn section:
                obj_ln = compile(
                    "m.obj=Objective(expr=m.s{:02d}[6], sense=maximize)".format(
                        prod_stream, prod_stream
                    ),
                    "<string>",
                    "exec",
                )
                exec(obj_ln)
            elif self.state[-1] == 1:  # sg==purification section
                m.tot_costs = Var(
                    initialize=sum(m.cost_dict[i] for i in range(1, value(m.cost_cnt))),
                    within=NonNegativeReals,
                )

                def sum_costs(m):
                    return m.tot_costs == sum(
                        m.cost_dict[i] for i in range(1, value(m.cost_cnt))
                    )

                m.C_sum_costs = Constraint(expr=sum_costs)
                obj_ln = compile(
                    "m.obj=Objective(expr=m.s{:02d}[3]*m.EO_cost-m.tot_costs, sense=maximize)".format(
                        prod_stream
                    ),
                    "<string>",
                    "exec",
                )
                exec(obj_ln)
                obj_ln1 = compile(
                    "m.conc_const=Constraint(expr=m.s{:02d}[6]>=0.95)".format(
                        prod_stream
                    ),
                    "<string>",
                    "exec",
                )
                exec(obj_ln1)
            elif self.state[-1] == 0.5:  ## sg==CO2 removal section:
                obj_ln = compile(
                    "m.obj=Objective(expr=m.s{:02d}[7]*m.s{:02d}[3], sense=maximize)".format(
                        prod_stream, prod_stream
                    ),
                    "<string>",
                    "exec",
                )
                exec(obj_ln)
                obj_ln1 = compile(
                    "m.conc_const=Constraint(expr=m.s{:02d}[7]>=0.9)".format(
                        prod_stream
                    ),
                    "<string>",
                    "exec",
                )
                exec(obj_ln1)
            try:  ###define explosive limit

                def Elim(m, t):
                    return (
                        sum(m.n[comp, t] for comp in m.components) >= 10 * m.n[1, t]
                    )  # + 10 * m.n[3,t]

                m.c_explsv = Constraint(m.t, rule=Elim)
            except:
                pass
            ### solve optimization
            # for cnt in range(m.heat_constr_cnt.value-1):
            #     ln=compile("{}.activate()".format(m.heat_constr[cnt+1].value),'<string>','exec');exec(ln)
            solver = SolverFactory("ipopt")
            solver.options = {"tol": 1e-6, "max_iter": 1100, "bound_relax_factor": 1e-6}
            results = solver.solve(m, tee=False)
            # m.obj.deactivate()
            try:
                m.conc_const.deactivate()
            except:
                pass
            try:
                m.c_explsv.deactivate()
            except:
                pass
            print("####", results.Solver.status, "####")
            for flux in m.control_dict.values():  ###unfix all decision vars for opt,
                unfix_line = compile(
                    "m.{}.fix()".format(flux.value), "<string>", "exec"
                )
                exec(unfix_line)
            # results = solver.solve(m, tee=False)
            # print('####',results.Solver.status,'####')
            # obj_ln=compile("print(', obj: ',value(m.obj),' flow: ',m.s{:02d}[3].value,', EO: ',m.s{:02d}[6].value)".format(prod_stream,prod_stream),'<string>','exec');exec(obj_ln)
            if results.Solver.status == "ok" and value(m.obj) > 5e-3:
                sol_time = np.round(results.Solver.Time, 2)
                reward += value(m.obj) * self.r_scale[self.rep]
                for i in df.index:  # update unit inprop and outprops
                    try:
                        ln = compile(
                            "df.at[i,'param'][0]=m.cont_var{:02d}.value".format(
                                df.at[i, "inlet"]
                            ),
                            "<string>",
                            "exec",
                        )
                        exec(ln, locals())
                    except:
                        pass
                    ln = compile(
                        "df.at[i,'inprop']=[m.s{:02d}[prop].value for prop in m.props]".format(
                            df.at[i, "inlet"]
                        ),
                        "<string>",
                        "exec",
                    )
                    exec(ln, locals())
                    outproplst = []
                    for j in range(len(df.loc[i]["outlets"])):
                        try:
                            ln = compile(
                                "outproplst.append([m.s{:02d}[prop].value for prop in m.props])".format(
                                    df.loc[i]["outlets"][j]
                                ),
                                "<string>",
                                "exec",
                            )
                            exec(ln, locals())
                        except:
                            pass
                        if df.loc[i]["outlets"][j] == prod_stream:
                            print("sg_stream", prod_stream)
                            self.state[:-1] = (
                                np.array(outproplst[-1]) - self.obs_intcp[:-1]
                            ) / self.obs_slope[:-1]
                    df.at[i, "outprop"] = outproplst
                ln = compile(
                    "self.key_props=[(m.s{:02d}[prop].value-self.obs_intcp[prop-1])/self.obs_slope[prop-1] for prop in m.props]".format(
                        self.key_path[-1]
                    ),
                    "<string>",
                    "exec",
                )
                exec(ln, locals())
                oprops = []
                for i in self.open_streams:
                    ln = compile(
                        "oprops.append([(m.s{:02d}[prop].value-self.obs_intcp[prop-1])/self.obs_slope[prop-1] for prop in m.props])".format(
                            i
                        ),
                        "<string>",
                        "exec",
                    )
                    exec(ln, locals())
                self.open_props = oprops
            else:
                reward -= 0.2
            if df.iloc[-1]["action"] == "sink" and reward <= 0:
                reward -= 0.1  ##update valeus from EOYieldEnv
            elif df.iloc[-1]["action"] == "sink" and reward > 0:
                reward += 3
            print("obj:", value(m.obj), "reward:", reward)
            # except: reward-=0.5; pass
        return m, df, reward


def solve_section(inlet, ops):
    ## full path solution w 3 recyclings:
    env = LLEOEnv()
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
