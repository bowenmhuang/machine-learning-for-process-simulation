'''
https://reader.elsevier.com/reader/sd/pii/S1385894712009205?token=E0238D3455B6F138F4DBFFC338E34550F10778CC924DF9611E655C171EE7C781F2848ABC5CCF3A081EDFD418D3737BF3EO
'''
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=RuntimeWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
from scipy.integrate import solve_ivp
from sklearn.ensemble import RandomForestRegressor
from itertools import product
from pyDOE import lhs
from skopt.space import Space
from skopt.sampler import Lhs
# from skopt.sampler import Lhs
# from skopt.space import Space
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,WhiteKernel
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Activation
from keras.models import Sequential
from tensorflow.keras import layers  
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
import matplotlib.font_manager as font_manager
import seaborn as sns

# font = font_manager.FontProperties(family='Helvetica',
#                                    style='normal', size=8)
# matplotlib.rcParams['font.serif'] = "Times New Roman"
# matplotlib.rcParams['font.family'] = "serif"
# plt.rcParams["font.family"] = "Computer Modern Roman"

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

T_l,T_h=350,530
C2H4_l,C2H4_h=0,10
O2_l,O2_h=0,10
Q_l,Q_h=-1,1
cat_l,cat_h=0,15
#T,P,C2H4,O2,EO,CO2,H2O
slope=np.array([180,12*1e5,10,10,10,10,10])
intcp=np.array([350,10*1e5, 0, 0, 0, 0, 0])
slope2=np.array([2,15])
intcp2=np.array([-1,0])
# [C2H4,O2,EO,CO2,H2O]
C_pT=[0.088, -0.0019, 8.0473, 0.0373, 0.0092]
C_p0=[42.9, 27.1, 43.71, 37.6, 75.4]
Mr = [28.05, 32, 44.05, 44.01, 18.015]

#Stream class has attributes T,P and component v_mol
class Stream():
    def __init__(self,T,P, *v_mol):
        self.T=T
        self.P=P
        self.v_mol=v_mol
    @property
    def list_prop(self):
        return [self.T, self.P, *self.v_mol]  

#UnitOp class takes in inlet (a Stream) and fluxes [Q/MJ,cat/kg]
class UnitOp():
    def __init__(self,inlet,fluxes):
        self.inlet_prop=np.array(inlet.list_prop)*slope+intcp
        self.fluxes=np.array(fluxes)*slope2+intcp2
        self.outlet_prop=np.array(inlet.list_prop)*slope+intcp #initialize it as the inlet
        self.cost=0
    def model(self,dx,model_inlet_prop):
        T,P,nC2H4,nO2,nEO,nCO2,nH2O = model_inlet_prop[0:7]
        v_mol = [nC2H4,nO2,nEO,nCO2,nH2O]
        totmol = np.sum(np.array(v_mol))
        molfrac = np.array(v_mol/totmol)
        p = P*np.array(molfrac)
        if T<530 and T>350:
            Cp = np.sum((np.array(C_pT)*T+np.array(C_p0))*np.array(v_mol))/(totmol+1e-5)
            k1 = np.exp(-4.087-43585.7/(8.3145*T))
            k2 = np.exp(3.503-77763.2/(8.3145*T))
            K1 = np.exp(-16.644+18321/(8.3145*T))
            K2 = np.exp(-14.823+34660.6/(8.3145*T))
            r1 = self.fluxes[1]*k1*p[0]*p[1]/(1+K1*p[1]+K2*p[1]**(1/2)*p[2])
            r2 = self.fluxes[1]*k2*p[0]*p[1]**(1/2)/(1+K1*p[1]+K2*p[1]**(1/2)*p[2])
            deltaH1 = 106.7*1e3
            deltaH2 = 1323*1e3
            # C2H4 + 0.5*O2 -> EO 
            # C2H4 + 3*O2 -> 2*CO2 + 2*H2O
            # r is in mol/(kg cat s)
            prop_step_change = np.array([(deltaH1*r1+deltaH2*r2+self.fluxes[0]*1e6)/(Cp*totmol), 0, -r1-r2, -0.5*r1-3*r2, r1, 2*r2, 2*r2])
            return prop_step_change
        else:
            return np.array([0, 0, 0, 0, 0, 0, 0])
    def calc_outlet(self):
        dx = (0,1)
        sol = solve_ivp(self.model,dx,self.inlet_prop)
        # results=sol.y
        # xvalues=np.linspace(0,1,len(results[0]))
        # plt.plot(xvalues, (results[0]-350)/180)
        # plt.plot(xvalues, (results[1]-10*1e5)/(12*1e5))
        # plt.plot(xvalues, results[2]/10)
        # plt.plot(xvalues, results[3]/10)
        # plt.plot(xvalues, results[4]/10)
        # plt.plot(xvalues, results[5]/10)
        # plt.plot(xvalues, results[6]/10)
        self.outlet_prop = sol.y[:,-1]
        return self.outlet_prop
    def results(self):
        self.calc_outlet()
        return [(self.outlet_prop[0]-intcp[0])/slope[0],*(self.outlet_prop[2:7]-intcp[2:7])/slope[2:7]]

def gen_random_data(pts):
    t0=time.process_time()
    inlets = []
    outlets = []
    for i in range(pts):
        T=np.random.uniform(0,1); P=0.5; C2H4=np.random.uniform(0,1); O2=np.random.uniform(0,1); Q=np.random.uniform(0,1); cat=np.random.uniform(0,1)
        s=[T,P,C2H4,O2,0,0,0]
        result = UnitOp(Stream(*s),[Q,cat]).results()
        inlets.append([T,C2H4,O2,Q,cat])
        outlets.append(result)
    with open('random_inlets_{}.pkl'.format(pts),'wb') as f:
        pickle.dump(inlets,f)
    with open('random_outlets_{}.pkl'.format(pts),'wb') as f:
        pickle.dump(outlets,f)
    return (time.process_time()-t0)/pts*100
    
def gen_strat_data(pts):
    t0=time.process_time()
    inlets = []
    outlets = []
    if pts == 100:
        segments=2
    else:
        segments=round(pts**(1/5))
    span = np.linspace(0,1,num=segments+1)
    for i,k,l,m,n in product(range(segments),repeat=5):
        T=np.random.uniform(span[i],span[i+1])
        P=0.5
        C2H4=np.random.uniform(span[k],span[k+1])
        O2=np.random.uniform(span[l],span[l+1])
        Q=np.random.uniform(span[m],span[m+1])
        cat=np.random.uniform(span[n],span[n+1])
        s=[T,P,C2H4,O2,0,0,0]
        result = UnitOp(Stream(*s),[Q,cat]).results()
        inlets.append([T,C2H4,O2,Q,cat])
        outlets.append(result)
    with open('strat_inlets_{}.pkl'.format(pts),'wb') as f:
        pickle.dump(inlets,f)
    with open('strat_outlets_{}.pkl'.format(pts),'wb') as f:
        pickle.dump(outlets,f)
    return (time.process_time()-t0)/pts*100
    
def gen_lhs_data(pts):
    t0=time.process_time()
    inlets = []
    outlets = []
    space = Space([(0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.)]) 
    lhss = Lhs(criterion="maximin", iterations=1000)
    lhs_points = lhss.generate(space.dimensions, pts)
    # lhs_points = lhs(5, samples=pts,criterion='maximin')
    for i in range(pts):
        T,C2H4,O2,Q,cat=lhs_points[i]
        P=0.5
        s=[T,P,C2H4,O2,0,0,0]
        result = UnitOp(Stream(*s),[Q,cat]).results()
        inlets.append([T,C2H4,O2,Q,cat])
        outlets.append(result)
    with open('lhs_inlets_{}.pkl'.format(pts),'wb') as f:
        pickle.dump(inlets,f)
    with open('lhs_outlets_{}.pkl'.format(pts),'wb') as f:
        pickle.dump(outlets,f)
    return (time.process_time()-t0)/pts*100

def gen_test_data(pts):
    inlets = []
    outlets = []
    for i in range(pts):
        T=np.random.uniform(0,1); P=0.5; C2H4=np.random.uniform(0,1); O2=np.random.uniform(0,1); Q=np.random.uniform(0,1); cat=np.random.uniform(0,1)
        s=[T,P,C2H4,O2,0,0,0]
        result = UnitOp(Stream(*s),[Q,cat]).results()
        inlets.append([T,C2H4,O2,Q,cat])
        outlets.append(result)
    with open('test_inlets.pkl','wb') as f:
        pickle.dump(inlets,f)
    with open('test_outlets.pkl','wb') as f:
        pickle.dump(outlets,f)

def train_test_RF(sampling_technique,pts):
    t0=time.process_time()
    with open('{}_inlets_{}.pkl'.format(sampling_technique,pts),'rb') as f:
        inlets = pickle.load(f)
    with open('{}_outlets_{}.pkl'.format(sampling_technique,pts),'rb') as f:
        outlets = pickle.load(f)
    with open('test_inlets.pkl','rb') as f:
        test_inlets = pickle.load(f)
    with open('test_outlets.pkl','rb') as f:
        test_outlets = pickle.load(f)
    inlet_df = pd.DataFrame(inlets)
    outlet_df = pd.DataFrame(outlets)
    test_inlet_df = pd.DataFrame(test_inlets)
    test_outlet_df = pd.DataFrame(test_outlets)
    RF = RandomForestRegressor(n_estimators=50)
    RF.fit(inlet_df,outlet_df)
    score=RF.score(test_inlet_df,test_outlet_df)
    time_taken=(time.process_time()-t0)/pts*100
    # with open('RF_{}_{}.pkl'.format(sampling_technique,pts),'wb') as f:
    #     pickle.dump(RF,f)
    return [score,time_taken]

def train_test_GP(sampling_technique,pts):
    t0=time.process_time()
    with open('{}_inlets_{}.pkl'.format(sampling_technique,pts),'rb') as f:
        inlets = pickle.load(f)
    with open('{}_outlets_{}.pkl'.format(sampling_technique,pts),'rb') as f:
        outlets = pickle.load(f)
    with open('test_inlets.pkl','rb') as f:
        test_inlets = pickle.load(f)
    with open('test_outlets.pkl','rb') as f:
        test_outlets = pickle.load(f)
    inlet_df = pd.DataFrame(inlets)
    outlet_df = pd.DataFrame(outlets)
    test_inlet_df = pd.DataFrame(test_inlets)
    test_outlet_df = pd.DataFrame(test_outlets)
    kernel = C(1e-5)*RBF()+RBF(0.1)+WhiteKernel()
    # kernel = RBF()
    GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
    GP.fit(inlet_df,outlet_df)
    score=GP.score(test_inlet_df,test_outlet_df)
    time_taken=(time.process_time()-t0)/pts*100
    # with open('GP_{}_{}.pkl'.format(sampling_technique,pts),'wb') as f:
    #     pickle.dump(GP,f)
    return [score,time_taken]

def train_test_NN(sampling_technique,pts):
    t0=time.process_time()
    with open('{}_inlets_{}.pkl'.format(sampling_technique,pts),'rb') as f:
        inlets = pickle.load(f)
    with open('{}_outlets_{}.pkl'.format(sampling_technique,pts),'rb') as f:
        outlets = pickle.load(f)
    with open('test_inlets.pkl','rb') as f:
        test_inlets = pickle.load(f)
    with open('test_outlets.pkl','rb') as f:
        test_outlets = pickle.load(f)
    sc = StandardScaler()
    inlet_df = sc.fit_transform(inlets) #5features
    test_inlet_df = sc.transform(test_inlets)
    outlet_df = sc.fit_transform(outlets) #6outputs
    test_outlet_df = sc.transform(test_outlets)
    model = Sequential()
    model.add(Dense(units = 10, activation = 'relu', input_dim = 5))
    model.add(Dense(units = 10, activation = 'relu'))
    model.add(Dense(units = 10, activation = 'relu'))
    model.add(Dense(units = 10, activation = 'relu'))
    model.add(Dense(units = 10, activation = 'relu'))
    model.add(Dense(units = 6))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['mse'])
    NN = model.fit(inlet_df, outlet_df, batch_size = 50, epochs = 20, validation_split=0.1)
    pred_outlet_df = model.predict(test_inlet_df)
    score = r2_score(test_outlet_df,pred_outlet_df)
    time_taken=(time.process_time()-t0)/pts*100
    return [score,time_taken]
    
def gen_inlets(pts_set):
    times=[]
    for pts in pts_set:
        times.append([gen_random_data(pts),gen_strat_data(pts),gen_lhs_data(pts)])
    with open('gen_times_{}.pkl'.format(pts_set[-1]),'wb') as f:
        pickle.dump(times,f)

def train_algs(pts_set):
    scores=np.empty([3,3,len(pts_set)])
    times=np.empty([3,3,len(pts_set)])
    for i, (method, colour) in enumerate(zip(['random', 'strat', 'lhs'], ['red', 'deepskyblue','black'])):
        for k,pts in enumerate(pts_set):
            RFscores, RFtimes=train_test_RF(method,pts)
            GPscores, GPtimes=train_test_GP(method,pts)
            NNscores, NNtimes=train_test_NN(method, pts)
            scores[0][i][k], scores[1][i][k], scores[2][i][k] = RFscores, GPscores, NNscores
            times[0][i][k], times[1][i][k], times[2][i][k] = RFtimes, GPtimes, NNtimes
            # with open('pts_marker_{}_{}.pkl'.format(pts,method),'wb') as f:
            #     pickle.dump(None,f)
    with open('scores_{}.pkl'.format(pts_set[-1]),'wb') as f:
        pickle.dump(scores,f)
    with open('times_{}.pkl'.format(pts_set[-1]),'wb') as f:
        pickle.dump(times,f)

def plot(pts_set):
    with open('scores_{}.pkl'.format(pts_set[-1]),'rb') as f:
        scores = pickle.load(f)
    with open('times_{}.pkl'.format(pts_set[-1]),'rb') as f:
        times = pickle.load(f)
    with open('gen_times_{}.pkl'.format(pts_set[-1]),'rb') as f:
        gen_times = pickle.load(f)
    gen_times=np.array(gen_times).T.tolist()
    fig_, ax_ = plt.subplots()
    fig_2, ax_2 = plt.subplots()
    for i, (method, colour) in enumerate(zip(['random', 'strat', 'lhs'], ['red', 'deepskyblue','black'])):
        for j, (ML_model, line) in enumerate(zip(['RF', 'GP', 'NN'], ['solid', 'dotted', 'dashed'])):
            ax_.plot(np.log10(pts_set), scores[j][i], linestyle=line, color=colour, label=ML_model+' '+method)
            ax_2.plot(np.log10(pts_set), np.log10(times[j][i]+gen_times[i]), linestyle=line, color=colour, label=ML_model+' '+method) # for j, ML_model in enumerate(['RF', 'GP', 'NN']):
    ax_.legend(fontsize=8)
    ax_2.legend(fontsize=8)
    ax_.set_xlabel('$log_{10}$(number of training points)')
    ax_.set_ylabel('$R^2$ test score')
    ax_2.set_xlabel('$log_{10}$(number of training points)')
    ax_2.set_ylabel('$log_{10}$(CPU time per 100 points / s)')
    ax_.set_xticks([2, 2.5, 3, 3.5, 4])
    ax_2.set_xticks([2, 2.5, 3, 3.5, 4])
    fig_.savefig('scores_{}.png'.format(pts_set[-1]), dpi=200,bbox_inches='tight')
    fig_2.savefig('times_{}.png'.format(pts_set[-1]), dpi=200,bbox_inches='tight')

# gen_random_data(10)
# with open('gen_times_{}.pkl'.format(10000),'rb') as f:
#     times = pickle.load(f)
# print(times)
# gen_test_data(1000)
# pts_set=[100,300,1000,3000,10000]
# pts_set=[100,300,1000]
# gen_inlets(pts_set)
# train_algs(pts_set)
# plot(pts_set)
#font+fontsize
#sig figs on axes maybe


# def plot():
#     plt.figure()
#     plt.plot(no_samples_set, avg_scores)
#     plt.xlabel('Number of training points')
#     plt.ylabel('Average test score over {} runs'.format(no_runs))
#     plt.plot([], [], ' ', label='LHS RF')
#     plt.legend()
#     plt.savefig('LHS RF, accuracy vs samples.png', bbox_inches='tight')
    
#     plt.figure()
#     plt.scatter(no_samples_set, avg_times_per_100samples)
#     plt.xlabel('Number of training points')
#     plt.ylabel('Average time per 100 samples over {} runs'.format(no_runs))
#     plt.plot([], [], ' ', label='LHS RF')
#     plt.legend()
#     plt.savefig('LHS RF, speed vs samples.png', bbox_inches='tight')


# C2H4_cost=1*Mr[0]/1000 # £1/kg * kg/kmol / 1000 mol/kmol
# O2_cost=2.3*Mr[1]/1000 # £2.3/kg * kg/kmol / 1000 mol/kmol
# Q_cost=0.2/3.6 # £0.2/kWh / 3.6 MJ/kWh = £/MJ
# T in K, component flows in mol/s, Q in MJ, cat in kg
# 405-530 K
# 10-22 bar
# yLowerCO2 >= 0.05

# inlet = Stream(0.5,0.5,0.5,0.5,0,0,0)
# result = UnitOp(inlet,[0.5,0.5]).results()
# print(result)

# def run_code():
#     for ML_model in ['RF','GP','NN']:
#         for sampling_technique in ['random','strat','lhs']:
#             ln=compile("{}_{}_scores,{}_{}_times=[],[]".format(ML_model,sampling_technique,ML_model,sampling_technique),'<string>','exec');exec(ln)
#             for pts in pts_set:
#                 ln=compile("{}_{}_score,{}_{}_time=train_test_{}({},pts)".format(ML_model,sampling_technique,ML_model,sampling_technique,ML_model,'sampling_technique'),'<string>','exec');exec(ln)
#                 ln=compile("{}_{}_scores.append({}_{}_score)".format(ML_model,sampling_technique,ML_model,sampling_technique),'<string>','exec');exec(ln)
#                 ln=compile("{}_{}_times.append({}_{}_time)".format(ML_model,sampling_technique,ML_model,sampling_technique),'<string>','exec');exec(ln)
#             ln=compile("plt.plot(pts_set,{}_{}_scores)";exec(ln)
#             ln=compile("plt.plot(pts_set,{}_{}_times)";exec(ln)
                     
# def run_code():
#     scores=[]
#     for pts in pts_set:
#         RFrandomscore,RFrandomtime=train_test_RF('random',pts)
#         RFstratscore,RFstrattime=train_test_RF('strat',pts)
#         RFlhsscore,RFlhstime=train_test_RF('lhs',pts)
#         GPrandomscore,GPrandomtime=train_test_GP('random',pts)
#         GPstratscore,GPstrattime=train_test_GP('strat',pts)
#         GPlhsscore,GPlhstime=train_test_GP('lhs',pts)
#         NNrandomscore,NNrandomtime=train_test_NN('random',pts)
#         NNstratscore,NNstrattime=train_test_NN('strat',pts)
#         NNlhsscore,NNlhstime=train_test_NN('lhs',pts)
#         scores.append([[RFrandomscore,RFstratscore,RFlhsscore],[GPrandomscore,GPstratscore,GPlhsscore],[NNrandomscore,NNstratscore,NNlhsscore]])
#     plt.plot(pts_set, [scores[0][0][0],scores[1][0][0]])#,scores[2][0][0],scores[3][0][0]]
# pts_set=[10,20]
# gen_inlets()
# run_code()

# def run_code2():
#     scores=np.empty([3,3,2])
#     times=np.empty([3,3,2])
#     pts_set=[10,20]
#     for i, method in enumerate(['random', 'strat', 'lhs']):
#         for k,pts in enumerate(pts_set):
#             RFscores, RFtimes=train_test_RF(method,pts)
#             GPscores, GPtimes=train_test_GP(method,pts)
#             NNscores, NNtimes=train_test_NN(method, pts)
#             scores[0][i][k], scores[1][i][k], scores[2][i][k] = RFscores, GPscores, NNscores
#             times[0][i][k], times[1][i][k], times[2][i][k] = RFtimes, GPtimes, NNtimes
#         for j, ML_model in enumerate(['RF', 'GP', 'NN']):
#             plt.figure(0)
#             plt.plot(pts_set, scores[i][j], label=(ML_model, method))
#             plt.figure(1)
#             plt.plot(pts_set, times[i][j], label=(ML_model, method))
#     plt.legend()
#     plt.figure(1)
#     plt.legend()

# pts=500
# # gen_test_data(pts)
# sampling_technique='random'
# # gen_random_data(pts)
# print(train_test_NN(sampling_technique,pts))
    
# def gen_inlets():
#     for sampling_technique in ['random','strat','lhs']:
#         for pts in pts_set:
#             ln=compile("gen_{}_data({})".format(sampling_technique,pts),'<string>','exec');exec(ln)

# def gen_timesTEMP():
#     with open('gen_times_{}.pkl'.format(10000),'rb') as f:
#         times = pickle.load(f)
#     times3000 = times[0:4]
#     with open('gen_times_{}.pkl'.format(3000),'wb') as f:
#         pickle.dump(times3000,f)

# gen_timesTEMP()

# print(gen_lhs_data(1069))