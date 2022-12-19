'''
These functions generate the training data of inlet-outlet pairs of the reactor,
for different sampling techniques of the input space: random, stratified, latin hypercube.

gen_training_data calls all 3 sampling techniques at once and pickles the data and the times to generate them.
'''


import numpy as np
import time
import pickle
from itertools import product
from skopt.space import Space
from skopt.sampler import Lhs

from physical_model import *

#random sampling
def gen_random_data(pts):
    t0=time.process_time()
    inlets = []
    outlets = []
    #generate a set of randomly sampled inlet streams within the potential operating conditions - 5 input variables T,C2H4,O2,Q,cat
    for i in range(pts):
        #generate inlet. pressure is not varying. there are no products in the inlet.
        T=np.random.uniform(0,1); P=0.5; C2H4=np.random.uniform(0,1); O2=np.random.uniform(0,1); Q=np.random.uniform(0,1); cat=np.random.uniform(0,1)
        s=[T,P,C2H4,O2,0,0,0]
        #calculate outlet
        result = UnitOp(Stream(*s),[Q,cat]).results()
        #record the inlet and outlet
        inlets.append([T,C2H4,O2,Q,cat])
        outlets.append(result)
    with open('pickled_data/random_inlets_{}.pkl'.format(pts),'wb') as f:
        pickle.dump(inlets,f)
    with open('pickled_data/random_outlets_{}.pkl'.format(pts),'wb') as f:
        pickle.dump(outlets,f)
    return (time.process_time()-t0)/pts*100

#stratified sampling
def gen_strat_data(pts):
    t0=time.process_time()
    inlets = []
    outlets = []
    #number of segments to split each input variable of the 5-dimensional space into, for stratified sampling.
    #divide the space into a rubik's cube so that there is one data point in each segment
    #actual number of data points is powers of 5 [32,243,1024,3125,7776]
    if pts == 100:
        segments=2
    else:
        segments=round(pts**(1/5))
    span = np.linspace(0,1,num=segments+1)
    #generate a stratified set of inlet streams within the 5 dimensional input space.
    for i,k,l,m,n in product(range(segments),repeat=5):
        #generate inlet
        T=np.random.uniform(span[i],span[i+1])
        P=0.5
        C2H4=np.random.uniform(span[k],span[k+1])
        O2=np.random.uniform(span[l],span[l+1])
        Q=np.random.uniform(span[m],span[m+1])
        cat=np.random.uniform(span[n],span[n+1])
        s=[T,P,C2H4,O2,0,0,0]
        #calculate outlet
        result = UnitOp(Stream(*s),[Q,cat]).results()
        inlets.append([T,C2H4,O2,Q,cat])
        outlets.append(result)
    with open('pickled_data/strat_inlets_{}.pkl'.format(pts),'wb') as f:
        pickle.dump(inlets,f)
    with open('pickled_data/strat_outlets_{}.pkl'.format(pts),'wb') as f:
        pickle.dump(outlets,f)
    return (time.process_time()-t0)/pts*100

#latin hypercube sampling
def gen_lhs_data(pts):
    t0=time.process_time()
    inlets = []
    outlets = []
    space = Space([(0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.)]) 
    lhss = Lhs(criterion="maximin", iterations=1000)
    lhs_points = lhss.generate(space.dimensions, pts)
    #generate a latin hypercube sampled set of inlet streams within the 5 dimensional input space.
    for i in range(pts):
        #generate inlet
        T,C2H4,O2,Q,cat=lhs_points[i]
        P=0.5
        s=[T,P,C2H4,O2,0,0,0]
        #calculate outlet
        result = UnitOp(Stream(*s),[Q,cat]).results()
        inlets.append([T,C2H4,O2,Q,cat])
        outlets.append(result)
    with open('pickled_data/lhs_inlets_{}.pkl'.format(pts),'wb') as f:
        pickle.dump(inlets,f)
    with open('pickled_data/lhs_outlets_{}.pkl'.format(pts),'wb') as f:
        pickle.dump(outlets,f)
    return (time.process_time()-t0)/pts*100

#generate the inlet-outlet training data pairs for different sampling techniques and different numbers of pts
def gen_training_data(pts_set):
    times=[]
    for pts in pts_set:
        times.append([gen_random_data(pts),gen_strat_data(pts),gen_lhs_data(pts)])
    with open('pickled_data/gen_times_{}.pkl'.format(pts_set[-1]),'wb') as f:
        pickle.dump(times,f)

#generate the inlet-outlet test data pairs for a given number of pts. the test data is randomly sampled and the same test data is used across all tests to make it fair.
def gen_test_data(pts):
    inlets = []
    outlets = []
    for i in range(pts):
        #generate inlet
        T=np.random.uniform(0,1); P=0.5; C2H4=np.random.uniform(0,1); O2=np.random.uniform(0,1); Q=np.random.uniform(0,1); cat=np.random.uniform(0,1)
        s=[T,P,C2H4,O2,0,0,0]
        #calculate outlet
        result = UnitOp(Stream(*s),[Q,cat]).results()
        inlets.append([T,C2H4,O2,Q,cat])
        outlets.append(result)
    with open('pickled_data/test_inlets.pkl','wb') as f:
        pickle.dump(inlets,f)
    with open('pickled_data/test_outlets.pkl','wb') as f:
        pickle.dump(outlets,f)
