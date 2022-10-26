'''
this function plots the scores and times calculated from train_algs for different combinations of ML model/sampling technique/number of pts


'''

import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_scores_times(pts_set):
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)

    #get the data
    with open('pickled_data/scores_{}.pkl'.format(pts_set[-1]),'rb') as f:
        scores = pickle.load(f)
    with open('pickled_data/times_{}.pkl'.format(pts_set[-1]),'rb') as f:
        times = pickle.load(f)
    with open('pickled_data/gen_times_{}.pkl'.format(pts_set[-1]),'rb') as f:
        gen_times = pickle.load(f)
    gen_times=np.array(gen_times).T.tolist()

    #make the plots
    fig_, ax_ = plt.subplots()
    fig_2, ax_2 = plt.subplots()
    #for each combination of ML model/sampling technique
    for i, (method, colour) in enumerate(zip(['random', 'strat', 'lhs'], ['red', 'deepskyblue','black'])):
        for j, (ML_model, line) in enumerate(zip(['RF', 'GP', 'NN'], ['solid', 'dotted', 'dashed'])):
            #plot the scores vs number of training pts
            ax_.plot(np.log10(pts_set), scores[j][i], linestyle=line, color=colour, label=ML_model+' '+method)
            #plot the times vs number of training pts. note that the data generation time (gen_times) is added to the training time (times)
            ax_2.plot(np.log10(pts_set), np.log10(times[j][i]+gen_times[i]), linestyle=line, color=colour, label=ML_model+' '+method)
    
    #make the plots pretty
    ax_.legend(fontsize=8)
    ax_2.legend(fontsize=8)
    ax_.set_xlabel('$log_{10}$(number of training points)')
    ax_.set_ylabel('$R^2$ test score')
    ax_2.set_xlabel('$log_{10}$(number of training points)')
    ax_2.set_ylabel('$log_{10}$(CPU time per 100 points / s)')
    ax_.set_xticks([2, 2.5, 3, 3.5, 4])
    ax_2.set_xticks([2, 2.5, 3, 3.5, 4])
    fig_.savefig('results/scores_{}.png'.format(pts_set[-1]), dpi=200,bbox_inches='tight')
    fig_2.savefig('results/times_{}.png'.format(pts_set[-1]), dpi=200,bbox_inches='tight')