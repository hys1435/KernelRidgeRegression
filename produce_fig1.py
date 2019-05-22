#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:23:40 2019

@author: Zhaoqi Li
"""
import numpy as np
from KRR_algorithm import compute_mse
from sim_study_helper_funs import init_sim_data, init_params
# from sklearn.kernel_ridge import KernelRidge
import time
import matplotlib.pyplot as plt
import matplotlib.ticker
from random import shuffle
#from numpy.random import shuffle
    
# code to reproduce Figure 1 of zhang15d paper

def main():
    # Initialize global variables
    #np.random.seed(521)
    start_time = time.time()
    NLst = np.logspace(8, 12, num = 5, base = 2).astype(int) # correct one is 8-13
    mLst = np.logspace(0, 3, num = 4, base = 4).astype(int)
    dist_metric = "sobolev"
    #dist_metric = "gaussian"
    sim_num = 5
    mse_lst = np.zeros((mLst.size, NLst.size, sim_num)) # list of mse with under-regularization
    mse_lst_nr = np.zeros((mLst.size, NLst.size, sim_num)) # list of mse without under-regularization
    for k in range(sim_num):
        for i, N in enumerate(NLst):
            X, y = init_sim_data(N)
            ind = [x for x in range(N)]
            shuffle(ind)
            np.take(X, ind, axis = 0, out = X)
            np.take(y, ind, axis = 0, out = y)
            for j, m in enumerate(mLst):
                lam, n, params = init_params(N, m)
                lam_nr = n**(-2/3)
                #lam_nr = 1e-4
                params_nr = [1, lam_nr]
                #kr = KernelRidge(kernel='rbf', alpha=10000000, gamma=1)
                #XX = X.reshape(-1,1)
                #kr.fit(XX, y)
                #y_pred = kr.predict(XX)
                #mse_lst[j,i,k] = np.mean((y - y_pred)**2)
                mse_lst[j,i,k] = compute_mse(X, y, N, m, params, dist_metric, integral = True)
                mse_lst_nr[j,i,k] = compute_mse(X, y, N, m, params_nr, dist_metric)
            #p.close()
            #p.join()
        print("run time is: ", (time.time() - start_time))

    mse_lst = np.mean(mse_lst, axis = 2)
    mse_lst_nr = np.mean(mse_lst_nr, axis = 2)
    print(mse_lst)
    print(mse_lst_nr)
    
    # Plot results
    fig, ax = plt.subplots()
    cols = ['red', 'blue', 'yellow', 'orange']
    markers = ['o', '^', 's', 'd']
    ax.set_xscale('log')
    plt.yscale('log')
    plt.xticks(NLst)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    for i in range(mLst.size):
        ax.plot(NLst, mse_lst[i], c=cols[i], marker=markers[i],label='m={}'.format(4**i))
        #ax.set_yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel("Total number of samples (N)")
    plt.ylabel("Mean square error")
    plt.show()
    
    fig2, ax2 = plt.subplots()
    ax2.set_xscale('log')
    plt.yscale('log')
    plt.xticks(NLst)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    for i in range(mLst.size):
        ax2.plot(NLst, mse_lst_nr[i], c=cols[i], marker=markers[i],label='m={}'.format(4**i))
    plt.legend(loc='upper right')
    plt.xlabel("Total number of samples (N)")
    plt.ylabel("Mean square error")
    #plt.title("Kernel Ridge Regression without under-regularization")

    plt.show()

if __name__ == '__main__':
     main()