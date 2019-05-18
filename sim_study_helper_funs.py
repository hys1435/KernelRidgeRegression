#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:49:17 2019

@author: zli9
"""

# helper functions for simulation studies

import numpy as np

def f_star(x):
    return np.minimum(x, 1-x)

def init_sim_data(N):
    # Initialize data given N number of samples
    X = np.random.uniform(size = N)
    epsilon = np.random.normal(scale = 1/np.sqrt(5), size = N)
    y = f_star(X) + epsilon
    return X, y

def init_params(N, m):
    lam = N**(-2/3)
    n = int(N / m)
    params = [-1, lam] # params are nothing and lambda
    return lam, n, params
"""
def init_params(N, m):
    lam = N**(-2/3)
    n = int(N / m)
    p = Pool(m)
    params = [-1, lam] # params are nothing and lambda
    return lam, n, p, params
    """