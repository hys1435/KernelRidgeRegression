#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:48:30 2019

@author: Zhaoqi Li
"""

# Kernel Ridge Regression Algorithm reproducing results from zhang15d paper

import numpy as np

def gaussianRBF(u, v, params):
    sigma = params[0]
    if (len((u-v).shape) == 1): 
        return np.exp(-np.einsum('i,i',u-v,u-v)/(sigma**2))
    return np.exp(-np.einsum('ij,ij->i',u-v,u-v)/(sigma**2)) # einsum computes norm squared

def sobolev(u, v, params):
    return 1 + np.minimum(u, v)
    
def get_kernel(dist_metric):
    if (dist_metric == "gaussian"):
        return gaussianRBF
    elif (dist_metric == "sobolev"):
        return sobolev
    
def f_star(x):
    return np.minimum(x, 1-x)

def compute_gram_mat(X1, X2, params, dist_metric):
    kernel = get_kernel(dist_metric)
    gram_mat = np.zeros((X1.shape[0],X2.shape[0]))
    for i, item in enumerate(X2):
        gram_mat[:,i] = kernel(X1, item, params)
    return gram_mat

def compute_kernel_ridge_coeffs(X, y, params, dist_metric):
    # Key algorithm to compute the kernel ridge coefficients alpha
    K = compute_gram_mat(X, X, params, dist_metric)
    alpha = compute_coeffs_from_K(K, y, params)
    return alpha

def compute_coeffs_from_K(K, y, params):
    # compute the kernel coefficients alpha given the gram matrix K, formula listed as 
    # equantion (37) of the original paper
    lam = params[-1]
    alpha = np.linalg.solve(K + lam * y.size * np.eye(K.shape[0]), y)
    return alpha

def split_into_m_parts(X, m):
    # split the entire dataset into subsets
    n = int(X.shape[0] / m)
    resShape = [m, n]
    if len(X.shape) > 1:
        for item in X.shape[1:]:
            resShape.append(item)
    res = np.zeros(resShape)
    for i in range(m):
        res[i] = X[i*n:(i+1)*n]
    return res
    
def predict(X_train, X_test, alpha, m, params, dist_metric, output = False):
    # compute the prediction of y using kernel coefficients alpha
    K = compute_gram_mat(X_test, X_train, params, dist_metric)
    print("K: ", K.shape)
    print("alpha: ", alpha.shape)
    y_pred = np.dot(K, alpha)
    if (output):
        return y_pred, K, alpha
    return y_pred

def compute_mse(X, y, N, m, params, dist_metric, 
                X_test = None, y_test = None, real = False, integral = False):
    # Key function to compute the mse, real is the parameter indicating if it's 
    # simulation study or real data
    # n = int(N / m)
    if (real): 
        y_pred_lst = np.zeros((m, X_test.shape[0]))
    elif (integral):
        y_pred_lst = np.zeros((m, 200))
    else:
        y_pred_lst = np.zeros((m, N))
    X_split = split_into_m_parts(X, m)
    y_split = split_into_m_parts(y, m)
    for k, (XX, yy) in enumerate(zip(X_split, y_split)):
        alpha = compute_kernel_ridge_coeffs(XX, yy, params, dist_metric)
        if (real):
            print("XX:", XX.shape)
            print("X: ", X_test.shape)
            y_pred_lst[k] = predict(XX, X_test, alpha, m, params, dist_metric)
        elif (integral): # integral not working right now -> due to the size of y_pred_lst not match with X_seq
            X_seq = np.linspace(start = 1e-4, stop = 1, num = 200)
            y_pred_lst[k] = predict(XX, X_seq, alpha, m, params, dist_metric)
        else:
            y_pred_lst[k] = predict(XX, X, alpha, m, params, dist_metric)
    if (integral):
        y_test = f_star(X_seq)
    elif (not real):
        y_test = f_star(X)
    y_pred = np.mean(y_pred_lst, axis = 0)
    mse = np.mean((y_test - y_pred)**2)
    return mse

def compute_mse_no_avg(X, y, N, m, params, dist_metric, 
                X_test = None, y_test = None, real = False):
    # Key function to compute the mse, real is the parameter indicating if it's 
    # simulation study or real data
    n = int(N / m)
    X1 = X[0:n]
    y1 = y[0:n]
    alpha = compute_kernel_ridge_coeffs(X1, y1, params, dist_metric)
    y_pred = predict(X1, X_test, alpha, m, params, dist_metric) # multiply by m to remove averaging
    mse = np.mean((y_test - y_pred)**2)
    return mse
    
"""
def compute_mse(X, y, N, m, p, params, dist_metric, 
                X_test = None, y_test = None, real = False):
    alpha = np.zeros(N)
    n = int(N / m)
    X_split = split_into_m_parts(X, m)
    y_split = split_into_m_parts(y, m)
    results = [p.apply_async(compute_kernel_ridge_coeffs, [XX, yy, params, dist_metric]) 
                for XX, yy in zip(X_split, y_split)]
    for k, r in enumerate(results):
        alpha[k*n:(k+1)*n] = r.get()
    if (real):
        y_pred = predict(X, X_test, alpha, m, params, dist_metric)
        mse = np.mean((y_test - y_pred)**2)
    else:
        y_pred = predict(X, X, alpha, m, params, dist_metric)
        mse = np.mean((y - y_pred)**2)
    return mse
    """