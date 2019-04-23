#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:48:30 2019

@author: Zhaoqi Li
"""

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
    
def compute_gram_mat(X1, X2, params, dist_metric):
    kernel = get_kernel(dist_metric)
    gram_mat = np.zeros((X1.shape[0],X2.shape[0]))
    for i, item in enumerate(X2):
        gram_mat[:,i] = kernel(X1, item, params)
    return gram_mat

def compute_kernel_ridge_coeffs(X, y, params, dist_metric):
    lam = params[-1]
    K = compute_gram_mat(X, X, params, dist_metric)
    K_sudinv = np.linalg.inv(K + lam * y.size * np.eye(K.shape[0]))
    alpha = np.dot(K_sudinv, y)
    return alpha

def split_into_m_parts(X, m):
    n = int(X.size / m)
    res = np.zeros((m, n))
    for i in range(m):
        res[i] = X[i*n:(i+1)*n]
    return res

def predict(X_train, X_test, alpha, m, params, dist_metric, output = False):
    K = compute_gram_mat(X_test, X_train, params, dist_metric)
    y_pred = 1/m * np.dot(K, alpha)
    if (output):
        return y_pred, K, alpha
    return y_pred