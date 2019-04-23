#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:49:15 2019

@author: Zhaoqi Li
"""

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
# fixed random seed for reproducibility
np.random.seed(0)

features = ['year', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't1 9', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36 ', 't37', 't38', 't39', 't40', 't41', 't42', 't43', 't44', 't45', 't46', 't47', 't48', 't49', 't50', 't51', 't52', 't53' , 't54', 't55', 't56', 't57', 't58', 't59', 't60', 't61', 't62', 't63', 't64', 't65', 't66', 't67', 't68', 't69', 't70', 't71', 't72', 't73', 't74', 't75', 't76', 't77', 't78', 't79', 't80', 't81', 't82', 't83', 't84', 't85', 't86', 't87', 't88', 't89', 't90']

# Note that our classes (which we have to predict from those 90 features), are all
# the years from 1922 to 2011: 1922, 1923, 1924, 1925, ..., 2011
# Theare exactly 90 years, so we also have 90 classes:
nb_classes = 90

data = pd.read_csv('YearPredictionMSD.csv', names=features)

X = data.ix[:,1:].as_matrix()  # this is the 90 columns without the year
Y = data.ix[:,0].as_matrix()   # this is the year column

# data normalizations (scaling down all values to the interval [0,1])
# The years 1922-2011 are scaled down to integers [0,1,2,..., 89] 
a = X.min()
b = X.max()
X = (X - a) / (b - a)  # all values now between 0 and 1 !
Y = Y - Y.min()        # The years 1922-2011 are mapped to 0-89

# Training data set
X_train = X[0:463715]
y_train = Y[0:463715]

# Validation data set
X_test = X[463715:]
y_test = Y[463715:]