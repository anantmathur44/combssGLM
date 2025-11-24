#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 22:11:17 2025

@author: s.moka
"""
import numpy as np
import _opt_log as olog

#%%
# Load the CSV file (assuming no headers)
train_path = "./data/n-200-p1000Replica1.csv"
#test_path = "./data/n-200-p1000Test-Replica1.csv"
data = np.loadtxt(train_path, delimiter='\t', skiprows=0)  # skiprows=1 if header exists

# Split into X and y
y = data[:, 0]  # First column is the response
X = data[:, 1:]  # All remaining columns are features



#%%
q = 10

result = olog.fw(X, y, q, solver = 'lbfgs', r = 1.2, m=10, delta_min = 0.02)

#%%

print(result.niters)

#%%
"""
Testing gradient
"""
(n, p) = X.shape

t = np.ones(p-1)*0.6
beta_start = np.ones(p)
delta = n


grad = olog.f_grad_cg(t, beta_start, X, y, delta, solver = "lbfgs") 

print(grad[:10])

h = 0.00001
j = 200

th = t.copy()
th[j] = th[j] + h

a = olog.f(t, X, y, delta, solver="lbfgs")
b = olog.f(th, X, y, delta, solver="lbfgs")

print("j", j, "num  partial der:", (b - a)/h)
print("j", j, "grad partial der:", grad[j])
