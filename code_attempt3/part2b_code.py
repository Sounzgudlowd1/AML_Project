# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:02:04 2018

@author: Erik
"""

import gradient_calculation as gc
import get_data as gd
from scipy.optimize import fmin_bfgs
from time import time
import numpy as np
import winsound
import part1c_code as p1c

def func_to_minimize(params, X, y, C):
    num_examples = len(X)
    reg = 1/2 * np.sum(params ** 2)
    avg_prob = gc.log_p_y_given_x_avg(params, X, y, num_examples)
    return -C * avg_prob + reg

def grad_func(params, X, y, C):
    num_examples = len(X)
    grad_avg =  gc.gradient_avg(params, X, y, num_examples)
    grad_reg = params
    return -C * grad_avg + grad_reg
    

def optimize(params, X, y, C, name):

    start = time()
    out = fmin_bfgs(func_to_minimize, params, grad_func, (X, y, C))
    print("Total time: ", end = '')
    print(time() - start)
    
    
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 500  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    
    with open("../result/" + name + ".txt", "w") as text_file:
        for i, elt in enumerate(out):
            text_file.write(str(elt))
            text_file.write("\n")

def get_optimal_params():
    
    file = open('../result/solution.txt', 'r') 
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)

def predict(X, w, t):
    y_pred = []
    for i, x in enumerate(X):
        M = p1c.optimize(x, w, t)
        y_pred.append(p1c.get_solution_from_M(M, x, w, t))
    return y_pred