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

def func_to_minimize(params, X, y, C):
    num_examples = len(X)
    reg = 1/2 * np.sum(params ** 2)
    sum_prob = gc.log_p_y_given_x_sum(params, X, y, num_examples)
    return -C/len(X) * sum_prob + reg

def grad_func(params, X, y, C):
    num_examples = len(X)
    grad_sum =  gc.gradient_sum(params, X, y, num_examples)
    grad_reg = params
    return -C/len(X) * grad_sum + grad_reg
    

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
            print(i)
            text_file.write(str(elt))
            text_file.write("\n")

def get_optimal_params():
    
    file = open('../result/solution.txt', 'r') 
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)

