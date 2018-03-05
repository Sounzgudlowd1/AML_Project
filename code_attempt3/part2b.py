# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:02:04 2018

@author: Erik
"""

import gradient_calculation as gc
import get_data as gd
from scipy.optimize import fmin_bfgs
from scipy.optimize import check_grad
from time import time
import numpy as np
import winsound

def func_to_minimize(params, X, y, C):
    reg = 1/2 * np.sum(params ** 2)
    avg_prob = gc.avg_log_p_y_given_x(params, X, y, len(X))
    return -C * avg_prob + reg

def grad_func(parms, X, y, C):
    grad_avg_prob =  gc.avg_gradient(params, X, y, len(X))
    grad_reg = params
    return -C * grad_avg_prob + grad_reg
    



params = gd.get_params()
X, y = gd.read_data_formatted()




start = time()
out = fmin_bfgs(func_to_minimize, params, grad_func, (X, y, 1000), maxiter = 1)
print("Total time: ", end = '')
print(time() - start)


start = time()
out = fmin_bfgs(func_to_minimize, params, grad_func, (X, y, 1000), maxiter = 2)
print("Total time: ", end = '')
print(time() - start)


frequency = 2500  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second
#winsound.Beep(frequency, duration)