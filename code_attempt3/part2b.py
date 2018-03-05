# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:02:04 2018

@author: Erik
"""

import gradient_calculation as gc
import get_data as gd
from scipy.optimize import fmin_bfgs
from time import time
import winsound


params = gd.get_params()
X, y = gd.read_data_formatted()

start = time()
out = fmin_bfgs(gc.neg_avg_log_p_y_given_x, params, gc.avg_gradient, (X, y, len(X), 1000), maxiter = 1)
print("Total time: ", end = '')
print(time() - start)


start = time()
out = fmin_bfgs(gc.neg_avg_log_p_y_given_x, params, gc.avg_gradient, (X, y, len(X), 1000), maxiter = 2)
print("Total time: ", end = '')
print(time() - start)


frequency = 2500  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)