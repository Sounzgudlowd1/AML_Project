# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:11:12 2018

@author: Erik
"""

import gradient_calculation as gc
import get_data as gd
import numpy as np
from scipy.optimize import check_grad
import part2b_code as p2b
import time


params = gd.get_params()
X, y = gd.read_data_formatted('train_struct.txt')

X = X[0:100]
y = y[0:100]

start = time.time()
print(check_grad(p2b.func_to_minimize, p2b.grad_func, params, X, y, 10))
print("Finished in " + str(time.time() - start) + " seconds")

