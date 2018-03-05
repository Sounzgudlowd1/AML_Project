# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:11:12 2018

@author: Erik
"""

import gradient_calculation as gc
import get_data as gd
import numpy as np
from scipy.optimize import check_grad

def func(x):
    return 0.5 *(x[0] * x[0] + x[1] * x[1])

def grad(x):
    return x


print(check_grad(func, grad, np.array( [10, 10])))