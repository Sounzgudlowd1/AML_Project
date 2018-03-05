# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:11:12 2018

@author: Erik
"""

import gradient_calculation as gc
import get_data as gd
import numpy as np
from scipy.optimize import check_grad


def func_to_minimize(params, X, y, C):
    num_examples = len(X)
    reg = 1/2 * np.sum(params ** 2)
    sum_prob = gc.log_p_y_given_x_sum(params, X, y, num_examples)
    return -C/len(X) * sum_prob + reg

