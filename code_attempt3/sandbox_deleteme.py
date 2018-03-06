# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:11:12 2018

@author: Erik
"""

import gradient_calculation as gc
import get_data as gd
import numpy as np
from scipy.optimize import check_grad


X, y = gd.read_data_formatted('train')
params = gd.get_params()
t = gc.t_matrix(params)
w = gc.w_matrix(params)

f_mess = gc.forward_propogate(w_x, t)
b_mess = gc.back_propogate(w_x, t)
den = gc.denominator(f_mess, w_x)

print(gc.log_p_y_given_x_avg(params, X, y, len(X)))