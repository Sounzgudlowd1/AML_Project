# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:11:12 2018

@author: Erik
"""

import gradient_calculation as gc
import get_data as gd
import numpy as np
from scipy.optimize import check_grad

''' do this next '''
def grad_wrt_wy(X, y, w_x, t, f_mess, b_mess, den):
    gradient = np.zeros(128 * 26)
    
    for i in range(len(w_x)):
        
            
    return gradient


X, y = gd.read_data_formatted()
params = gd.get_params()


w = gc.w_matrix(params)
w_x = np.inner( X[1], w)
t = gc.t_matrix(params)
f_mess = gc.forward_propogate(w_x, t)
b_mess = gc.back_propogate(w_x, t)
den = gc.denominator(f_mess, w_x)

