# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:11:12 2018

@author: Erik
"""

import gradient_calculation as gc
import get_data as gd
import numpy as np
from scipy.optimize import check_grad

def grad_wrt_t(y, w_x, t, f_mess, b_mess, den):
    gradient = np.zeros(26 * 26)
    for i in range(2):
        for j in range(26):
            gradient[j * 26 : (j + 1) * 26] -= np.exp(w_x[i] + t[j] + w_x[i + 1][j] +b_mess[i + 1][j] + f_mess[i])
    
    gradient /= den
                
    for i in range(len(w_x) - 1):
        t_index = y[i]
        t_index += 26 * y[i+1]
        gradient[t_index] += 1        
        
        
    return gradient


X, y = gd.read_data_formatted()
params = gd.get_params()


w = gc.w_matrix(params)
w_x = np.inner( X[1][:3], w)
t = gc.t_matrix(params)
f_mess = gc.forward_propogate(w_x, t)
b_mess = gc.back_propogate(w_x, t)
den = gc.denominator(f_mess, w_x)

true_grad = gc.grad_wrt_t(y[1][:3], w_x, t, f_mess, b_mess, den) 
print(true_grad[0])


test_grad = grad_wrt_t(y[1][:3], w_x, t, f_mess, b_mess, den) 
print(test_grad[0])
# + ))


