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
def forward_propogate(w_x, t):
    word_len = len(w_x)
    #establish matrix to hold results
    M = np.zeros((word_len, 26))
    #set first row to inner <wa, x0> <wb, x0>...
    
    #iterate through length of word
    for i in range(1, word_len):
        #
        vect = M[i-1]+ t
        print(vect.shape)
        #get max
        vect_max = np.max(vect, axis = 0)
        #subtract max from vector
#        vect = vect - vect_max
        #finally set the ith word position and jth letter to the max plus the log of the vector plus the current word's value
#        M[i][j] = vect_max + np.log(np.sum(np.exp(vect + w_x[i-1]))) 
            
#    return M


X, y = gd.read_data_formatted()
params = gd.get_params()


w = gc.w_matrix(params)
w_x = np.inner( X[0], w)
t = gc.t_matrix(params)
f_mess = gc.forward_propogate(w_x, t)
b_mess = gc.back_propogate(w_x, t)
den = gc.denominator(f_mess, w_x)

true_f_prop = gc.forward_propogate(w_x, t)
print(true_f_prop[1])


test_f_prop = forward_propogate(w_x, t)


