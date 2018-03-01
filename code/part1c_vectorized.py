# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:47:36 2018

@author: Erik
"""
import numpy as np


def get_x_w_t():
    file = open('../data/decode_input.txt', 'r') 
    X = np.zeros((100, 128))
    params = np.zeros( 26*128 + 26 **2)
    count = 0
    row = 0
    col = 0
    
    for line in file:
        if(count < 128 * 100):
            if(count < 100 * 128):
                X[row][col] = line
                col += 1
                if(col == 128):
                    col = 0
                    row += 1
        else:
            params[count - 128 * 100] = line
        count +=1
    return X, params

def optimize(X, params):
    #100 letters each with a possible 26 letters
    M = np.zeros((100, 26))
    
    # initialize first row of M.  IGNORING T VALUES!!!!!!
    for i in range(26):
        start = i * 128
        end = i * 128 + 128
        M[0][i] = np.inner(params[start:end], X[0])
    
    for row in range(1, len(X)):
        for cur_letter in range(26):
            w_start = cur_letter * 128
            w_end = cur_letter * 128 + 128
            t_start = cur_letter * 26 + 26*128
            t_end = cur_letter * 26 + 26 + 26*128
            M[row][cur_letter] = np.max(M[row-1] + params[t_start: t_end] ) + np.inner(params[w_start: w_end], X[row])
                
    return M
    
    
X, params = get_x_w_t()

M = optimize(X, params)
print(np.max(M))
    

