# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:02:06 2018

@author: Erik
"""

import numpy as np


def read_data():
    file = open('../data/model.txt', 'r') 
    params = np.zeros(26*128 + 26*26)
    for i, line in enumerate(file):
        params[i] = line
    return np.array(params)


#you really shouldn't call these for production level code, just use the above function
    #Mainly you just don't want to have separate vecotrs for each wy and tyy+1, it really just
    #needs to be a giant vector of size 26*128+26^2 = 4004
def split_w_t(params):
    w = np.zeros(128*26)
    t = np.zeros(26 *26)
    count = 0
    for elt in params:
        if(count < 128 * 26):
            w[count] =  elt
        else:
            t[count - 128*26] = elt
        count += 1
    return w, t

def parse_w(w):
    w_fin = np.zeros((26,128))
    count = 0
    for i in range(26):
        for j in range(128):
            w_fin[i][j] = w[count]
            count += 1
    return w_fin

def parse_t(t):
    t_fin = np.zeros((26, 26))
    count = 0
    for i in range(26):
        for j in range(26):
            t_fin[j][i] = t[count]
            count += 1
    return t_fin

def get_w_and_t(params):
    w, t = split_w_t(params)
    return parse_w(w), parse_t(t)
