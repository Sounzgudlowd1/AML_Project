# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:02:06 2018

@author: Erik
"""

import numpy as np
import re


def read_data():
    file = open('../data/model.txt', 'r') 
    w = np.zeros(128*26)
    t = np.zeros(26 *26)
    count = 0
    for line in file:
        if(count < 128 * 26):
            w[count] =  line
        else:
            t[count - 128*26] = line
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

def get_w_and_t():
    w, t = read_data()

    return parse_w(w), parse_t(t)
