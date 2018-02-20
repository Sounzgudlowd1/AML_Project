# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:10:01 2018

@author: Erik
"""
import numpy as np
import csv

def get_weights():
    file = open('../data/decode_input.txt', 'r') 
    x_array = []
    w_array = []
    t_array = []
    for i, elt in enumerate(file):
        if(i < 100 * 128):
            x_array.append(elt)
        elif(i < 100 * 128 + 128 * 26):
            w_array.append(elt)
        else:
            t_array.append(elt)
    return x_array, w_array, t_array

def parse_x(x):
    count = 0
    x_array = np.zeros((100, 128))
    for i in range(100):
        for j in range(128):
            x_array[i][j] = x[count]
            count += 1
    return x_array

def parse_w(w):
    w_array = np.zeros((26, 128))
    count = 0
    for i in range(26):
        for j in range(128):
            w_array[i][j] = w[count]
            count += 1
    return w_array

def parse_t(t):
    t_array = np.zeros((26, 26))
    count = 0
    for i in range(26):
        for j in range(26):
            #this is actuqlly right.  it goes T11, T21, T31...
            t_array[j][i] = t[count]
            count += 1
    return t_array

def get_weights_formatted():
    x, w, t = get_weights()
    x_array = parse_x(x)
    w_array = parse_w(w)
    t_array = parse_t(t)
    return x_array, w_array, t_array

def print_weights(X, w, t):
    with open('X.csv', 'w', newline = '') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for x in X:
            spamwriter.writerow(x)
    
    with open('w.csv', 'w', newline = '') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in w:
            spamwriter.writerow(row)
            
    with open('t.csv', 'w', newline = '') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in t:
            spamwriter.writerow(row)