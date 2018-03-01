# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:45:13 2018

@author: Erik
"""
import numpy as np
import get_data as gd
from copy import deepcopy

def get_weights():
    file = open('../data/decode_input.txt', 'r') 
    x_array = []
    params = []
    for i, elt in enumerate(file):
        if(i < 100 * 128):
            x_array.append(float(elt))
        else:
            params.append(float(elt))
    return x_array, np.array(params)

def parse_x(x):
    count = 0
    x_array = np.zeros((100, 128 * 26 + 26 * 26))
    for i in range(100):
        for j in range(128):
            for k in range(26):
                x_array[i][j + k * 128] = x[count]
            count += 1
        x_array[i][26*128: 26*128 + 26 * 26] = 1
    return x_array

def decode(params, X):
    mask = gd.get_mask()
    M = np.zeros((len(X) , 26))
    
    #now this is the same as
    #initialize--this <X0, Wa> , <X0, Wb> ... <X0, wz>
    #note that there is no Sum(T)
    M[0] = np.inner(X[0] * params, mask)
    
    for i in range(1, len(X)):
        for j in range(26):
            #add t for each
            #previous row plus all hypothetical values of t for this letter
            vect = M[i-1] + params[26 * 128 + 26 * j: 26 * 128 + (j + 1) * 26]
            M[i][j] = np.max(vect)
        #and add this vector to it
        M[i] += np.inner(X[i] * params, mask)
    return M


def value_of_y(params, X, y):
    mask = gd.get_mask()
    running_total = np.inner(X[0] * params, mask[y[0]])
    for i in range(1, len(X)):
        running_total += np.inner(X[i] * params, mask[y[i]])
        t_index = 128 * 26 + 26 * y[i]
        t_index += y[i-1]
        running_total += params[t_index]
    return running_total
    
    

def decode_bf(params, X):
    y = []
    y_fin = []
    for i in range(len(X)):
        y.append(0)
        y_fin.append(25)
    #evaluate aaa...a as initial trial solution
    best = y
    best_val = value_of_y(params, X, y)
    while True:
        
        #generate new trial solution
        i = 0
        while True:

            y[i] += 1
            if(y[i] == 26):
                y[i] = 0
                i += 1
            else:
                break
        y_val = value_of_y(params, X, y)
        if(y_val > best_val):
            best = deepcopy(y)
            best_val = y_val     
        
        
        if(y == y_fin):
            break
    return best, best_val
    

def get_solution_from_M(M, params, X):
    solution = []
    mask = gd.get_mask()
    cur_letter = np.argmax(M[-1])
    prev_best = np.max(M)
    solution.insert(0, cur_letter)
    cur_position = len(M) - 1
    
    while cur_position > 0:
        for i in range(26):
            t_index = 128 * 26 + 26 * cur_letter
            t_index += i
            diff = abs(prev_best - M[cur_position-1][i] - np.inner(X[cur_position] * params, mask[cur_letter]) - params[t_index])
            if(diff < 0.0000000001 ):
                solution.insert(0, i)
                cur_letter = i
                prev_best = M[cur_position-1][i]
                break
        
        
        cur_position -= 1
    return solution

def part1c():

    #params is np array of size (1, 128 * 26 + 26 **2)
    X, params =  get_weights()
    
    #X is np array of size (word_len, 128 * 26 + )
    X = parse_x(X)
    
    M1 = decode(params, X)
    
    best = get_solution_from_M(M1, params, X)
    
    with open("part1c.txt", "w") as text_file:
        for i, elt in enumerate(best):
            text_file.write("i= " + str(i + 1) + "  :  Predicted = " +  gd.rev_letter_lookup[elt])
            text_file.write("\n")

        