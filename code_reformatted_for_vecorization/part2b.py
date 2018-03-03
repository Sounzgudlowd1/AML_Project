# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:21:21 2018

@author: Erik
"""
import get_data as gd
import numpy as np
import time
from scipy.optimize import check_grad
import part1c as p1c

def numerator(params, X, y):
    mask = gd.get_mask()
    total = 0
    for i in range(len(X)):
        
        total += np.inner(X[i] * params, mask[y[i]])
        if(i > 0):
            t_index = 128 * 26 + y[i] * 26
            t_index += y[i-1]
            total += params[t_index]
    return np.exp(total)

def forward_propogate(params, X, label, position):
    if(position == 0):
        return 0
    
    mask = gd.get_mask()
    
    M = np.zeros((position, 26))
    X_P = X * params
    M[0] = np.inner(X_P[0], mask)
    for i in range(1, position):    
        for j in range(26):
            t_start = 128 *26 + 26 * j
            t_end = 128 *26 + 26 * (j + 1)
            vect = M[i-1] + params[t_start: t_end]
            max_vect = np.max(vect)
            vect -= max_vect
            M[i][j] =  max_vect + np.log(np.sum(np.exp(vect)))+ np.inner(X_P[i], mask[j])
    
    t_start = 128 *26 + 26 * label
    t_end = 128 *26 + 26 * (label + 1)
    return np.log(np.sum(np.exp(M[position-1] + params[t_start: t_end])))  


def back_propogate(params, X, label, position):
    if(position == len(X) - 1):
        return 0
    
    mask = gd.get_mask()
    
    M = np.zeros((len(X) - position - 1, 26))
    X_P = X * params
    
    #need to get t values for taa, tab, tac... so every 26th leter
    transpose_t = []
    for i in range(26):
        transpose_t.append(params[128 * 26 + label + 26 * i])
    transpose_t = np.array(transpose_t)
    
    #position + 1 th handled
    M[0] = np.inner(X_P[position + 1], mask) + transpose_t
    #position + 2 ... still need sto be
    for i, x in enumerate(X_P[position + 2:]):
        for j in range(26):
            t_start = 128 * 26 + j * 26
            t_end = 128 * 26 + (j + 1) * 26
            vect = M[i] + params[t_start: t_end]
            max_vect = np.max(vect)
            vect -= max_vect
            M[i + 1][j] =  max_vect + np.log(np.sum(np.exp(vect)))+ np.inner(x, mask[j])
    return np.log(np.sum(np.exp(M[-1])))  
    
def numerator_letter(params, X, label, position):
    mask = gd.get_mask()
    forward_message = forward_propogate(params, X, label, position)
    back_message = back_propogate(params, X, label, position)
    return np.exp(forward_message + back_message + np.inner(X[position] * params, mask[label]))

def numerator_letter_pair(params, X, label1, label2, position):
    mask = gd.get_mask()
    forward_message = forward_propogate(params, X, label1, position)
    back_message = back_propogate(params, X, label2, position + 1)
    t_index = 128 * 26 + 26 * label2
    t_index += label1
    local_term =  np.inner(X[position] * params, mask[label1]) + np.inner(X[position + 1] * params, mask[label2]) + params[t_index]
    return np.exp(forward_message + back_message + local_term)
    
    
def grad_wrt_wy(params, X, y, den):
    gradient = np.zeros(128 * 26)
    for i in range(26):
        #add in terms that are equal
        start = 128* i
        end = 128 * (1 + i)
        
        #for each position subtract off the probability of the letter
        for j in range(len(X)):
            if(y[j] == i):
                gradient[start: end] += X[j][:128]
            gradient[start : end] -= numerator_letter(params, X, i, j) / den * X[j][:128]
            
    return gradient

def grad_wrt_t(params, X, y, den):
    gradient = np.zeros(26 * 26)
    for i in range(26):
        for j in range(26):
            #add in terms that are equal
            t_index = 26 * j
            t_index += i
            #for each position subtract off the probability of the letter
            for k in range(len(X) - 1):
                if(y[k] == i and y[k+1] == j):
                    gradient[t_index] += 1
                gradient[t_index] -= numerator_letter_pair(params, X, i, j, k) / den
    return gradient

def gradient(params, X, y):
    den = denominator(params, X)
    return np.concatenate((grad_wrt_wy(params, X, y, den), grad_wrt_t(params, X, y, den)))
    
def denominator(params, X):
    
    mask = gd.get_mask()  
    
    M = np.zeros((len(X), 26))
    X_P = X * params
    M[0] = np.inner(X_P[0], mask)

    for i in range(1, len(X)):  
        for j in range(26):
            t_start = 128 * 26 + j * 26
            t_end = 128 * 26 + (j + 1) * 26
            vect = M[i-1] + params[t_start: t_end]
            vect_max = np.max(vect)
            vect = vect - vect_max
            M[i][j] = vect_max + np.log(np.sum(np.exp(vect))) + np.inner(X_P[i], mask[j])
        
    return np.sum(np.exp(M[-1]))

def log_p_y_given_x(params, X, y):
    return np.log(numerator(params, X, y)/denominator(params, X))

#BRUTE FORCE Calculations NOT USED IN PROD CODE
def value_of_y(params, X, y):
    mask = gd.get_mask()
    running_total = np.inner(X[0] * params, mask[y[0]])
    for i in range(1, len(X)):
        running_total += np.inner(X[i] * params, mask[y[i]])
        t_index = 128 * 26 + 26 * y[i]
        t_index += y[i-1]
        running_total += params[t_index]
    return np.exp(running_total)


def den_bf(params, X):
    y = []
    y_fin = []
    for i in range(len(X)):
        y.append(0)
        y_fin.append(25)
    
    total = value_of_y(params, X, y)
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
        total += value_of_y(params, X, y)   
        
        
        if(y == y_fin):
            break
    return total

def num_letter_bf(params, X, letter, position):
    y = np.array([0, 0, letter])
    
    total = 0
    for i in range(26):
        for j in range(26):
            y[0] = i
            y[1] = j
            total += value_of_y(params, X, y)
    return total

def num_letter_pair_bf(params, X, letter1, letter2, position):
    y = np.array([letter1, letter2, 0])
    
    total = 0
    for i in range(26):
        y[2] = i
        total += value_of_y(params, X, y)
    return total        
    
def get_params():
    file = open('../data/model.txt', 'r') 
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)
        
        
        
        
#params = np.ones(128*26 + 26 ** 2)
params = get_params()
#X_test, y_test = gd.get_xy()
start = time.time()

#print(check_grad(log_p_y_given_x, gradient, params, X_test[0], y_test[0]))
#print(forward_propogate(params, X_test[1], 0, 4))

print(log_p_y_given_x(params, X_test[1], y_test[1]))
'''
gradient_tot = 0
for k in range(len(X)):
    print(k)
    gradient_tot += gradient(params, X[k], y[k])

gradient_tot = gradient_tot / len(X)

with open("part2a.txt", "w") as text_file:
    for i, elt in enumerate(gradient_tot):
        text_file.write(str(elt))
        text_file.write("\n")
'''
print("Total time:" + str(time.time() - start))