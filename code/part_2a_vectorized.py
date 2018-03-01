# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:41:04 2018

@author: Erik
"""

import read_model as rm
import get_data as gd
import numpy as np
import math
from scipy.optimize import check_grad

#goal to sum accross all possible values of y^m
def denominator(params, X):
    word_len = len(X)
    M = np.zeros((word_len, 26))
    
    
    for letter in range(26):
        start = letter * 128
        end = letter * 128 + 128
        M[0][letter] = np.inner(params[start: end], X[0])
    
    for row in range(1, word_len):
        for letter in range(26):
            w_start = letter * 128
            w_end = letter * 128 + 128
            t_start = letter * 26 + 26 * 128
            t_end = letter * 26 + 26 + 26 * 128
            vect = M[row-1] + np.inner(params[w_start: w_end], X[row]) + params[t_start: t_end]
            vect_max = np.max(vect)
            M[row][letter] = math.log(np.sum(np.exp(vect - vect_max))) + vect_max
    
    return np.sum(np.exp(M[word_len - 1]))

def numerator(params, X, y):
    running_total = 0
    for i, letter in enumerate(y):
        w_start = letter * 128
        w_end = letter * 128 + 128
        running_total += np.inner(X[i], params[w_start: w_end])
        
        
        if(i > 0):
            #t in form T11, T21, T31... to get T
            #so anything in the neighborhood of ta... would start at 26*128
            #figure out where t[previous] starts
            t_index = letter * 26 +  26 * 128
            #now add this letter to it
            t_index += y[i -1]
            running_total += params[t_index]
        
    return np.exp(running_total)

def p_y_given_x(params, X, y):
    return numerator(params, X, y)/denominator(params, X)

def log_p_y_given_x(params, X, y):
    return math.log(p_y_given_x(params, X, y))


def numerator_letter_pair_at_position_s(params, X, l1, l2, s):
    word_len = len(X)
    M = np.zeros((s, 26))
    if(s > 0):
        #Compute factor up to the sth position
        for letter in range(26):
            start = letter * 128
            end = letter * 128 + 128
            M[0][letter] = np.inner(params[start: end], X[0])
        
        for row in range(1, s):
            for letter in range(26):
                w_start = letter * 128
                w_end = letter * 128 + 128
                t_start = letter * 26 + 26 * 128
                t_end = letter * 26 + 26 + 26 * 128
                vect = M[row-1] + np.inner(params[w_start: w_end], X[row]) + params[t_start: t_end]
                vect_max = np.max(vect)
                M[row][letter] = math.log(np.sum(np.exp(vect - vect_max))) + vect_max
        #finished computing up to the sth position
        
        #now add the factor associated with the sth position, since sth position is known, just addit in
        t_start = l1 * 26 + 26 * 128
        t_end = l1 * 26 + 26 + 26 * 128
        w_start = l1 * 128
        w_end = l1 * 128 + 128
        vect = M[-1] + params[t_start: t_end]
        cumulative_factor = np.log(np.sum(np.exp(vect))) + np.inner(params[w_start:w_end], X[s])
        
        #deal with second letter
        w_start = l2 * 128
        w_end = l2 * 128 + 128
        t_index = 128 * 26 + l2 * 26
        t_index += l1
        cumulative_factor += np.inner(params[w_start: w_end], X[s+1]) + params[t_index]
    else:
        w_start = l1 * 128
        w_end = l1 * 128 + 128
        cumulative_factor = np.inner(params[w_start:w_end], X[s])
        
        #deal with second letter
        w_start = l2 * 128
        w_end = l2 * 128 + 128
        t_index = 128 * 26 + l2 * 26
        t_index += l1
        cumulative_factor += np.inner(params[w_start: w_end], X[s+1]) + params[t_index]
        
    #deal with part after s+ 1
        #now deal with part after s
    if(s < word_len - 2):
        #Create array to hold intermediate results
        M = np.zeros((word_len - s -2, 26))
        #populate the first row of the array with the s+1st information
        for letter in range(26):
            #if y is 'a' then we need Taa Tab Tac..., locat at positions 26*128, 26*128+26...
            t_index = 26*128 + l2 + letter * 26
            w_start = letter * 128
            w_end = letter * 128 + 128
            
            M[0][letter] = np.inner(params[w_start: w_end], X[s + 2]) + cumulative_factor + params[t_index]
        
        
        for row in range(1, word_len - s -2):
            for letter in range(26):
                w_start = letter * 128
                w_end = letter * 128 + 128
                t_start = letter * 26 + 26 * 128
                t_end = letter * 26 + 26 + 26 * 128
                vect = M[row - 1] + np.inner(params[w_start: w_end], X[row + s + 2]) + params[t_start: t_end]
                vect_max = np.max(vect)
                M[row][letter] = math.log(np.sum(np.exp(vect - vect_max))) + vect_max
        cumulative_factor = math.log(np.sum(np.exp(M[ -1])))
        #finished computing up to the sth position
    
    return np.exp(cumulative_factor)
        
def p_letter_pair_at_position_s(params, X, l1, l2, s):
    return numerator_letter_pair_at_position_s(params, X, l1, l2, 0)/denominator(params, X_tot[1])

#Need to sum out all except the given position and letter
def numerator_y_at_position_s(params, X, y, s):
    word_len = len(X)
    M = np.zeros((s, 26))
    if(s > 0):
        #Compute factor up to the sth position
        for letter in range(26):
            start = letter * 128
            end = letter * 128 + 128
            M[0][letter] = np.inner(params[start: end], X[0])
        
        for row in range(1, s):
            for letter in range(26):
                w_start = letter * 128
                w_end = letter * 128 + 128
                t_start = letter * 26 + 26 * 128
                t_end = letter * 26 + 26 + 26 * 128
                vect = M[row-1] + np.inner(params[w_start: w_end], X[row]) + params[t_start: t_end]
                vect_max = np.max(vect)
                M[row][letter] = math.log(np.sum(np.exp(vect - vect_max))) + vect_max
        #finished computing up to the sth position
        
        #now add the factor associated with the sth position, since sth position is known, just addit in
        t_start = y * 26 + 26 * 128
        t_end = y * 26 + 26 + 26 * 128
        w_start = y * 128
        w_end = y * 128 + 128
        vect = M[-1] + params[t_start: t_end]
        cumulative_factor = np.log(np.sum(np.exp(vect))) + np.inner(params[w_start:w_end], X[s])
    else:
        w_start = y * 128
        w_end = y * 128 + 128
        cumulative_factor = np.inner(params[w_start:w_end], X[s])
    
    
    #now deal with part after s
    if(s < word_len - 1):
        #Create array to hold intermediate results
        M = np.zeros((word_len - s -1, 26))
        #populate the first row of the array with the s+1st information
        for letter in range(26):
            #if y is 'a' then we need Taa Tab Tac..., locat at positions 26*128, 26*128+26...
            t_index = 26*128 + y + letter * 26
            w_start = letter * 128
            w_end = letter * 128 + 128
            
            M[0][letter] = np.inner(params[w_start: w_end], X[s + 1]) + cumulative_factor + params[t_index]
        
        
        for row in range(1, word_len - s -1):
            for letter in range(26):
                w_start = letter * 128
                w_end = letter * 128 + 128
                t_start = letter * 26 + 26 * 128
                t_end = letter * 26 + 26 + 26 * 128
                vect = M[row - 1] + np.inner(params[w_start: w_end], X[row + s + 1]) + params[t_start: t_end]
                vect_max = np.max(vect)
                M[row][letter] = math.log(np.sum(np.exp(vect - vect_max))) + vect_max
        cumulative_factor = math.log(np.sum(np.exp(M[ -1])))
        #finished computing up to the sth position
    
    return np.exp(cumulative_factor)


def p_sth_letter_is_y(params, X, y, s):
    return numerator_y_at_position_s(params, X, y, s) / denominator(params, X)
    

def gradient_wrt_wy(params,  X, y):
    gradient = np.zeros(128*26)
  
    for i in range(len(X)):
        w_start = y[i] * 128
        w_end = y[i] * 128 + 128          
        gradient[w_start : w_end] += X[i]
    
        for j in range(26):
            w_start = j * 128
            w_end = j * 128 + 128        
            gradient[w_start: w_end] -= p_sth_letter_is_y(params, X, j, i) * X[i]
    return gradient

def gradient_wrt_t(params, X, y):
    gradient = np.zeros(26 * 26)
    for i in range(1, len(X)):
        #taa += 1 if y[0]= a and y[1] = a
        t_index = y[i] * 26
        t_index += y[i-1]
        gradient[t_index] += 1
        for l1 in range(26):#first letter
            for l2 in range(26): #second letter
                t_index =  l2 * 26
                t_index += l1
                gradient[t_index] -= p_letter_pair_at_position_s(params, X, l1, l2, i-1)  
    return gradient

def gradient(params, X, y):
    return np.concatenate((gradient_wrt_wy(params, X, y), gradient_wrt_t(params, X, y)))

params = rm.read_data()
#params = np.ones(26*128 + 26 * 26)
y_tot, X_tot = gd.read_data_formatted()

print(denominator(params, X_tot[0]))
