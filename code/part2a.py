# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:10:45 2018

@author: Erik
"""
import get_weights
import get_data
import numpy as np
import math
import read_model
import common_utilities
from scipy.optimize import check_grad
import time


def numerator(X, y, w, t):
    tot = 0
    for s in range(len(y)):
        tot += np.inner(w[y[s]], X[s])
        if(s > 0):
            tot += t[y[s-1]][y[s]]
    return np.exp(tot)



def denominator(X, w, t):
    word_len = len(X)
    M = np.zeros((word_len, 26))
    
    #initialize first row
    M[0] = np.inner(w, X[0])

    
    #populate M letter by letter in the word
    for s in range(1, word_len):
        #now populate each column of this row
        for cur_letter in range(26):
            t_vect = np.transpose(t)[cur_letter]
            #create vector to store sum of all previous values + t[prev][cur] + inner product of w and X
            temp = M[s-1] + np.inner(w[cur_letter], X[s]) + t_vect
            max_fact = np.max(temp)
            temp = temp - max_fact
            M[s][cur_letter] = max_fact + math.log(np.sum(np.exp(temp)))

    print(M[1][0])  
    print(M[0][0])
    return np.sum(np.exp(M[-1]))

def p_y_given_x(X, y, w, t):
    return numerator(X, y, w, t) / denominator(X, w, t)

def log_p_y_given_x(params):
    y_tot, X_tot = get_data.read_data_formatted()
    X = X_tot[0]
    y = y_tot[0]
    w, t = read_model.get_w_and_t(params)
    return math.log(numerator(X, y, w, t) / denominator(X, w, t))

def grad_wrt_wy(params):
    y_tot, X_tot = get_data.read_data_formatted()
    X = X_tot[0]
    y = y_tot[0]
    w, t = read_model.get_w_and_t(params)
    
    gradient = np.zeros((26, 128))
    probability_of_word = p_y_given_x(X, y, w, t) 
    
    #kind of a strange representation.  Need to calculate wy = ([ys = y] - p(y|x)) * xs
    #which is equivilent to subtracting the probability of the word * xs from all wy
    #then adding back 1 * Xs for the weight that is actually associated with Y.
    for s in range(len(y)):
        gradient -= probability_of_word * X[s]
        gradient[y[s]] += X[s]

        
    return gradient
    
def total_grad(X, y, w, t):
    gradient = np.zeros((26, 128))
    for i in range(len(y)):
        gradient += grad_wrt_wy(X[i], y[i], w, t)
    return gradient


def den_brute_force(X, w, t):
    final_y = []
    y_prime = []
    #initialize stopping criteria and trial solutions
    for i in range(len(X)):
        final_y.append(25)
        y_prime.append(0)
    
    final_sum = 0
    
    #first solution is [0, 0] so calculate value
    while(True):
            
        #for this trial solution add up the exponents then raise e^that
        exp_sum = 0
        for i in range(len(X)):
            exp_sum += np.inner(w[y_prime[i]], X[i])
            if(i > 0):
                exp_sum += t[y_prime[i-1]][y_prime[i]]
                
        final_sum += np.exp(exp_sum)

        
        if(y_prime == final_y):
            break
        
        #create next trial solution
        j = 0
        while(True):
            y_prime[j] += 1
            if(y_prime[j]== 26):
                y_prime[j] = 0
                j += 1
            else:
                break
    return final_sum




#get_weights.print_weights(X_tot[0], w, t)
#t = np.zeros((26, 26))
params = read_model.read_data()
#params = np.ones(128*26 + 26 **2)
w, t = read_model.get_w_and_t(params)

y_tot, X_tot = get_data.read_data_formatted()

#print(numerator(X_tot[0], y_tot[0], w, t))
print(denominator(X[0], w, t))