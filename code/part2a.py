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


'''
TODO: refactor for over and under flow issues
implement gradient
'''




def exponent_sum(X, y, w, t):
    tot = 0
    for s in range(len(y)):
        tot += np.inner(w[y[s]], X[s])
        if(s > 0):
            tot += t[y[s-1]][y[s]]
    return np.exp(tot)

def denominator(X, w, t):
    word_len = len(X)
    M = np.zeros((word_len, 26))
    
    for i in range(26):
        M[0][i] = math.log(np.exp(np.inner(w[i], X[0])))

    
    #populate M letter by letter in the word
    for s in range(1, word_len):
        #now populate each column of this row
        
        for cur_letter in range(26):
            # sum over each letter of the previous row
            letter_sum = 0
            for prev_letter in range(26):
                letter_sum += np.exp(M[s-1][prev_letter]) * ( np.exp(np.inner(w[cur_letter], X[s]) + t[prev_letter][cur_letter]  ))
            M[s][cur_letter] = math.log(letter_sum)
    
    return np.sum(np.exp(M[-1]))



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


w, t = read_model.get_w_and_t()

y_tot, X_tot = get_data.read_data_formatted()
#get_weights.print_weights(X_tot[0], w, t)
#t = np.zeros((26, 26))

out_standard = denominator(X_tot[0][:3], w, t)
out_brute_force = den_brute_force(X_tot[0][:3], w, t)
print(out_standard)
print(out_brute_force)