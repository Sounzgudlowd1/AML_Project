# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:11:12 2018

@author: Erik
"""

import gradient_calculation as gc
import get_data as gd
import numpy as np
from scipy.optimize import check_grad


def forward_propogate(w_x, t):
    word_len = len(w_x)
    #establish matrix to hold results
    M = np.zeros((word_len, 26))
    #set first row to inner <wa, x0> <wb, x0>...
    M[0] = w_x[0]
    
    #iterate through length of word
    for i in range(1, word_len):
        #iterate through each letter
        for j in range(26):
            #remember t[a] = [taa, tba, tca...] so this returns all previous values for the current value + the previous row
            vect = M[i-1]+ t[j]
            #get max
            vect_max = np.max(vect)
            #subtract max from vector
            vect = vect - vect_max
            #finally set the ith word position and jth letter to the max plus the log of the vector plus the current word's value
            M[i][j] = vect_max + np.log(np.sum(np.exp(vect))) + w_x[i][j]
    return M

def back_propogate(w_x, t):
    #get the index of the final letter of the word
    fin_index = len(w_x) - 1

    #only need to go from the end to stated position
    M = np.zeros((len(w_x), 26))
    #now we need taa, tab, tac... because we are starting at the end and working backwards
    #which is exactly the transposition of the t matrix
    t_trans = t.transpose()
    
    #initialize with wa Xfinal, wb Xfinal...
    M[fin_index] = w_x[fin_index]
    for i in range(fin_index -1, -1, -1):
        for j in range(26):
            vect = M[i + 1] + t_trans[j]
            vect_max = np.max(vect)
            vect = vect - vect_max
            M[i][j] = vect_max +np.log(np.sum(np.exp(vect))) + w_x[i][j]
    return M

def num_letter_pair(w_x, t, f_mess, b_mess, position, letter1, letter2 ):
    return np.sum(np.exp(f_mess[position][letter1] 
                         + b_mess[position + 1][letter2] 
                         + t[letter2][letter1]))

def num_letter(w_x, f_mess, b_mess, position, letter):
    return np.sum(np.exp(
            f_mess[position][letter] + 
            b_mess[position][letter]
            - w_x[position][letter]))


X, y = gd.read_data_formatted()
params = gd.get_params()

w = gc.w_matrix(params)
w_x = np.inner( X[1], w)
t = gc.t_matrix(params)
f_mess, b_mess = gc.get_messages(w_x, t)
den = gc.denominator(w_x, t)




b_mess_tot = back_propogate(w_x, t)
f_mess_tot = forward_propogate(w_x, t)
print(num_letter(w_x, f_mess_tot, b_mess_tot, 4, 0))
