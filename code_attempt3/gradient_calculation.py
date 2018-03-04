# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:25:34 2018

@author: Erik
"""
import numpy as np

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
    factor = 0
    if(position > 0):
        factor += f_mess[position][letter1] 
    
    if(position < len(w_x) - 2):
        factor += b_mess[position + 1][letter2]
    
    if(position == 0):
        factor += w_x[position][letter1]
    
    if(position == len(w_x) - 2):
        factor += w_x[position + 1][letter2]
        
    return np.sum(np.exp(factor + t[letter2][letter1]))

def num_letter(w_x, f_mess, b_mess, position, letter):
    factor = 0
    if(position > 0):
        factor += f_mess[position][letter]
    
    if(position < len(w_x) -1):
        factor += b_mess[position][letter]
        
    if(position > 0 and position < len(w_x) - 1):
        factor -= w_x[position][letter]
    return np.sum(np.exp(factor))

def numerator(y, w_x, t):
    #full numerator for an entire word
    total = 0
    #go through whole word
    for i in range(len(w_x)):
        #no matter what add W[actual letter] inner Xi
        total += w_x[i][y[i]]
        if(i > 0):
            #again we have t stored as Tcur, prev
            total += t[y[i]][y[i-1]]
    return np.exp(total)


def denominator(w_x, t):
    #this is  eassy, just forward propogate to the end of the word and return the sum of the exponentials
    return np.sum(np.exp(forward_propogate(w_x, t)[-1]))       


#split up params into w and t.  Note that this only needs to happen once per word!!! do not calculate per letter
def w_matrix(params):
    w = np.zeros((26, 128))
    for i in range(26):
        w[i] =  params[128 *  i : 128 * (i +1)]
    return w

def t_matrix(params):
    t = np.zeros((26, 26))
    count = 0
    for i in range(26):
        for j in range(26):
            #want to be able to say t[0] and get all values of Taa, tba, tca...
            t[i][j] = params[128 * 26 + count]
            count += 1
    return t
    
def grad_wrt_wy(X, y, w_x, t, f_mess, b_mess, den):
    gradient = np.zeros(128 * 26)
    for i in range(26):
        #add in terms that are equal
        start = 128* i
        end = 128 * (1 + i)
        
        #for each position subtract off the probability of the letter
        for j in range(len(X)):
            if(y[j] == i):
                gradient[start: end] += X[j]
            gradient[start : end] -= num_letter(w_x, f_mess, b_mess, j, i) / den * X[j]
            
    return gradient
    
def grad_wrt_t(y, w_x, t, f_mess, b_mess, den):
    gradient = np.zeros(26 * 26)
    for i in range(26):
        for j in range(26):
            #add in terms that are equal
            t_index = 26 * j
            t_index += i
            #for each position subtract off the probability of the letter pair
            for k in range(len(w_x) - 1):
                if(y[k] == i and y[k+1] == j):
                    gradient[t_index] += 1
                gradient[t_index] -= num_letter_pair(w_x, t, f_mess, b_mess, k, i, j) / den
    return gradient

def gradient_word(X, y, w, t, word_num):
    w_x = np.inner(X[word_num], w)
    f_mess = forward_propogate(w_x, t)
    b_mess = back_propogate(w_x, t)
    den = denominator(w_x, t)
    wy_grad = grad_wrt_wy(X[word_num], y[word_num], w_x, t, f_mess, b_mess, den)
    t_grad = grad_wrt_t(y[word_num], w_x, t, f_mess, b_mess, den)
    return np.concatenate((wy_grad, t_grad))
    
def avg_gradient(params, X, y, up_to_index):
    w = w_matrix(params)
    t = t_matrix(params)
        
    count = 0
    total = np.zeros(128 * 26 + 26 ** 2)
    for i in range(up_to_index):
        total += gradient_word(X, y, w, t, i)
        count += 1
    return total / count        
        
def log_p_y_given_x(w_x, y, t, word_num):
    return np.log(numerator(y, w_x, t) / denominator(w_x, t))


def avg_log_p_y_given_x(params, X, y, up_to_index):
    w = w_matrix(params)
    t = t_matrix(params)
        
    count = 0
    total = 0
    for i in range(up_to_index):
        w_x = np.inner(X[i], w)
        count += 1
        total += log_p_y_given_x(w_x, y[i], t, i)
    return total / count




