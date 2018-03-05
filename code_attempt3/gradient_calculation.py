# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:25:34 2018

@author: Erik
"""
import numpy as np
import time

def forward_propogate(w_x, t):
    word_len = len(w_x)
    #establish matrix to hold results
    M = np.zeros((word_len, 26))
    #set first row to inner <wa, x0> <wb, x0>...
    
    #iterate through length of word
    for i in range(1, word_len):
        vect = M[i-1] + t
        vect_max = np.max(vect, axis = 1)
        vect = (vect.transpose() - vect_max).transpose()
        M[i] = vect_max + np.log(np.sum(np.exp(vect + w_x[i-1]), axis = 1))
            
    return M

def back_propogate(w_x, t):
    #get the index of the final letter of the word
    fin_index = len(w_x) - 1

    #only need to go from the end to stated position
    M = np.zeros((len(w_x), 26))
    #now we need taa, tab, tac... because we are starting at the end and working backwards
    #which is exactly the transposition of the t matrix
    t_trans = t.transpose()

    for i in range(fin_index -1, -1, -1):
        vect = M[i + 1] + t_trans
        vect_max = np.max(vect, axis = 1)
        vect = (vect.transpose() - vect_max).transpose()
        M[i] = vect_max +np.log(np.sum(np.exp(vect + w_x[i+1]), axis =  1)) 
    return M


def num_letter(w_x, f_mess, b_mess, position, letter):
    factor = 0
    if(position > 0):
        factor += f_mess[position][letter]
    
    if(position < len(w_x) -1):
        factor += b_mess[position][letter]
        
    return np.exp(factor + w_x[position][letter])

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


def denominator(f_message, w_x):
    #this is  eassy, just forward propogate to the end of the word and return the sum of the exponentials
    return np.sum(np.exp(f_message[-1] + w_x[-1]))       


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
    for j in range(len(X)):
    
        #add in terms that are equal

        
        #for each position subtract off the probability of the letter
        for i in range(26):
            start = 128* i
            end = 128 * (1 + i)
            if(y[j] == i):
                gradient[start: end] += X[j]
                
            gradient[start : end] -= num_letter(w_x, f_mess, b_mess, j, i) / den * X[j]
            
    return gradient
    
def grad_wrt_t(y, w_x, t, f_mess, b_mess, den):
    gradient = np.zeros(26 * 26)
    for i in range(len(w_x) -1):
        for j in range(26):
            gradient[j * 26 : (j + 1) * 26] -= np.exp(w_x[i] + t[j] + w_x[i + 1][j] +b_mess[i + 1][j] + f_mess[i])
    
    gradient /= den
                
    for i in range(len(w_x) - 1):
        t_index = y[i]
        t_index += 26 * y[i+1]
        gradient[t_index] += 1        
        
        
    return gradient

def gradient_word(X, y, w, t, word_num):
    #print()
    #O(n|Y|)
    #start = time.time()
    w_x = np.inner(X[word_num], w)
    #print("w_x time : " + str(time.time() - start))
    #O(n{Y}^2)
    
    #start = time.time()
    f_mess = forward_propogate(w_x, t)
    #print("f_prop_time : " + str(time.time() - start))
    
    #O(n{Y}^2)
    #start = time.time()
    b_mess = back_propogate(w_x, t)
    #print("b_prop_time : " + str(time.time() - start))
    
    #O(1)
    #start = time.time()
    den = denominator(f_mess, w_x)
    #print("den_time : " + str(time.time() - start))
    
    
    #O(n|Y|^2)
    #start = time.time()
    wy_grad = grad_wrt_wy(X[word_num], y[word_num], w_x, t, f_mess, b_mess, den)
    #print("grad_wy time : " + str(time.time() - start))
    
    
    
    #O(n|Y|^2)
    #start = time.time()
    t_grad = grad_wrt_t(y[word_num], w_x, t, f_mess, b_mess, den)
    #print("grad_t time : " + str(time.time() - start))
    
    
    return np.concatenate((wy_grad, t_grad))
    
def gradient_sum(params, X, y, up_to_index):
    w = w_matrix(params)
    t = t_matrix(params)
        
    total = np.zeros(128 * 26 + 26 ** 2)
    for i in range(up_to_index):
        total += gradient_word(X, y, w, t, i)
    return total      
        
def log_p_y_given_x(w_x, y, t, word_num):
    f_mess = forward_propogate(w_x, t)
    return np.log(numerator(y, w_x, t) / denominator(f_mess, w_x))


def log_p_y_given_x_sum(params, X, y, up_to_index):
    w = w_matrix(params)
    t = t_matrix(params)
        
    total = 0
    for i in range(up_to_index):
        w_x = np.inner(X[i], w)
        total += log_p_y_given_x(w_x, y[i], t, i)
    return  total




