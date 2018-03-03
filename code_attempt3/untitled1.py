# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:25:34 2018

@author: Erik
"""

import get_data as gd
import numpy as np
from scipy.optimize import check_grad
from time import time

def forward_propogate(w_x, t, position):
    
    if position == 0:
        return 0
    
    M = np.zeros((position, 26))
    M[0] = w_x[0]
    for i in range(1, position):
        for j in range(26):
            vect = M[i-1]+ t[j]
            vect_max = np.max(vect)
            vect = vect - vect_max
            M[i][j] = vect_max + np.log(np.sum(np.exp(vect))) + w_x[i][j]
    return M[-1]

def back_propogate(w_x, t, position):
    fin_index = len(w_x) - 1
    if position == fin_index:
        return 0 
    M = np.zeros((fin_index - position, 26))
    t_trans = t.transpose()
    M[0] = w_x[fin_index]
    cur = 0
    for i in range(fin_index - position -1, 0, -1):
        for j in range(26):
            vect = M[cur] + t_trans[j]
            vect_max = np.max(vect)
            vect = vect - vect_max
            M[cur + 1][j] = vect_max +np.log(np.sum(np.exp(vect))) + w_x[fin_index - cur - 1][j]
        cur += 1
    return M[-1]

def numerator_letter(w_x, t, forward_messages, back_messages, letter, position):
    left = forward_messages[position] + t[letter]
    right = back_messages[position] + t.transpose()[letter]
    
    if position == 0:
        factor = np.log(np.sum(np.exp(right)))
    elif position == len(w_x) -1:
        factor = np.log(np.sum(np.exp(left)))
    else: 
        factor = np.log(np.sum(np.exp(left))) + np.log(np.sum(np.exp(right)))
    
    return np.exp(factor + w_x[position][letter])
    

def numerator_letter_pair(w_x, t, forward_message, back_message, letter1, letter2, position):
    left = forward_message[position] + t[letter1]

    

    if position == len(w_x) - 2:
        factor = np.log(np.sum(np.exp(left)))
    else:
        right = back_message[position+1] + t.transpose()[letter2]
        if position == 0:
            factor = np.log(np.sum(np.exp(right)))
        else: 
            factor = np.log(np.sum(np.exp(left))) + np.log(np.sum(np.exp(right)))
    
    return np.exp(factor + w_x[position][letter1] + w_x[position + 1][letter2] + t[letter2][letter1] )

def numerator(y, w_x, t):
    total = 0
    for i in range(len(w_x)):
        total += w_x[i][y[i]]
        if(i > 0):
            total += t[y[i]][y[i-1]]
    return np.exp(total)


def denominator(w_x, t):
    return np.sum(np.exp(forward_propogate(w_x, t, len(w_x))))


def get_forward_messages(w_x, t):
    #this just pre-calculates the messages for each position of the word
    #later you can reference the precalculated values with extreme reduction of runtime
    messages = []
    for i in range(len(w_x)):
        messages.append(forward_propogate(w_x, t, i))
    return messages        

def get_back_messages(w_x, t):
    #this just pre-calculates the messages for each position of the word
    #later you can reference the precalculated values with extreme reduction of runtime
    messages = []
    for i in range(len(w_x)):
        messages.append(back_propogate(w_x, t, i))
    return messages        



def get_params():
    file = open('../data/model.txt', 'r') 
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)

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
            gradient[start : end] -= numerator_letter(w_x, t, f_mess, b_mess, i, j) / den * X[j]
            
    return gradient
    
def grad_wrt_t(y, w_x, t, f_mess, b_mess, den):
    gradient = np.zeros(26 * 26)
    for i in range(26):
        for j in range(26):
            #add in terms that are equal
            t_index = 26 * j
            t_index += i
            #for each position subtract off the probability of the letter
            for k in range(len(w_x) - 1):
                if(y[k] == i and y[k+1] == j):
                    gradient[t_index] += 1
                gradient[t_index] -= numerator_letter_pair(w_x, t, f_mess, b_mess, i, j, k) / den
    return gradient

def gradient_word(X, y, w, t, word_num):
    w_x = np.inner(X[word_num], w)
    f_mess = get_forward_messages(w_x, t)
    b_mess = get_back_messages(w_x, t)
    den = denominator(w_x, t)
    wy_grad = grad_wrt_wy(X[word_num], y[word_num], w_x, t, f_mess, b_mess, den)
    t_grad = grad_wrt_t(y[word_num], w_x, t, f_mess, b_mess, den)
    return np.concatenate((wy_grad, t_grad))
    
def avg_gradient(params, X, y):
    w = w_matrix(params)
    t = t_matrix(params)
        
    count = 0
    total = np.zeros(128 * 26 + 26 ** 2)
    for i in range(len(y)):
        total += gradient_word(X, y, w, t, i)
        count += 1
    return total / count        
        
def log_p_y_given_x(w_x, y, t, word_num):
    return np.log(numerator(y, w_x, t) / denominator(w_x, t))


def avg_p_y_given_x(params, X, y):
    w = w_matrix(params)
    t = t_matrix(params)
        
    count = 0
    total = 0
    for i in range(1, 3):
        w_x = np.inner(X[i], w)
        count += 1
        total += log_p_y_given_x(w_x, y[i], t, i)
    return total / count


X, y = gd.read_data_formatted()
params = get_params()

start = time()
av_grad = avg_gradient(params, X, y)
print("Total time:")
print(time() - start)


with open("part2a.txt", "w") as text_file:
    for i, elt in enumerate(av_grad):
        text_file.write(str(elt))
        text_file.write("\n")