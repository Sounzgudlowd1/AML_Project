# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:25:34 2018

@author: Erik
"""
import numpy as np
import time

def forward_propogate(w_x, t, position):
    #can't forward propogate to positino 0
    if position == 0:
        return 0
    
    #establish matrix to hold results
    M = np.zeros((position, 26))
    #set first row to inner <wa, x0> <wb, x0>...
    M[0] = w_x[0]
    
    #iterate through length of word
    for i in range(1, position):
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
    return M[-1]

def back_propogate(w_x, t, position):
    #this works but could be clearer
    
    #get the index of the final letter of the word
    fin_index = len(w_x) - 1
    
    #can't back propogate to the final letter, so return if nothing
    if position == fin_index:
        return 0 
    
    #only need to go from the end to stated position
    M = np.zeros((fin_index - position, 26))
    #now we need taa, tab, tac... because we are starting at the end and working backwards
    #which is exactly the transposition of the t matrix
    t_trans = t.transpose()
    
    #initialize with wa Xfinal, wb Xfinal...
    M[0] = w_x[fin_index]
    #used to tract position in M matrix
    cur = 0
    for i in range(fin_index - position -1, 0, -1):
        for j in range(26):
            #very similar to forward propigation, get the M[i-1th] row and add the tranposition term
            #so this is something like Xfin wa + taa + Xfin wb + tab... if I am currently working on the letter 'a'
            vect = M[cur] + t_trans[j]
            #same tricks for numerical robustnesss
            vect_max = np.max(vect)
            vect = vect - vect_max
            #indices are a ltttle cluttered but functionally the same as forward propogation
            M[cur + 1][j] = vect_max +np.log(np.sum(np.exp(vect))) + w_x[fin_index - cur - 1][j]
        cur += 1
    #return final row.  This makes no assumptions on the value of the letter at the stated position
    return M[-1]

def numerator_letter(w_x, t, forward_messages, back_messages, letter, position):
    #figure out the numerator for an 'a' being at position 3 for instance
    
    #translate messages for this letter
    
    factor = 0
    
    #determine if its the first letter of word, last letter or somewhere in the middle
    #if first or last need to ignore forward and backward message respectively
    if position > 0:
        left = forward_messages[position] + t[letter]    
        factor += np.log(np.sum(np.exp(left)))
    
    if position < len(w_x) -1:
        right = back_messages[position] + t.transpose()[letter]
        factor += np.log(np.sum(np.exp(right)))
    
    return np.exp(factor + w_x[position][letter])
    

def numerator_letter_pair(w_x, t, forward_message, back_message, letter1, letter2, position):
    #this determines the numerator for 'kz' being in position 4 for instance
    
    #get left message and add t to it
    
    factor = 0
    #if this is the end of the word ignore right message
    if position < len(w_x) - 2:
        right = back_message[position+1] + t.transpose()[letter2]
        factor += np.log(np.sum(np.exp(right)))
    
    if position > 0:
        left = forward_message[position] + t[letter1]
        factor += np.log(np.sum(np.exp(left)))
    
    #now return the factor so far + wletter1 xposition + wletter2 Xposition + 1 + the transition from letter1 to letter2 
    return np.exp(factor + w_x[position][letter1] + w_x[position + 1][letter2] + t[letter2][letter1] )

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
    return np.sum(np.exp(forward_propogate(w_x, t, len(w_x))))


def get_messages(w_x, t):
    #this just pre-calculates the messages for each position of the word
    #later you can reference the precalculated values with extreme reduction of runtime
    f_messages = []
    b_messages = []
    for i in range(len(w_x)):
        f_messages.append(forward_propogate(w_x, t, i))
        b_messages.append(back_propogate(w_x, t, i))
    return f_messages, b_messages        

       


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
            #for each position subtract off the probability of the letter pair
            for k in range(len(w_x) - 1):
                if(y[k] == i and y[k+1] == j):
                    gradient[t_index] += 1
                gradient[t_index] -= numerator_letter_pair(w_x, t, f_mess, b_mess, i, j, k) / den
    return gradient

def gradient_word(X, y, w, t, word_num):
    w_x = np.inner(X[word_num], w)
    f_mess, b_mess = get_messages(w_x, t)
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




