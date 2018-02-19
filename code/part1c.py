# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:06:02 2018

@author: Erik
"""

import common_utilities
import get_weights
import numpy as np
from copy import deepcopy


def optimize(X, w, t):
    #for 100 letters I need an M matrix of 100  X 26
    M = np.zeros((len(X), 26))
    
    #populates first row
    for j in range(26):
        M[0][j] = np.inner(X[0], w[j])
    
    #go row wise through M matrix, starting at line 2 since first line is populated
    for row in range(1, len(X)):
        
        #go column wise, populating the best sum of the previous + T[previous letter][
        for cur_letter in range(26):
            #initialize with feasible solution of previous letter = a and this letter = a
            best = M[row - 1][0] + np.inner(X[row], w[0])
            
            #iterate over all values of the previous letter, fixing the current letter
            for prev_letter in range(26):
                temp_product = M[row-1][prev_letter] + np.inner(X[row], w[cur_letter]) + t[prev_letter][cur_letter]
                if(temp_product > best):
                    best = temp_product
            M[row][cur_letter] = best
    return M



def trial_solution_value(trial_solution, X, w, t):
    running_total = np.inner(X[0], w[trial_solution[0]])
    for i in range(1, len(X)):
        running_total += np.inner(X[i], w[trial_solution[i]]) + t[trial_solution[i-1]][trial_solution[i]]
    return running_total

def optimize_brute_force(X, w, t):
    stopping_criteria = []
    trial_solution = []
    #initialize stopping criteria and trial solutions
    for i in range(len(X)):
        stopping_criteria.append(25)
        trial_solution.append(0)
    
        #increment trial solution
    best = trial_solution_value(trial_solution, X, w, t)
    best_solution = trial_solution
    while(True):
        i = 0
        while(True):
            trial_solution[i] += 1
            if(trial_solution[i]== 26):
                trial_solution[i] = 0
                i += 1
            else:
                break
            
        trial_value = trial_solution_value(trial_solution, X, w, t)
        if(trial_value > best):
            best = trial_value
            best_solution = deepcopy(trial_solution)
        
        if(trial_solution == stopping_criteria):
            break
    return best, best_solution


def get_solution_from_M(M):
    solution = []
    cur_word_pos = len(M) - 1
    prev_word_pos = cur_word_pos - 1
    cur_letter = np.argmax(M[cur_word_pos])
    cur_val = M[cur_word_pos][cur_letter]
    solution.insert(0, cur_letter)
    
    while(cur_word_pos > 0):
        for prev_letter in range(26):
            if(abs(cur_val - M[prev_word_pos][prev_letter] - t[prev_letter][cur_letter] - np.inner(X[cur_word_pos], w[cur_letter]) ) < 0.00001):
                solution.insert(0, prev_letter)
                cur_letter = prev_letter
                cur_word_pos -=1
                prev_word_pos -=1
                cur_val = M[cur_word_pos][cur_letter]
                break
        
    return solution
        
        
        

#y, X = get_data.read_data_formatted()
X, w, t = get_weights.get_weights_formatted()
best_val, best_soln =  optimize_brute_force(X[:6], w, t)
print(best_soln)
print(best_val)

M = optimize(X[:6], w, t)


print(get_solution_from_M(M))
print(np.max(M))