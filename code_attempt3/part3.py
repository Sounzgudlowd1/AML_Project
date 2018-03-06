# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:18:29 2018

@author: Erik
"""
import part2b_code as p2b
import gradient_calculation as gc
import get_data as gd
import numpy as np

X_train, y_train = gd.read_data_formatted('train')
X_test, y_test = gd.read_data_formatted('test')
params = gd.get_params()



def accuracy(y_pred, y_act):
    word_count = 0
    correct_word_count = 0
    letter_count = 0
    correct_letter_count = 0
    for i in range(len(y_pred)):
        word_count += 1
        correct_word_count += np.sum(y_pred[i] == y_act[i]) == len(y_pred[i])
        letter_count += len(y_pred[i])
        correct_letter_count += np.sum(y_pred[i] == y_act[i])
    return correct_word_count/word_count, correct_letter_count/letter_count
#Run optimization
'''
cvals = [1, 10, 100, 1000]

for elt in cvals:
    p2b.optimize(params, X_train, y_train, elt, 'solution' + str(elt))
    print("done with" + str(elt))
    
'''

#check accuracy
cvals = [1, 10, 100, 1000]

for elt in cvals:
    params = p2b.get_optimal_params('solution' + str(elt))
    w = gc.w_matrix(params)
    t = gc.t_matrix(params)
    prediction = p2b.predict(X_test, w, t)
    print("Accuracy for: " + str(elt))
    print(accuracy(prediction, y_test))