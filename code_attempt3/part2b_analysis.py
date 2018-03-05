# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:35:11 2018

@author: Erik
"""
import numpy as np
import get_data as gd
import gradient_calculation as gc
import part2b_code as p2b
import part1c_code as p1c
import time
    
#parameter set for 2a

def predict(X, w, t):
    y_pred = []
    for i, x in enumerate(X):
        M = p1c.optimize(x, w, t)
        y_pred.append(p1c.get_solution_from_M(M, x, w, t))
    return y_pred

X_test, y_test = gd.read_data_formatted('test')
X_train, y_train = gd.read_data_formatted('train')
params = gd.get_params()


p2b.optimize(params, X_train, y_train, 1000, 'solution')



params = p2b.get_optimal_params()
w = p1c.parse_w(params)
t = p1c.parse_t(params)





print(p2b.func_to_minimize(params, X_train, y_train, 1000))

y_pred = predict(X_test, w, t)


with open("../result/prediction.txt", "w") as text_file:
    for i, elt in enumerate(y_pred):
        text_file.write(str(elt))
        text_file.write("\n")  
        
print(accuracy(y_pred, y_test))