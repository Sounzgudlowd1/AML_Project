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



X_test, y_test = gd.read_data_formatted('test_struct.txt')
X_train, y_train = gd.read_data_formatted('train_struct.txt')
params = gd.get_params()

#run optimization.  For C= 1000 it takes about an hour so just read params and predict
p2b.optimize(params, X_train, y_train, 1000, 'solution')



params = p2b.get_optimal_params('solution')
w = gc.w_matrix(params)
t = gc.t_matrix(params)




print("Function value: ")
print(p2b.func_to_minimize(params, X_train, y_train, 1000))


y_pred = p2b.predict(X_test, w, t)

with open("../result/prediction.txt", "w") as text_file:
    for i, elt in enumerate(y_pred):
        text_file.write(str(elt))
        text_file.write("\n")  

print(p2b.accuracy(y_pred[:1], y_test[:1]))
