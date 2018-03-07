# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 19:30:06 2018

@author: Erik
"""

import get_data as gd
import part2b_code as p2b
import part1c_code as p1c
import gradient_calculation as gc

X_test, y_test = gd.read_data_formatted('test_struct.txt')
X_train, y_train = gd.read_data_formatted('train_struct_1500.txt')
params = gd.get_params()


p2b.optimize(params, X_train, y_train, 1000, 'solution_1500_distortion')



params = p2b.get_optimal_params('solution_1500_distortion')
w = gc.w_matrix(params)
t = gc.t_matrix(params)

print("Function value: ")
print(p2b.func_to_minimize(params, X_train, y_train, 1000))

 
y_pred = p2b.predict(X_test, w, t)

print(p2b.accuracy(y_pred, y_test))
