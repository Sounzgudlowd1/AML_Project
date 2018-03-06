# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:18:29 2018

@author: Erik
"""
import part2b_code as p2b
import gradient_calculation as gc
import get_data as gd
import part2b_code as p2b

X_train, y_train = gd.read_data_formatted('train')
X_test, y_test = gd.read_data_formatted('test')
params = gd.get_params()




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
    print(p2b.accuracy(prediction, y_test))