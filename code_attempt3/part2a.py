# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:07:45 2018

@author: Erik
"""

from scipy.optimize import check_grad
from time import time
import gradient_calculation as gc
import get_data as gd


def check_gradient(params, X, y):
    #kind of like a unit test.  Just check the gradient of the first 10 words
    #this takes a while so be forwarned
    print(check_grad(
            gc.avg_log_p_y_given_x, 
            gc.avg_gradient, 
            params, X, y, 20))
    
def avg_log_prob(params, X, y):
    return gc.log_p_y_given_x_sum(params, X, y, len(X)) / len(X)    


def timed_gradient_calculation(params, X, y):
    #this takes 621 seconds for me (10 min 21 seconds)
    start = time()
    av_grad = gc.gradient_sum(params, X, y, len(X))/len(X)
    print("Total time:")
    print(time() - start)
    
    
    with open("../result/gradient.txt", "w") as text_file:
        for i, elt in enumerate(av_grad):
            text_file.write(str(elt))
            text_file.write("\n")
    


X, y = gd.read_data_formatted()
params = gd.get_params()

#print(gc.avg_log_p_y_given_x(params, X, y, len(X)))

#check_gradient(params, X, y)

timed_gradient_calculation(params, X, y)
print(avg_log_prob(params, X, y))