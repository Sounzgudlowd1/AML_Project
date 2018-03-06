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
            gc.log_p_y_given_x_avg, 
            gc.gradient_avg, 
            params, X, y, 1))



def timed_gradient_calculation(params, X, y):
    #this takes 621 seconds for me (10 min 21 seconds)
    start = time()
    av_grad = gc.gradient_avg(params, X, y, len(X))
    print("Total time:")
    print(time() - start)
    
    
    with open("../result/gradient.txt", "w") as text_file:
        for i, elt in enumerate(av_grad):
            text_file.write(str(elt))
            text_file.write("\n")
   
#print(gc.gradient_avg(params, X, y, 1)[0: 50])


X, y = gd.read_data_formatted('train')
params = gd.get_params()

#print(gc.log_p_y_given_x_avg(params, X, y, len(X)) )

check_gradient(params, X, y)

timed_gradient_calculation(params, X, y)
