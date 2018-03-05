# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:18:29 2018

@author: Erik
"""
import part2b_code as p2b
import get_data as gd

X_train, y_train = gd.read_data_formatted('train')
params = gd.get_params()

cvals = [1, 10, 100]

for elt in cvals:
    p2b.optimize(params, X_train, y_train, 'solution' + str(elt))