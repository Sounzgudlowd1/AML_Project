# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 12:13:56 2018

@author: Erik
"""
import part1c_code as cd
import numpy as np

X, w, t = cd.get_weights_formatted()

M = cd.optimize(X, w, t)
print(np.max(M))
soln = cd.get_solution_from_M(M, X, w, t)

with open("../result/decode_output.txt", "w") as text_file:
    for i, elt in enumerate(soln):
        text_file.write(str(elt + 1))
        text_file.write("\n")