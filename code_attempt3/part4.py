# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:23:15 2018

@author: Erik
"""

import numpy as np

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