# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:25:03 2018

@author: Erik
"""
import numpy as np

letter_lookup = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g':6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u':20, 'v':21, 'w': 22, 'x': 23, 'y':24, 'z':25}

rev_letter_lookup = {0: 'a', 1:'b', 2:'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i',
               9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14:'o', 
               15:'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 
               22: 'w', 23: 'x', 24: 'y', 25: 'z'}
        
def get_letter(in_num):
    return rev_letter_lookup[in_num]


def get_xy():
    file = open('../data/train.txt', 'r')
    X = []
    y = []
    word_x = []
    word_y = []
    for i, line in enumerate(file):
        #split line into individual elements
        
        temp = line.split()        
        is_first_letter = int(temp[4]) == 1
        if(is_first_letter and i > 0):
            X.append(np.array(word_x))
            y.append(np.array(word_y))
            word_x = []
            word_y = []
  
        #the
        word_y.append(letter_lookup[temp[1] ])
        
        #128 pixel representation
        temp_x = np.array(temp[5:128*26 + 5])
        letter = np.zeros(26 * 128 + 26 * 26)
        for i in range(26):
            letter[i * 128: (1+i) * 128] = temp_x
        letter[26*128:] = 1
        word_x.append(letter)

    return X, y

def get_mask():
    mask = np.zeros((26, 26*128 + 26 * 26))
    for i in range(26):
        mask[i][128 * i : 128 * (i + 1)] = 1
    return mask

def get_mask_with_t():
    mask = np.zeros((26, 26, 26*128 + 26 * 26))
    for i in range(26):
        for j in range(26):
            mask[j][i][128 * 26 + 26 * i + j] = 1
    return mask

