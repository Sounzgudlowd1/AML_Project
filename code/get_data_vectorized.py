# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:25:03 2018

@author: Erik
"""
import numpy as np

letter_lookup = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g':6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u':20, 'v':21, 'w': 22, 'x': 23, 'y':24, 'z':25}



def read_data_with_dummies():
    file = open('../data/train.txt', 'r')
    file.seek(0)
    dummies = []
    for i, line in enumerate(file):
        print(i)
        temp = line.split()
        x = np.array(temp[5:128*26 + 5])
        dummy = np.zeros((26, 128 * 26 + 26 * 26))
        for i in range(26):
            min_pos = i * 128
            max_pos = (i +  1) * 128
            dummy[i][min_pos: max_pos] = x[0: 128]
            t_pos_min = 128 * 26 + 26 * i
            t_pos_max = 128 * 26 + 26 * (i + 1)
            dummy[i][t_pos_min: t_pos_max] =  1
            dummies.append(dummy)
            
    return dummies
        

def read_data():
    file = open('../data/train.txt', 'r')
    file_len = sum(1 for line in file)
    print(file_len)
    file.seek(0)
    full_data_set = np.zeros((file_len, 128 * 26 + 26*26))
    i = 0
    prev_letter = 0
    for line in file:
        temp = line.split()
            
        letter_id = int(temp[0]) -1 #starts at 1 so subtract for an array
        
        letter = letter_lookup[temp[1]]
        y_start = letter * 128
        y_end = letter * 128 + 128
        is_first_letter = int(temp[4]) == 1
        #128 pixel representation
        temp_X = np.array(temp[5:128*26 + 5])
        
        
        full_data_set[letter_id][y_start:y_end] = temp_X
        
        if(not is_first_letter):
            t_index = prev_letter * 26 +  26 * 128
            #now add this letter to it
            t_index += letter
            full_data_set[letter_id][t_index] = 1
            
        prev_letter = letter    
        i += 1

    return full_data_set

data = read_data() 