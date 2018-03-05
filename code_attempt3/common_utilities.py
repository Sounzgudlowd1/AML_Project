# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:04:22 2018

@author: Erik
"""

letter_dict = {1: 'a', 2:'b', 3:'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i',
               10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15:'o', 
               16:'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 
               23: 'w', 24: 'x', 25: 'y', 26: 'z'}

def print_x(x):
    count = 0
    for elt in x:
        count += 1
        if(count == 8):
            print(int(elt))
            count = 0
        else:
            print(int(elt), end = '')

    print()
    
def print_word(word):
    for letter in word:
        print(letter_dict[letter], end = '')
    print()