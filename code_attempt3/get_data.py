# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:19:33 2018

@author: Erik
"""
import numpy as np
import re


def read_data(file_name):
    file = open('../data/' + file_name , 'r') 
    y = []
    qids = []
    X = []
    
    for line in file:
        temp = line.split()
        #get label
        label_string = temp[0]
        #normalizes y to 0...25 instead of 1...26
        y.append(int(label_string) -1)
        
        #get qid
        qid_string = re.split(':', temp[1])[1]
        qids.append(int(qid_string))
        
        #get x values
        x = np.zeros(128)
        
        #do we need a 1 vector?
        #x[128] = 1
        for elt in temp[2:]:
            index = re.split(':', elt)
            x[int(index[0] )-1] = 1
        X.append(x)
    y = np.array(y)
    qids = np.array(qids)
    return y, qids, X

def read_data_formatted(file_name):
    #get initial output
    y, qids, X = read_data(file_name)
    y_tot = []
    X_tot = []
    current = 0;
    
    y_tot.append([])
    X_tot.append([])
    
    for i in range(len(y)):
        y_tot[current].append(y[i])
        X_tot[current].append(X[i])
        
        if(i+ 1 < len(y) and qids[i] != qids[i + 1]):
            y_tot[current] = np.array(y_tot[current])
            X_tot[current] = np.array(X_tot[current])
            y_tot.append([])
            X_tot.append([])
            current = current + 1
            
    return X_tot, y_tot
            


#parameter set for 2a
def get_params():
    
    file = open('../data/model.txt', 'r') 
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)



  