# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:19:36 2019

@author: xxzac
"""
#For assignment 3
import pandas as pd
import numpy as np

#Create a class in python that is a homebrewed K Nearest-neighbors classifier.
#Using the scipy distance import, more distance metrics can be added as needed.
#I don't think that the manhattan edge distance is a part of the scipy distance
#metrics.

class KNN2:
    def __init__(self, k, distance_f):
        #TODO: Create instance variables that store these values
        self.k = k
        self.distance_f = distance_f
        
    def fit(self, z, y): 
        #MAKE SURE YOU CHECK IF INPUTS ARE NP ARRAYS AND NOT DATAFRAMES AND CONVERT THEM IF NEEDED
        if isinstance(z, pd.DataFrame):
            z = z.values
        if isinstance(y, pd.DataFrame):
            y = y.values
            
        self.z = z
        self.y = y
    def predict(self, x):
        #Implement prediction logic and return list/array of predicted labels
        preds = []
        #OK HERE WE GO...!
        #For every row in the training set, create an empty list to store distance
        #measures and "pre-predictions" (pre-predictions are the k predictions before voting)
        for i in x:
            
            distances = [] 
            pre_preds = []
            #For each item in each row, append the distance measurement to distances
            for e in self.z:
                distances.append(self.distance_f(i,e))
            #Convert distances into an array, then grab the indexes for the k smallest
            #distance values
            distances = np.array(distances)
            inds = distances.argsort()[:self.k]
            #For each index, append its cooresponding value from y_train into pre-preds
            #Pre-preds should have a length of k.
            for t in inds:
                pre_preds.append(self.y.values[t])
            #Finally, append the mode of pre preds to preds, then do it all over again.
            preds.append(max(set(pre_preds), key=pre_preds.count))
        return preds