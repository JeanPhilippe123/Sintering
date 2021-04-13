# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:06:43 2021

@author: jplan58
"""

import csv
import os 
import numpy as np
import scipy.interpolate

class Material:
    def __init__(self,Material,folder='Z:\Sintering\Glass_Catalog'):
        self.datafile = open(os.path.join(folder,str(Material)+'.txt'), 'r')
        
    def parse_file(self):
        myreader = csv.reader(self.datafile, delimiter='\t')
        data = []
        for row in myreader:
            try:
                try:
                    row = [[float(row[0]),float(row[1])]]
                    data += row
                except IndexError:
                    #Sketch mais fonctionne
                    index_real=np.array(data)
                    data = []
                    pass
            except ValueError:
                pass
        index_imag=np.array(data)
        return index_real,index_imag
    
    def get_refractive_index(self,wl):
        index_real,index_imag = self.parse_file()
        #1D interpolatation result to get the real part the imaginary part
        inter_real = scipy.interpolate.interp1d(index_real[:,0], index_real[:,1])
        inter_imag = scipy.interpolate.interp1d(index_imag[:,0], index_imag[:,1])
        return inter_real(wl).item(),inter_imag(wl).item()
    
if __name__=='__main__':
    mat = Material('B270')
    data = mat.get_refractive_index(0.55)