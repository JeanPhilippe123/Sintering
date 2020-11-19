# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:47:53 2020

@author: jplan58
"""
import dask.array as da
import numpy as np

data = da.from_array(np.array([[]]*3),chunks=[1,1])

for i in range(0,5):
    da2 = da.arange((i),chunks=(1))
    da3 = da.arange((i),chunks=(1))+4
    da4 = da.arange((i),chunks=(1))+10
    data = da.concatenate([data,[da2,da3,da4]],axis=1)#.compute()
