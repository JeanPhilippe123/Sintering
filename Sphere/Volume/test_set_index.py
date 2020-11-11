# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:46:42 2020

@author: jplan58

test setting index
"""
import pandas as pd, numpy as np 
import dask.array as da, dask.dataframe as dd

c1 = da.from_array(np.arange(100000, 190000), chunks=1000)
c2 = da.from_array(np.arange(200000, 290000), chunks=1000)
c3 = da.from_array(np.arange(300000, 390000), chunks=1000)

# generate dask dataframe
ddf = dd.concat([dd.from_dask_array(c) for c in [c1,c2,c3]], axis = 1) 
# name columns
ddf.columns = ['c1', 'c2', 'c3']

#%%
unique_c2_list = [0,1,2,3]
a=sorted(unique_c2_list + [unique_c2_list[-1]])