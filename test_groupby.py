# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:46:45 2020

@author: jplan58
"""
import numpy as np
import dask.array as da
import dask.dataframe as dd

x = np.array([[1,0.5],[1,2],[2,3]],dtype='float32')
df = dd.from_array(x,columns=['A','B'])
df = df.set_index(['A'])
cs = df.groupby(df.index).aggregate()