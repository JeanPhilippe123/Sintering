# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:46:45 2020

@author: jplan58
"""
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd

x = np.array([[1,0.5,2],[1,2,3],[2,3,4]],dtype='float32')
df = dd.from_array(x,columns=['A','B','C'])
df = df.set_index(['A'])
meta = dd.utils.make_meta({'B': 'O'}, index=df.index)
cs = df.groupby(df.index).apply(np.cumsum,meta=meta)

