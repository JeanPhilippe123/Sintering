# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:10:02 2020

@author: jplan58
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np

df = pd.DataFrame(data=[0,1,2,3,4,5,4,3,2,1,0,1,0,1,2,3,2,1,2,3],columns=['A'])
df = dd.from_pandas(df,npartitions=2)
df = df.assign(segmentLevel = dd.from_array(np.arange(0,21)))
df = df.assign(segmentLevel2 = dd.from_array(np.arange(0,21)))
depth=1.5
df1 = df.query('A<= {} & A.shift() >= {}'.format(depth,depth)).compute()
df2 = df.query('(A>= {} & A.shift() <= {})'.format(depth,depth)).compute() #downwelling irradiance
df3 = df.query('(A>= {} & A.shift() <= {})|(A<= {} & A.shift() >= {})'.format(depth,depth,depth,depth)).compute() #downwelling irradiance
