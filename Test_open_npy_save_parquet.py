# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:11:44 2020

@author: jplan58
"""
import os
import dask.dataframe as dd
import numpy as np
import dask.array as da

from dask.distributed import Client
from dask.diagnostics import visualize
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler

#Open parquet and save it to npy file (back and forth)
#Open a npy and retrieve data
# directory = 'Z:\\Sintering\\Monte_Carlo\\Simulations\\test1\\'
directory = os.path.join(os.path.dirname(__file__),'Monte_Carlo','Simulations','test1')
path_npy = os.path.join(directory,'test1_4631.0774_1.2521_0.8054_0.89_[1, 1, 0, 90]_10000_not_diffuse.npy')
path_parquet = os.path.join(directory,'test1_4631.0774_1.2521_0.8054_0.89_(1,1,0,90)_10000_not_diffuse.parquet')
headers = np.array(["segmentLevel", "hitObj", "insideOf", "numray",
                    "x", "y", "z", "n", "exr", "exi", "eyr", "eyi", "ezr", "ezi", "intensity", "pathLength"])
print(path_parquet)
if __name__=='__main__':
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof,CacheProfiler() as cprof:
        #Load_npy file
        dataframe = np.load(path_npy)
        #Create dask array
        df = da.from_array(dataframe,chunks=(16,100_000)).transpose()
        
        #Create dask dataframe
        df = df.to_dask_dataframe(columns=headers)
        # df = df.set_index('numray',sorted=True)
        
        #Save it to parquet
        dd.to_parquet(df,path_parquet)
        
        df = dd.read_parquet(path_parquet)
    
    visualize([prof, rprof, cprof])