# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:07:08 2020

@author: jplan58
"""

import numpy as np
import psutil
import h5py
import pandas as pd
import time
from sys import getsizeof
import feather
import dask.array as da
import numpy as np
import h5py

import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import visualize
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler

if __name__=='__main__':
    
    
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof,CacheProfiler() as cprof:
        df = da.random.random((100_000_000, 3), chunks=(1_000_000,3)).to_dask_dataframe(columns=['A','B','C'])
        
        df.to_parquet('test_hdf.paquet')
        data_2 = dd.read_parquet('test_hdf.paquet',columns='A')
        
    visualize([prof, rprof, cprof])
    # df.to_hdf('test_hdf.hdf', '/df')
    # e = time.time()
    # print(psutil.virtual_memory())
    # print(e-s)
    
    # data_1 = pd.read_parquet('test_hdf.hdf','/df')
    # print(psutil.virtual_memory())
    # print(e-s)
    # del data_1, data_2
    # print(psutil.virtual_memory())