# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 11:28:17 2020

@author: jplan58
"""
import time
import dask.array  as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

path_parquet = 'test_memory_error_repartition.parquet'
numrays = 50000
headers = np.array(["segmentLevel", "hitObj", "insideOf", "numray",
    "x", "y", "z", "n", "exr", "exi", "eyr", "eyi", "ezr", "ezi", "intensity", "pathLength"])
seg_ray = 600

Level = da.random.random((10_000_000),chunks="10 MiB")
HitObject = da.random.random((10_000_000),chunks="10 MiB")
# RayNumber = da.arange(100_000_000,chunks="10 MiB")
RayNumber = []
Inside = []
for i in range(0,10_000):
    RayNumber += list(np.ones(1_000)*i)
for i in range(0,10_000):
    Inside += list(np.arange(0,1_000))
RayNumber = da.from_array(RayNumber,chunks="10 MiB")
InsideOf = da.from_array(Inside,chunks="10 MiB")
X = da.random.random((10_000_000),chunks="10 MiB")
Y = da.random.random((10_000_000),chunks="10 MiB")
Z = da.random.random((10_000_000),chunks="10 MiB")
L = da.random.random((10_000_000),chunks="10 MiB")
M = da.random.random((10_000_000),chunks="10 MiB")
N = da.random.random((10_000_000),chunks="10 MiB")
Exr = da.random.random((10_000_000),chunks="10 MiB")
Exi = da.random.random((10_000_000),chunks="10 MiB")
Eyr = da.random.random((10_000_000),chunks="10 MiB")
Eyi = da.random.random((10_000_000),chunks="10 MiB")
Ezr = da.random.random((10_000_000),chunks="10 MiB")
Ezi = da.random.random((10_000_000),chunks="10 MiB")
Intensity = da.random.random((10_000_000),chunks="10 MiB")
PathLen = da.random.random((10_000_000),chunks="10 MiB")

start_df = time.time()
#Create Dataframe from the retrieve data
data = [Level,HitObject,InsideOf,RayNumber,X,Y,Z,L,Exr,Exi,Eyr,Eyi,Ezr,Ezi,Intensity,PathLen]
data = da.stack(data).transpose()
df = data.to_dask_dataframe(columns=headers)
df = df.repartition(partition_size='100 MiB')
end_df = time.time()
print("Time took for creating df: ",round(end_df-start_df,2))

# start_persist = time.time()
df = df.persist()
# end_persist = time.time()
# print("Time took for persisting: ",round(end_persist-start_persist,2))

# start_set_index = time.time()
df = df.set_index('numray')
df = df.persist()
# end_set_index = time.time()
# print("Time took for setting index paquet file: ",round(end_set_index-start_set_index,2))

# start_persist = time.time()
# df = df.persist()
# end_persist = time.time()
# print("Time took for persisting: ",round(end_persist-start_persist,2))

# start_write = time.time()
# #Write to parquet
# # df.to_parquet(path_parquet)
# end_write = time.time()
# print("Time took for writing paquet file: ",round(end_write-start_write,2))
# print("Time took: ",round(end_write-start_df,2))

# print(df.map_partitions(len).compute())
# print(len(df))
# for i in range(0,10):
#     print(len(df.loc[i]))
