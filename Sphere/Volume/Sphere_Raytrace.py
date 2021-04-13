"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""
import numpy as np
import os
import time
import dask.array as da
import dask.dataframe as dd
import sys
import pandas as pd
from dask.diagnostics import visualize
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler

path_RaytraceDLL = 'C:\Zemax Files\ZOS-API\Libraries'
sys.path.insert(2,path_RaytraceDLL)
import PythonNET_ZRDLoader as init_Zemax

from dask.distributed import Client

# from dask.dataframe.multi import concat
def profiler(func):
    def wrapper(*args,**kwargs):
        with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof,CacheProfiler() as cprof:
            func(*args,**kwargs)
        visualize([prof, rprof, cprof])
        return
    return wrapper

def retrieve_data(self,ZRDReader,path_parquet):
    # client = Client()
    results = ZRDReader.GetResults();
    dataReader = self.BatchRayTrace.ReadZRDData(results)
    maxSegments = 9_000_000;
    ZRDData = dataReader.InitializeOutput(maxSegments);

    isFinished = False;
    totalSegRead = 0;
    totalRaysRead = 0;

    start_read = time.time()
    i=0
    
    headers = np.array(["numray","segmentLevel", "hitObj", "insideOf",
    "x", "y", "z", "L", "N", "M", "exr", "exi", "eyr", "eyi", "ezr", "ezi", "intensity", "pathLength", "Indice"])
    while isFinished == False and ZRDData is not None:
        readSegments = dataReader.ReadNextBlock(ZRDData);
        if readSegments == 0:
            isFinished = True;
        else:
            # totalSegRead = write_partitions(self,ZRDData,totalSegRead,readSegments,i,path_parquet)
            totalSegRead = totalSegRead + readSegments
            totalRaysRead = (self.zosapi.LongToNumpy(ZRDData.RayNumber).max()).compute()
            RayNumber = self.zosapi.LongToNumpy(ZRDData.RayNumber)[:readSegments]
            Level = self.zosapi.LongToNumpy(ZRDData.Level)[:readSegments]
            HitObject = self.zosapi.LongToNumpy(ZRDData.HitObject)[:readSegments]
            InsideOf = self.zosapi.LongToNumpy(ZRDData.InsideOf)[:readSegments]
            X =  self.zosapi.DoubleToNumpy(ZRDData.X)[:readSegments]
            Y = self.zosapi.DoubleToNumpy(ZRDData.Y)[:readSegments]
            Z =  self.zosapi.DoubleToNumpy(ZRDData.Z)[:readSegments]
            L =  self.zosapi.DoubleToNumpy(ZRDData.L)[:readSegments]
            N =  self.zosapi.DoubleToNumpy(ZRDData.N)[:readSegments]
            M =  self.zosapi.DoubleToNumpy(ZRDData.M)[:readSegments]
            Exr = self.zosapi.DoubleToNumpy(ZRDData.Exr)[:readSegments]
            Exi = self.zosapi.DoubleToNumpy(ZRDData.Exi)[:readSegments]
            Eyr = self.zosapi.DoubleToNumpy(ZRDData.Eyr)[:readSegments]
            Eyi = self.zosapi.DoubleToNumpy(ZRDData.Eyi)[:readSegments]
            Ezr = self.zosapi.DoubleToNumpy(ZRDData.Ezr)[:readSegments]
            Ezi = self.zosapi.DoubleToNumpy(ZRDData.Ezi)[:readSegments]
            Intensity = self.zosapi.DoubleToNumpy(ZRDData.Intensity)[:readSegments]
            PathLen = self.zosapi.DoubleToNumpy(ZRDData.PathLen)[:readSegments]
            Indice = self.zosapi.DoubleToNumpy(ZRDData.index)[:readSegments]
            data = da.stack([RayNumber,Level,HitObject,InsideOf,X,Y,Z,L,N,M,Exr,Exi,Eyr,Eyi,Ezr,Ezi,Intensity,PathLen,Indice])
            df_i = dd.from_dask_array(data.transpose(),columns=headers).set_index('numray',sorted=True).repartition(npartitions=12)
            
            print(i)
            if i==0:
                df_i.to_parquet(path_parquet)
            else:
                df_i.to_parquet(path_parquet,append=True)
                
            i+=1
        if totalRaysRead >= maxSegments:
            isFinished = True
    print('----------------------------------------------------')
    print('Rays read:                                 %i' % totalRaysRead)
    print('Total loop:                                 %i' % i)
    print('Segments read:                             %i' % totalSegRead)
    print('----------------------------------------------------')
    end_read = time.time()
    print('Time took for reading: ',round(end_read-start_read,2))
    
    return #df

# @profile
def Shoot(self,Filter,numrays,path_parquet,nameZRD):
    # Open file
    pathZMX = os.path.dirname(self.fileZMX)
    
    #Find indice of 'Source Object' and add the correct numrays for stereo
    Source_obj = self.find_source_object()[0]
    Source = self.TheNCE.GetObjectAt(Source_obj)
    Source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par2).IntegerValue = numrays
    
    start_trace = time.time()
    # Trace and save a ZRD file for test later
    NSCRayTrace = self.TheSystem.Tools.OpenNSCRayTrace()
    NSCRayTrace.ScatterNSCRays = True
    NSCRayTrace.SplitNSCRays = True
    NSCRayTrace.UsePolarization = True
    NSCRayTrace.IgnoreErrors = True 
    NSCRayTrace.SaveRays = True
    fileZRD = nameZRD
    NSCRayTrace.SaveRaysFile = fileZRD
    NSCRayTrace.Filter = Filter
    NSCRayTrace.ClearDetectors(0)
    NSCRayTrace.RunAndWaitForCompletion()
    NSCRayTrace.Close()
    end_trace = time.time()
    print('Time took for tracing: ',round(end_trace-start_trace,2))
    
    ZRDReader = self.TheSystem.Tools.OpenRayDatabaseReader()
    ZRDReader.ZRDFile = pathZMX + os.sep + fileZRD
    
    ZRDReader.RunAndWaitForCompletion()
    
    #Retrieve Datas from simulation into a dataframe
    retrieve_data(self,ZRDReader,path_parquet)
    # Remove_MSP_errors(self)
    
    try:
        ZRDReader.Close()
    except :
        pass

    start_write = time.time()
    
    end_write = time.time()
    print("Time took for writing paquet file: ",round(end_write-start_write,2))
    
    return path_parquet

def Remove_MSP_errors(self):
    print('Removing MSP errors')
    start_MSP = time.time()
    
    df = Load_parquet(self.path_parquet,print_statement=False)
    index_error = df.query('(intensity.diff()>0.)&(segmentLevel != 0)').index.drop_duplicates()
    if list(index_error) == 0:
        print('No MSP errors')
    else:
        df = df.loc[list(df.index.drop_duplicates().compute().difference(index_error))]
        df.to_parquet(self.path_parquet)
    
    end_MSP = time.time()
    print('Time took for removing MSP errors:',round(end_MSP-start_MSP,2))
    return df
    
def Load_parquet(path_parquet,name='',print_statement=True):
    #Load_hdf and transform it into a dataframe with low memory usage
    start_load = time.time()
    
    df = dd.read_parquet(path_parquet)
    # df = df.persist()
    
    end_load = time.time()
    
    if print_statement == True:
        print("Time took for loading {} parquet file : ".format(name), round(end_load-start_load,2))
    return df