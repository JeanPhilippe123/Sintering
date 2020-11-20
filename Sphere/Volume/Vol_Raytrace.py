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
import pandas as pd
# from dask.dataframe.multi import concat

def retrieve_data(self,ZRDReader):
    results = ZRDReader.GetResults();
    dataReader = self.BatchRayTrace.ReadZRDData(results)
    maxSegments = 1_000_000;
    ZRDData = dataReader.InitializeOutput(maxSegments);

    isFinished = False;
    totalSegRead = 0;
    totalRaysRead = 0;

    start_read = time.time()
    i=0
    
    # data = da.from_array(np.array([[]]*17),chunks=[1,1]).transpose()
    headers = np.array(["numray","segmentLevel", "hitObj", "insideOf",
    "x", "y", "z", "L", "exr", "exi", "eyr", "eyi", "ezr", "ezi", "intensity", "pathLength", "Indice"])
    data = np.array([[]]*17)
    df = pd.DataFrame(columns=headers)
    datas2=[]
    while isFinished == False and ZRDData is not None:
        readSegments = dataReader.ReadNextBlock(ZRDData);
        if readSegments == 0:
            isFinished = True;
        else:
            #Get the right data from the file
            totalSegRead = totalSegRead + readSegments
            totalRaysRead = np.max(self.zosapi.LongToNumpy(ZRDData.RayNumber))
            RayNumber = self.zosapi.LongToNumpy(ZRDData.RayNumber)[:readSegments]
            Level = self.zosapi.LongToNumpy(ZRDData.Level)[:readSegments]
            HitObject = self.zosapi.LongToNumpy(ZRDData.HitObject)[:readSegments]
            InsideOf = self.zosapi.LongToNumpy(ZRDData.InsideOf)[:readSegments]
            X =  self.zosapi.DoubleToNumpy(ZRDData.X)[:readSegments]
            Y = self.zosapi.DoubleToNumpy(ZRDData.Y)[:readSegments]
            Z =  self.zosapi.DoubleToNumpy(ZRDData.Z)[:readSegments]
            L =  self.zosapi.DoubleToNumpy(ZRDData.L)[:readSegments]
            Exr = self.zosapi.DoubleToNumpy(ZRDData.Exr)[:readSegments]
            Exi = self.zosapi.DoubleToNumpy(ZRDData.Exi)[:readSegments]
            Eyr = self.zosapi.DoubleToNumpy(ZRDData.Eyr)[:readSegments]
            Eyi = self.zosapi.DoubleToNumpy(ZRDData.Eyi)[:readSegments]
            Ezr = self.zosapi.DoubleToNumpy(ZRDData.Ezr)[:readSegments]
            Ezi = self.zosapi.DoubleToNumpy(ZRDData.Ezi)[:readSegments]
            Intensity = self.zosapi.DoubleToNumpy(ZRDData.Intensity)[:readSegments]
            PathLen = self.zosapi.DoubleToNumpy(ZRDData.PathLen)[:readSegments]
            Indice = self.zosapi.DoubleToNumpy(ZRDData.index)[:readSegments]
            
            data = da.from_array(np.array([RayNumber,Level,HitObject,InsideOf,X,Y,Z,L,Exr,Exi,Eyr,Eyi,Ezr,Ezi,Intensity,PathLen,Indice]),chunks=[1,readSegments])
            datas2 +=[data]
            i+=1
        if totalRaysRead >= maxSegments:
            isFinished = True
    
    # datas = da.concatenate(datas2,axis=1).rechunk((17,maxSegments)).persist()
    # print(len(datas[0]))
    #BUg avec dask set_index... to counter the bug
    dfs=[]
    for i in range(0,len(datas2)):
        df_i = dd.from_dask_array(datas2[i].transpose(),columns=headers)
        dfs += [df_i.set_index('numray',sorted=True).persist()]
    df=dd.concat(dfs)
    print(len(df))
    # print(len(df),df.npartitions)
    # df2 = df.set_index('numray',sorted=True).compute()
    # print(len(df2))
    # df[headers[jj]] = datas[jj]
    df=df.persist()
    self.df = df
    # self.datas = datas
    
    #Create DataFrame

    print('----------------------------------------------------')
    print('Rays read:                                 %i' % totalRaysRead)
    print('Total loop:                                 %i' % i)
    print('Segments read:                             %i' % totalSegRead)
    print('----------------------------------------------------')
    end_read = time.time()
    print('Time took for reading: ',round(end_read-start_read,2))
    
    #Create Dataframe from the retrieve data
    return df

# @profile
def Shoot(self,Filter,numrays,path_parquet,nameZRD,path_metadata=None):
    
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
    
    if path_metadata!=None:
        #Write metadata
        np.save(path_metadata,self.array_objects())
    
    #Retrieve Datas from selfulation into a dataframe
    df = retrieve_data(self,ZRDReader)
    
    start_write = time.time()
    
    df.repartition(npartitions=10)
    # #Add the good index
    df = df.persist()
    
    #Write to parquet
    df.to_parquet(path_parquet)
    
    end_write = time.time()
    print("Time took for writing paquet file: ",round(end_write-start_write,2))
    
    return path_parquet

def Load_parquet(path_parquet,name=''):
    #Load_hdf and transform it into a dataframe with low memory usage
    start_load = time.time()
    
    df = dd.read_parquet(path_parquet)
    df = df.persist()
    
    end_load = time.time()
    
    print("Time took for loading {} paquet file : ".format(name), round(end_load-start_load,2))
    return df