"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""
import numpy as np
import pandas as pd
import os
import time
import psutil

# @profile
def Shoot(Sim,Filter,numrays,pathnpy,nameZRD):
    
    if Sim.diffuse_light == True:
        diffuse_str = 'diffuse'
    else:
        diffuse_str = 'not_diffuse'
    # # Open file
    pathZMX = os.path.dirname(Sim.fileZMX)
    
    #Find indice of 'Source Object' and add the correct numrays for stereo
    Source_obj = Sim.find_source_object()[0]
    Source = Sim.TheNCE.GetObjectAt(Source_obj)
    Source.GetObjectCell(Sim.ZOSAPI_NCE.ObjectColumn.Par2).IntegerValue = numrays
    
    start_trace = time.time()
    # Trace and save a ZRD file for test later
    NSCRayTrace = Sim.TheSystem.Tools.OpenNSCRayTrace()
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
    
    ZRDReader = Sim.TheSystem.Tools.OpenRayDatabaseReader()
    ZRDReader.ZRDFile = pathZMX + os.sep + fileZRD
    
    ZRDReader.RunAndWaitForCompletion()
    
    results = ZRDReader.GetResults();
    dataReader = Sim.BatchRayTrace.ReadZRDData(results)
    maxSegments = 50000000; #50000000 is max
    ZRDData = dataReader.InitializeOutput(maxSegments);

    isFinished = False;
    totalSegRead = 0;
    totalRaysRead = 0;

    start_read = time.time()
    i=0
    
    while isFinished == False and ZRDData is not None:
        readSegments = dataReader.ReadNextBlock(ZRDData);
        if readSegments == 0:
            isFinished = True;
        else:
            test = len(Sim.zosapi.DoubleToNumpy(ZRDData.RayNumber)[:totalSegRead])
            totalSegRead = totalSegRead + readSegments;
            RayNumber =  Sim.zosapi.LongToNumpy(ZRDData.RayNumber)[:totalSegRead]
            # WlUM =  Sim.zosapi.LongToNumpy(ZRDData.WlUM)[:totalSegRead]
            totalRaysRead = np.max(Sim.zosapi.LongToNumpy(ZRDData.RayNumber))
            Level = Sim.zosapi.LongToNumpy(ZRDData.Level)[:totalSegRead]
            # Parent = Sim.zosapi.LongToNumpy(ZRDData.Parent)[:totalSegRead]
            HitObject = Sim.zosapi.LongToNumpy(ZRDData.HitObject)[:totalSegRead]
            # HitFace = Sim.zosapi.LongToNumpy(ZRDData.HitFace)[:totalSegRead]
            InsideOf = Sim.zosapi.LongToNumpy(ZRDData.InsideOf)[:totalSegRead]
            # Status = Sim.zosapi.LongToNumpy(ZRDData.Status)[:totalSegRead]
            X =  Sim.zosapi.DoubleToNumpy(ZRDData.X)[:totalSegRead]
            Y = Sim.zosapi.DoubleToNumpy(ZRDData.Y)[:totalSegRead]
            Z =  Sim.zosapi.DoubleToNumpy(ZRDData.Z)[:totalSegRead]
            L =  Sim.zosapi.DoubleToNumpy(ZRDData.L)[:totalSegRead]
            # M =  Sim.zosapi.DoubleToNumpy(ZRDData.M)[:totalSegRead]
            # N =  Sim.zosapi.DoubleToNumpy(ZRDData.N)[:totalSegRead]
            Exr =  Sim.zosapi.DoubleToNumpy(ZRDData.Exr)[:totalSegRead]
            Exi =  Sim.zosapi.DoubleToNumpy(ZRDData.Exi)[:totalSegRead]
            Eyr =  Sim.zosapi.DoubleToNumpy(ZRDData.Eyr)[:totalSegRead]
            Eyi = Sim.zosapi.DoubleToNumpy(ZRDData.Eyi)[:totalSegRead]
            Ezr = Sim.zosapi.DoubleToNumpy(ZRDData.Ezr)[:totalSegRead]
            Ezi =  Sim.zosapi.DoubleToNumpy(ZRDData.Ezi)[:totalSegRead]
            Intensity =  Sim.zosapi.DoubleToNumpy(ZRDData.Intensity)[:totalSegRead]
            PathLen =  Sim.zosapi.DoubleToNumpy(ZRDData.PathLen)[:totalSegRead]
            # index = Sim.zosapi.DoubleToNumpy(ZRDData.index)[:totalSegRead]
            # startingPhase =  Sim.zosapi.DoubleToNumpy(ZRDData.startingPhase)[:totalSegRead]
            i+=1
        if totalRaysRead >= maxSegments:
            isFinished = True
    
    ZRDReader.Close();
    
    if i>1:
        print('-------------------------------------------------------')
        print('LOOPS IN READING HIGHER THAN 1 VALUES PROBABLY CORRUPTED')
        print('-------------------------------------------------------')
    
    print('----------------------------------------------------')
    print('Filter:                                    %s' % Filter);
    print('Rays read:                                 %i' % totalRaysRead);
    print('Segments read:                             %i' % totalSegRead);
    Mem_used = totalSegRead/maxSegments*100
    print('Percentage of memory for segments use:     %f' % Mem_used);
    print('----------------------------------------------------')
    end_read = time.time()
    print('Time took for reading: ',round(end_read-start_read,2))
    start_write = time.time()
    
    filenpy = pathnpy
    
    with open(filenpy,'wb') as f:
        np.save(f,np.array([Level,HitObject,InsideOf,RayNumber,
                            X,Y,Z,L,Exr,Exi,Eyr,Eyi,Ezr,Ezi,Intensity,PathLen]))
    end_write = time.time()
    print("Time took for writing npy file: ",round(end_write-start_write,2))

    return filenpy

def Load_npy(path):
    #load_npy and transform it into a dataframe with low memory usage
    start_load = time.time()
    
    with open(path, 'rb') as f:
        data = np.load(f,allow_pickle=True)
    
    headers = np.array(["segmentLevel", "hitObj", "insideOf", "numray",
    "x", "y", "z", "n", "exr", "exi", "eyr", "eyi", "ezr", "ezi", "intensity", "pathLength"])
    
    #Sparse fo int columns segmentLevel, hitObj, insideOf and numray
    df = pd.DataFrame(np.transpose(data),columns = headers)
    
    types = [int,int,int,int,float,float,float,float,float,float,float,
             float,float,float,float,float]
    
    for h,t in zip(headers,types):
        df[h] = df[h].astype(t)
        
    def mem_usage(pandas_obj):
        if isinstance(pandas_obj,pd.DataFrame):
            usage_b = pandas_obj.memory_usage(deep=True).sum()
        else: # we assume if not a df it's a series
            usage_b = pandas_obj.memory_usage(deep=True)
        usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
        return "{:03.2f} MB".format(usage_mb)
    
    df_int = df.select_dtypes(include=['int'])
    converted_int = df_int.apply(pd.to_numeric,downcast='unsigned')
    df_float = df.select_dtypes(include=['float'])
    converted_float = df_float.apply(pd.to_numeric,downcast='float')
    
    df[converted_int.columns] = converted_int
    df[converted_float.columns] = converted_float
    df.set_index('numray',inplace=True)
    end_load = time.time()
    
    print("Time took for loading npy file and creating df : ", round(end_load-start_load,2))
    return df