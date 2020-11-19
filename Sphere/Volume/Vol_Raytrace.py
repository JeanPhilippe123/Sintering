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

def retrieve_data(Sim,ZRDReader):
    results = ZRDReader.GetResults();
    dataReader = Sim.BatchRayTrace.ReadZRDData(results)
    maxSegments = 1_000_000;
    ZRDData = dataReader.InitializeOutput(maxSegments);

    isFinished = False;
    totalSegRead = 0;
    totalRaysRead = 0;

    start_read = time.time()
    i=0
    
    data = da.from_array(np.array([[]]*17),chunks=[1,1])
    while isFinished == False and ZRDData is not None:
        readSegments = dataReader.ReadNextBlock(ZRDData);
        if readSegments == 0:
            isFinished = True;
        else:
            #Get the right data from the file
            totalSegRead = totalSegRead + readSegments;
            totalRaysRead = np.max(Sim.zosapi.LongToNumpy(ZRDData.RayNumber))
            RayNumber = Sim.zosapi.LongToNumpy(ZRDData.RayNumber)[:readSegments]
            Level = Sim.zosapi.LongToNumpy(ZRDData.Level)[:readSegments]
            HitObject = Sim.zosapi.LongToNumpy(ZRDData.HitObject)[:readSegments]
            InsideOf = Sim.zosapi.LongToNumpy(ZRDData.InsideOf)[:readSegments]
            X =  Sim.zosapi.DoubleToNumpy(ZRDData.X)[:readSegments]
            Y = Sim.zosapi.DoubleToNumpy(ZRDData.Y)[:readSegments]
            Z =  Sim.zosapi.DoubleToNumpy(ZRDData.Z)[:readSegments]
            N =  Sim.zosapi.DoubleToNumpy(ZRDData.N)[:readSegments]
            Exr = Sim.zosapi.DoubleToNumpy(ZRDData.Exr)[:readSegments]
            Exi = Sim.zosapi.DoubleToNumpy(ZRDData.Exi)[:readSegments]
            Eyr = Sim.zosapi.DoubleToNumpy(ZRDData.Eyr)[:readSegments]
            Eyi = Sim.zosapi.DoubleToNumpy(ZRDData.Eyi)[:readSegments]
            Ezr = Sim.zosapi.DoubleToNumpy(ZRDData.Ezr)[:readSegments]
            Ezi = Sim.zosapi.DoubleToNumpy(ZRDData.Ezi)[:readSegments]
            Intensity = Sim.zosapi.DoubleToNumpy(ZRDData.Intensity)[:readSegments]
            PathLen = Sim.zosapi.DoubleToNumpy(ZRDData.PathLen)[:readSegments]
            Indice = Sim.zosapi.DoubleToNumpy(ZRDData.index)[:readSegments]
            data = da.concatenate([data,[Level,HitObject,InsideOf,RayNumber,X,Y,Z,N,Exr,Exi,Eyr,Eyi,Ezr,Ezi,Intensity,PathLen,Indice]],axis=1).persist()
            i+=1
        if totalRaysRead >= maxSegments:
            isFinished = True
    
    ZRDReader.Close();
        
    #Create DataFrame
    headers = np.array(["segmentLevel", "hitObj", "insideOf", "numray",
    "x", "y", "z", "n", "exr", "exi", "eyr", "eyi", "ezr", "ezi", "intensity", "pathLength", "Indice"])

    print('----------------------------------------------------')
    print('Rays read:                                 %i' % totalRaysRead);
    print('Segments read:                             %i' % totalSegRead);
    print('----------------------------------------------------')
    end_read = time.time()
    print('Time took for reading: ',round(end_read-start_read,2))
    
    #Create Dataframe from the retrieve data
    data = data.transpose()
    df = data.to_dask_dataframe(columns=headers)
    return df

# @profile
def Shoot(Sim,Filter,numrays,path_parquet,nameZRD,path_metadata=None):
    
    # Open file
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
    
    if path_metadata!=None:
        #Write metadata
        np.save(path_metadata,Sim.array_objects())
    
    #Retrieve Datas from simulation into a dataframe
    df = retrieve_data(Sim,ZRDReader)
    
    start_write = time.time()
    
    # df.repartition(npartitions=10,force=True)
    df = df.set_index('numray', sorted=True, partition_size='100 MiB')
    
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
    
    # df.repartition(npartitions=10,force=True)
    # df = df.set_index('numray', partition_size='100 MiB')
    
    # #Add the good index
    # df = df.persist()
    
    end_load = time.time()
    
    print("Time took for loading {} paquet file : ".format(name), round(end_load-start_load,2))
    return df