import h5py
import os
import numpy as np
from pathlib import Path

class HLC_HDF5_fast:

    def __init__(self, filenameWithPath):
        self.filename = filenameWithPath
        self.path = "".join([s + '/' for s in self.filename.split('/')[:-1]])

        # Create path if it doesnt exist
        dataDirectory = Path(self.path)
        if not dataDirectory.is_dir():
           os.mkdir(self.path)

        self.h = h5py.File(self.filename, 'a', libver='latest')
        self.h.swmr_mode = True
        self.fileOpen = True

    def __del__(self):
        self.close()


    def close(self):
        if self.fileOpen:
            self.h.close()
            self.fileOpen = False


    def write_data(self, data):
        """ Assumes the data is a dictionary with the following format:
            [table name][field]
            If append=True, append the  data, else crush the file"""

        for tableName, tableContent in data.items():
            if tableName not in self.h.keys():
                grp  = self.h.create_group(tableName)
            else:
                grp = self.h[tableName]
            for fieldName, value in tableContent.items():
                if fieldName not in grp.keys():
                    dset = grp.create_dataset(fieldName, (0,), maxshape=(None,), chunks=True)
                else:
                    dset = grp[fieldName]

                if value == None:
                    break

                if isinstance(value, list):
                    numberOfNewRows = len(value)
                elif isinstance(value, (int, float, complex)):
                    value = [value]
                    numberOfNewRows = 1
                else:
                    continue  # dont write strings or others

                dset.resize((dset.shape[0] + numberOfNewRows), axis=0)
                dset[-numberOfNewRows:] = np.array(value)

                dset.flush()

    def read_data(self, path=None, maxSize=100):
        """ Read the HDF5 file.
            If path is None, returns all the file: BE CAREFUL!!, if the file is too big, you're gonna bust your RAM
            maxSize defines the maximum number of data per field to get
        """
        ret = {}
        if path == None:
            try :
                for tableName in self.h.keys():
                    ret[tableName] = {}
                    for fieldName in self.h[tableName].keys():
                        ret[tableName][fieldName] = list(self.h[tableName][fieldName])
            except:
                ret = None
        else:
            try :
                if self.h[path].shape[0] > maxSize:
                    ret = list(self.h[path][-maxSize:])
                else:
                    ret = list(self.h[path])
            except:
                ret = None

        return ret

    def write_images(self, data, type=None):
       """ Assumes the data is a dictionary with the following format:
           [table name][field]
           If append=True, append the  data, else crush the file"""


       pathData = PathedDictionnary(data)

       for path, value in zip(pathData.pathList, pathData.valueList):
           shp = np.shape(value)
           print(shp)
           if len(shp) == 2:
               numberOfNewElements = 1
               numRows = shp[0]
               numCols = shp[1]
           elif len(shp) == 3:
               numberOfNewElements = shp[0]
               numRows = shp[1]
               numCols = shp[2]

           if path in self.h:
               dset = self.h[path]
           else:

               # dset = self.h.create_dataset(path, (0,numRows,numCols), maxshape=(None,None,None), chunks=True)
               if isinstance(value, str):
                   dt = h5py.special_dtype(vlen=str)
                   dset = self.h.create_dataset(path, (0,numRows,numCols), maxshape=(None,None,None), chunks=True, dtype=dt)
               else:
                   if type:
                       dset = self.h.create_dataset(path, (0,numRows,numCols), maxshape=(None,None,None), chunks=True, dtype=type)
                   else:
                       dset = self.h.create_dataset(path, (0,numRows,numCols), maxshape=(None,None,None), chunks=True)


           dset.resize((dset.shape[0] + numberOfNewElements), axis=0)
           dset[-numberOfNewElements:, :, :] = value



    def read_images(self, path=None, maxSamples=1):
        """ Read the HDF5 file.
            If path is None, returns all the file: BE CAREFUL!!, if the file is too big, you're gonna bust your RAM
            maxSamples defines number of frames to get
        """
        ret = {}
        if path == None:
            try :
                for tableName in self.h.keys():
                    ret[tableName] = {}
                    for fieldName in self.h[tableName].keys():
                        ret[tableName][fieldName] = list(self.h[tableName][fieldName])
            except:
                ret = None
        else:
            try :
                if self.h[path].shape[0] > maxSamples:
                    ret = list(self.h[path][-maxSamples:,:,:])
                else:
                    ret = list(self.h[path])
            except:
                ret = None

        return ret


    def write_dict(self, data, type=None):
        """ Assumes the data is a dictionary with the following format:
            [table name][field]
            If append=True, append the  data, else crush the file"""

        pathData = PathedDictionnary(data)

        for path, value in zip(pathData.pathList, pathData.valueList):
            if value == None:
                continue

            if path in self.h:
                dset = self.h[path]
            else:
                if isinstance(value, str):
                    dt = h5py.special_dtype(vlen=str)
                    dset = self.h.create_dataset(path, (0,), maxshape=(None,), chunks=True, dtype=dt)
                else:
                    if type:
                        dset = self.h.create_dataset(path, (0,), maxshape=(None,), chunks=True, dtype=type)
                    else:
                        dset = self.h.create_dataset(path, (0,), maxshape=(None,), chunks=True)

            if isinstance(value, list):
                numberOfNewRows = len(value)
            elif isinstance(value, (int, float, complex)):
                value = [value]
                numberOfNewRows = 1
            elif isinstance(value, str):
                # value = self.convertToNumber(value)
                # if value == None:
                #     continue
                if len(value) < 50:
                    value.ljust(50)
                elif len(value) > 50:
                    value = value[:49]
                numberOfNewRows = 1
            else:
                continue  # dont write strings or others

            dset.resize((dset.shape[0] + numberOfNewRows), axis=0)
            dset[-numberOfNewRows:] = np.array(value)

            dset.flush()

    def convertToNumber(self, s):
        try:
            return int(s)
        except ValueError:
            pass

        try:
            return float(s)
        except ValueError:
            pass

        return None

class PathedDictionnary():

    def __init__(self, my_dict=None):
        self.pathList = []
        self.valueList = []
        if my_dict:
            self.dict_path("", my_dict)

    def dict_path(self, path, my_dict):
        for k, v in my_dict.items():
            if isinstance(v, dict):
                if path == "":
                    self.dict_path(k, v)
                else:
                    self.dict_path(path + "/" + k, v)
            else:
                if path == "":
                    self.pathList.append(str(k))
                else:
                    self.pathList.append(path + "/" + str(k))
                self.valueList.append(v)

    def clear(self):
        self.pathList = []
        self.valueList = []

if __name__ == '__main__':

    HH = HLC_HDF5_fast('test_fast.hdf5')

    test_data = {'table1': {'field1': [1,2], 'field2': [3,4]}}
    test_image = {'test_device': {'image': [[[1,2], [3,4]], [[1,2], [3,4]] ]}}

    HH.write_data(test_data)
    read_data = HH.read_data()

    HH.write_images(test_image)
    read_image = HH.read_images('test_device/image')

    pd = PathedDictionnary()
    pd.dict_path("",test_data)

    print(pd.pathList, pd.valueList)


    print(read_data)
    print(read_image)

    del HH
