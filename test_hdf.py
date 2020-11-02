# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:07:08 2020

@author: jplan58
"""

import numpy as np
import random 
import psutil
import h5py
import pandas as pd
import time
from sys import getsizeof
import feather
import pandas as pd
import numpy as np

print(psutil.virtual_memory())
arr = np.random.randn(10_000_000) # 10% nulls
arr[::10] = np.nan
df = pd.DataFrame({'column_{0}'.format(i): arr for i in range(10)})
print(psutil.virtual_memory())
# del a
# print(psutil.virtual_memory())
# del x
# print(psutil.virtual_memory())

# with h5py.File('test_hdf.h5',mode='w') as hdf:
#     print(psutil.virtual_memory())
#     hdf.create_dataset('Datas_test',data=x)
#     print(psutil.virtual_memory())
# print(psutil.virtual_memory())
# memory_x = getsizeof(x)
# free_memory_pree = psutil.virtual_memory()[4]
# del x
# free_memory_post = psutil.virtual_memory()[4]
# print(free_memory_pree-free_memory_post+memory_x)
# print(psutil.virtual_memory())
# with h5py.File('test_hdf.h5',mode='r') as hdf:
#     data = hdf.get('Datas_test')
#     # dataset1 = np.array(data)
#     # z = data[]
#     print(psutil.virtual_memory())
# print(psutil.virtual_memory())

print(psutil.virtual_memory())
s = time.time()
with pd.HDFStore('test_hdf.h5') as hdf:
    hdf.put('df',df,format='table',data_columns=True)
e = time.time()
print(psutil.virtual_memory())
print(e-s)
print(psutil.virtual_memory())
with pd.HDFStore('test_hdf.h5') as hdf:
    s = time.time()
    data = hdf.select('/df')
    e = time.time()
    print(psutil.virtual_memory())
print(psutil.virtual_memory())
print(data)
print(e-s)
del data
print(psutil.virtual_memory())