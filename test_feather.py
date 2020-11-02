# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:04:43 2020

@author: jplan58
"""

import feather
import pandas as pd
import numpy as np
import time
import psutil

arr = np.random.randn(10_000_000) # 10% nulls
arr[::10] = np.nan
df = pd.DataFrame({'column_{0}'.format(i): arr for i in range(10)})
start= time.time()
print(psutil.virtual_memory())
feather.write_dataframe(df, 'test.feather')
end= time.time()
print(start-end)
print(psutil.virtual_memory())

print(psutil.virtual_memory())
start= time.time()
df = feather.read_dataframe('test.feather')
end= time.time()
print(start-end)
print(psutil.virtual_memory())
