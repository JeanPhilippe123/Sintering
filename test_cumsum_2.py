import pandas as pd
import dask.dataframe as ddf
from dask import delayed, compute
import numpy as np

df = pd.DataFrame(np.array([0,1,2,3,4,5]),columns=['A'],index=[0,1,1,2,2,2])
df['ones']=1
df['twos']=2

dadf = ddf.from_pandas(df, npartitions=3)
cum = dadf.groupby(dadf.index)
res0 = cum['A'].apply(np.cumsum,meta='object').compute()
print(res0)

#don't work for whatever reason

# df = pd.DataFrame(np.array([0,1,2,3,4,5]),columns=['A'],index=[0,1,1,2,2,2])
# df['ones']=1
# df['twos']=2

# dadf = ddf.from_pandas(df, npartitions=3)
# cum = dadf.groupby(dadf.index)
# res0 = cum['A'].apply(np.cumsum,meta='object').compute()
# print(res0)