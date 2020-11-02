import pandas as pd
import dask.dataframe as ddf
from dask import delayed, compute

df = pd.DataFrame(dict(a=list('aabbcc')),
                  index=pd.date_range(start='20100101', periods=6))
df['ones']=1
df['twos']=2

dadf = ddf.from_pandas(df, npartitions=3)
cum = dadf.groupby(['a'])
res0 = compute(cum['ones'].cumsum())
print(res0)
