import pandas as pd
import datetime as dt
import numpy as np

# 1.时间戳
t1 = pd.Timestamp(dt.datetime(2014, 6, 1))
t2 = pd.Timestamp('2014-6-1')
t3 = pd.Timestamp(2014, 6, 1)
print(t1)
print(t2)
print(t3)
# 2014-06-01 00:00:00
# 2014-06-01 00:00:00
# 2014-06-01 00:00:00


# 2.时间间隔
p1 = pd.Period('2014-06') # 默认间隔为月
p2 = pd.Period('2014-06', freq='D') # 间隔为天
print(p1)
print(p2)
# 2014-06
# 2014-06-01


# 3.Timestamp和Period作为索引
dates = [pd.Timestamp('2014-06-01'), 
         pd.Timestamp('2014-06-02'),
         pd.Timestamp('2014-06-03')]
ts_data = pd.Series(np.random.rand(3), dates) # 将dates作为索引
print(ts_data)
print(type(ts_data.index))
'''
2014-06-01    0.697095
2014-06-02    0.860069
2014-06-03    0.003550
dtype: float64
<class 'pandas.core.indexes.datetimes.DatetimeIndex'>
'''


# 4.转换时间戳
date = pd.to_datetime(['2012/11/23', '2012.12.31', 'Jul 31, 2012', '2012-01-10'])
print(date)
# DatetimeIndex(['2012-11-23', '2012-12-31', '2012-07-31', '2012-01-10'], dtype='datetime64[ns]', freq=None)
print(pd.Series([1, 2, 3, 4], index=date))
'''
2012-11-23    1
2012-12-31    2
2012-07-31    3
2012-01-10    4
dtype: int64
'''

date1 = pd.to_datetime('2018/11/12', format='%Y/%m/%d')
date2 = pd.to_datetime('11-11-2018 00:00', format='%d-%m-%Y %H:%M')
print(date1)
print(date2)
# 2018-11-12 00:00:00
# 2018-11-11 00:00:00






