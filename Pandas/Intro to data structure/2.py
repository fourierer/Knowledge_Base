#========================DataFrame==============================
# DataFrame是二维有标签数据

from typing import Collection
import numpy as np
import pandas as pd

########################1.Series dict生成DataFrame############################
d = {
    'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
    'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
}
df1 = pd.DataFrame(d)
df2 = pd.DataFrame(d, index=['d', 'b', 'a'])
df3 = pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three'])
# print(df1)
# print(df2)
# print(df3)
print(df1.index)
print(df1.columns)


########################2.array或者list的dict(字典中的值是列表)生成DataFrame############################
d = {'one': [1, 2, 3, 4], 'two': [4, 3, 2, 1]}
df1 = pd.DataFrame(d)
df2 = pd.DataFrame(d, index=['a', 'b', 'c', 'd'])
# print(df1)
# print(df2)


########################3.dict的list(字典列表，列表元素是字典)生成DataFrame############################
d1 = [{'a':1, 'b':2}]
d2 = [{"a": 1, "b": 2}, {"a": 5, "b": 10, "c": 20}]
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2, index=['first', 'second'])
df3 = pd.DataFrame(d2, columns=['a', 'b'])
# print(df1)
# print(df2)
# print(df3)


########################4.DataFrame获取、设置和删除column属性，和Series一样获取、设置和删除index属性############################
d = {
    'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
    'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
}
df1 = pd.DataFrame(d)
# print(df1)
# print(df1['one'])

df1['three'] = df1['one'] * df1['two']
df1['flag'] = df1['one'] > 2
# print(df1)

del df1['two']
three = df1.pop('three')
# print(df1)
# print(three)
# print(type(three)) # <class 'pandas.core.series.Series'>

df1['foo'] = 'bar'
df1['one_trunc'] = df1['one'][:2]
# print(df1)

df1.insert(1, 'bar', df1['one']) # 1表示插入的索引位置
# print(df1)
'''
   one  bar   flag  foo  one_trunc
a  1.0  1.0  False  bar        1.0
b  2.0  2.0  False  bar        2.0
c  3.0  3.0   True  bar        NaN
d  NaN  NaN  False  bar        NaN
'''


########################4.DataFrame索引############################
df1['one'] # 获取某一列，结果为Series
df1.loc['b'] # 通过index属性获取某一行，结果为Series
df1.iloc[2] # 通过index索引序号获取某一行，结果为Series
df1[1:3] # 通过slice获取某些行，结果为DataFrame
df1[[True, False, True, False]] # 通过布尔数组获取某些行，结果为DataFrame


########################5.DataFrame数据对齐和算术############################
df1 = pd.DataFrame(np.random.randn(10, 4), columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.random.rand(7, 3), columns=['a', 'b', 'e'])
# print(df1 + df2)
# print(df1 - df1.iloc[0])
# print(df1 * 5 + 2)
# print(1 / df1)
# print(df1 ** 4)


########################6.DataFrame转置############################
df1 = pd.DataFrame(np.random.randn(10, 4), columns=['a', 'b', 'c', 'd'])
# print(df1[:5].T)
# print(df1[:5].T.columns) # RangeIndex(start=0, stop=5, step=1)
# print(df1[:5].T.index) # Index(['a', 'b', 'c', 'd'], dtype='object')


########################7.DataFrame和numpy函数的互操作性############################
df2 = pd.DataFrame(np.random.rand(7, 3), columns=['a', 'b', 'e'])
print(np.exp(df2))
print(np.asarray(df2))



