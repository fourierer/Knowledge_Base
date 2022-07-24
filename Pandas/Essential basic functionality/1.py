
import numpy as np
from numpy.random import f
import pandas as pd


index = pd.date_range('1/1/2000', periods=8)
# print(index)
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=['A', 'B', 'C'])
# print(df)


########################1.head()函数和tail()函数############################
long_sceries = pd.Series(np.random.randn(1000))
# print(long_sceries.head()) # 头部的元素，默认取5个
# print(long_sceries.tail(3)) # 尾部的元素，传入参数指定最后3个


########################2.属性和基础数据############################
# print(df[:2])
df.columns = [x.lower() for x in df.columns]
# print(df)
# print(s.to_numpy())
# print(np.asarray(s))


########################3.运算维度匹配和扩展（broadasting）############################
df = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)
row = df.iloc[1]
column = df['two']
# print(df)
# print(row)
# print(column)
# print(df.sub(row, axis='columns')) # 从一维的Series扩展到DataFrame
# print(df.sub(row, axis=1))
# print(df.sub(column, axis='index')) # 维度名称是'index'，不是'row'
# print(df.sub(column, axis=0))

s = pd.Series(np.arange(10))
div, rem = divmod(s, 3)
# print(div)
# print(rem)


########################4.比较函数############################
df1 = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)
df2 = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
    }
)
# print(df1.gt(df2))
# eq
# ne
# lt
# gt
# le
# ge


########################5.describe()函数计算各种统计量，nan除外############################
series = pd.Series(np.random.randn(1000))
series[::2] = np.nan
# print(series.describe())
# print(type(series.describe())) # <class 'pandas.core.series.Series'>

frame = pd.DataFrame(np.random.rand(1000, 5), columns=['a', 'b', 'c', 'd', 'e'])
frame.iloc[::2] = np.nan
# print(frame.describe()) # 对于DataFrame, describe()函数是按照列计算统计量
# print(type(frame.describe())) # <class 'pandas.core.frame.DataFrame'>


########################6.idxmin()函数和idxmax()计算最大最小值的索引############################
s1 = pd.Series(np.random.rand(5))
# print(s1.idxmax())
# print(s1.idxmin())

df1 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
# print(df1.idxmin(axis=0))
# print(df1.idxmax(axis=1))


########################7.排序############################
# 对index排序
df = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)
unsorted_df = df.reindex(
    index=["a", "d", "c", "b"], columns=["three", "two", "one"]
)
# print(unsorted_df.sort_index())
# print(unsorted_df.sort_index(axis=1))
# print(unsorted_df.sort_index(ascending=False))
# print(unsorted_df["three"].sort_index())


# 对value排序
df1 = pd.DataFrame(
    {"one": [2, 1, 1, 1], "two": [1, 3, 2, 4], "three": [5, 4, 3, 2]}
)
# print(df1.sort_values(by="two"))
# print(df1[["one", "two", "three"]].sort_values(by=["one", "two"]))







