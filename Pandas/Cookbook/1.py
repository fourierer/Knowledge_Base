import pandas as pd
import numpy as np


############################### df.where() #######################
# replace values where the condition is False, the default replace value is NaN

df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
# print(type(df['AAA'])) # <class 'pandas.core.series.Series'>
# print(type(df.loc[:, 'AAA'])) # <class 'pandas.core.series.Series'>

df_mask = pd.DataFrame(
    {"AAA": [True] * 4, "BBB": [False] * 4, "CCC": [True, False] * 2}
)
# print(df.where(df_mask, -1000))
# print(df.where(df_mask))


s = pd.Series(np.random.randn(10))
# print(s.where(s>0))



# print(type(df.groupby('AAA')['BBB'].idxmin())) # <class 'pandas.core.series.Series'>
# print(df.groupby('AAA')['BBB'].idxmin()) # groupby里面的字段内的数据(即AAA对应的字段)重构后都会变成索引（所以重复的会变成一个），即AAA中的1，2，3都变为索引
'''
AAA
1    1
2    5
3    6
Name: BBB, dtype: int64
'''


############################### 3.Missing data #######################
df = pd.DataFrame(
    np.random.randn(6, 1),
    index=pd.date_range("2013-08-01", periods=6, freq="B"),
    columns=list("A"),
)
# print(df)
df.loc[df.index[3], "A"] = np.nan
new_index = pd.date_range("2013-08-01", periods=7, freq="B")
# print(df.reindex(new_index)) # 按照new_index的顺序重新规划df，并将新出现index的地方补充为NaN
# print(df.reindex(new_index).ffill()) # 使得NaN和前一个数保持一致



############################### 4.Grouping #######################



############################### 5.Merge #######################
rng = pd.date_range("2000-01-01", periods=6)
df1 = pd.DataFrame(np.random.randn(6, 3), index=rng, columns=["A", "B", "C"])
df2 = df1.copy()
# print(df1.append(df2, ignore_index=False))
# print(df1.append(df2, ignore_index=False)['2000-01-01'])
# print(df1.append(df2, ignore_index=False)['A'])

df = pd.DataFrame(
    data={
        "Area": ["A"] * 5 + ["C"] * 2,
        "Bins": [110] * 2 + [160] * 3 + [40] * 2,
        "Test_0": [0, 1, 0, 1, 2, 0, 1],
        "Data": np.random.randn(7),
    }
)
df["Test_1"] = df["Test_0"] - 1
# print(df)
# print(pd.merge(df))
# print(df.index)







