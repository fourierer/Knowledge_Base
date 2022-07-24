
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


########################1.Series和DataFrame基本索引操作############################

# Series[label] # 返回scalar value
# DataFrame[columns] # 返回Series corresponding to colname
# Series.loc[index]
# DataFrame.loc[row_index, col_columns] # 通过index和columns值访问
# DataFrame.iloc[integer]
# DataFrame.iloc[row_index, col_columns] # 通过数字索引访问


dates = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 4), index=dates, columns=['A', 'B', 'C', 'D'])
# print(df)

df[['A', 'B']] = df[['B', 'A']] # 'A', 'B'互换
# print(df)

df.loc[:, ['A', 'B']] = df[['B', 'A']] # 不改变df，column alignment is before value assignment
# print(df)

# print(df[['A', 'B']].to_numpy()) # 8行2列的数组
df.loc[:, ['B', 'A']] = df[['A', 'B']].to_numpy() # 'A', 'B'再次互换
# print(df)

# print(df.loc['1/1/2000'])
# print(df.loc[0]) # error, loc不能直接用数字索引，只能通过键值
# print(df.iloc[0])



########################2.通过属性访问Series和DataFrame的index或者columns############################

sa = pd.Series([1, 2, 3], index=list('abc'))
dfa = df.copy()
sa.a = 5
# print(sa)
# print(dfa.A)


########################3.Slicng Ranges############################
s = pd.Series(np.random.randn(8))
df = pd.DataFrame(np.random.randn(8, 3), index=np.arange(8), columns=list('abc'))

# print(s[:5])
# print(s[::2])
# print(s[::-1])

# print(df[:3])
# print(df[::-1])


########################4.通过label选择############################
dfl = pd.DataFrame(np.random.randn(5, 4),
                   columns=list('ABCD'),
                   index=pd.date_range('20130101', periods=5))
# print(dfl.loc[2:3]) # error, loc不能用数字索引
# print(dfl.loc['20130102':'20130104'])

s1 = pd.Series(np.random.randn(6), index=list('abcdef'))
# print(s1.loc['b'])
# print(s1.loc['c':]) # 从'c'开始取值
s1.loc['c':] = 0
# print(s1)

df1 = pd.DataFrame(np.random.randn(6, 4),
                   index=list('abcdef'),
                   columns=list('ABCD'))
# print(df1.loc[['a', 'b', 'd'], :])
# print(df1.loc['d':, 'A':'C'])
# print(df1.loc['a']>0)
# print(df1.loc[:, df1.loc['a']>0])

s = pd.Series(list('abcde'), index=[0, '3', 2, 5, 4])
# print(s.loc['3':5])
# print(s.loc[0]) # 0是index对应的值，不是数字索引0
# print(s.loc['a'])


########################5.通过位置（数字）选择############################
s1 = pd.Series(np.random.randn(5), index=list(range(0, 10, 2)))
# print(s1.iloc[3])
# print(s1.iloc[:3])
s1.loc[:3] = 0
# print(s1)

df1 = pd.DataFrame(np.random.randn(6, 4),
                   index=list(range(0, 12, 2)),
                   columns=list(range(0, 8, 2)))
# print(df1.iloc[:3])
# print(df1.iloc[1:5, 2:4])
# print(df1.iloc[[1, 3, 5], [1, 3]])



