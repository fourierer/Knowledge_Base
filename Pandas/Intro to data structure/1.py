#========================Series==============================
# Series是一维有标签数据

import numpy as np
import pandas as pd

########################1.基础用法############################
# s = pd.Series(data, index=index)

s = pd.Series(np.random.rand(3), index=['a', 'b', 'c'])
# print(s)
# print(s.index)

s = pd.Series(np.random.rand(3))
# print(s)
# print(s.index)

data = {'a':1, 'b':2, 'c':3}
s = pd.Series(data)
# print(s)

s = pd.Series(data, index=['a'])
# print(s.index)
# print(s)

s = pd.Series(data, index=['b', 'a', 'c', 'd'])
# print(s.index)
# print(s)

s = pd.Series(5, index=[1, 'a', 'b'])
# print(s.index)
# print(s)


########################2.Series类似numpy############################
s = pd.Series(np.random.rand(5), index=['a', 'b', 'c', 'd', 'e'])
# print(s[0])
# print(s[1])
# print(s[:3])
# print(s.median())
# print(s[s>s.median()])
# print(s[[4, 3, 1]])
# print(np.exp(s))
# print(s.dtype)
# print(s.to_numpy())
# print(type(s.to_numpy())) # <class 'numpy.ndarray'>


########################3.Series类似dict############################
s = pd.Series(np.random.rand(5), index=['a', 'b', 'c', 'd', 'e'])
# print(s)
# print(s['a'])
# del s['a']
# print(s)
# s['e'] = 12
# print(s)
# print('c' in s)


########################4.Series中的向量操作############################
s = pd.Series(np.random.rand(5), index=['a', 'b', 'c', 'd', 'e'])
# print(s+s)
# print(2*s)
# print(np.exp(s))
# print(s[1:])
# print(s[:-1])
# print(s[1:] + s[:-1]) # index不匹配的值设为Nan


########################5.Series的name属性############################
s1 = pd.Series(np.random.rand(5), index=['a', 'b', 'c', 'd', 'e'])
s2 = pd.Series(np.random.rand(5), index=['a', 'b', 'c', 'd', 'e'], name='something')
s3 = s2.rename('different') # s2和s3代表不同的对象
# print(s1.name) # None
# print(s2.name) # something
# print(s3.name) # different

