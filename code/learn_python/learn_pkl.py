# 一. JSON 是一个文本序列化格式（它输出 unicode 文本，尽管在大多数时候它会接着以 utf-8 编码），而 pickle 是一个二进制序列化格式；
# 二. JSON 是我们可以直观阅读的，而 pickle 不是；
# 三. JSON是可互操作的，在Python系统之外广泛使用，而pickle则是Python专用的；
# 四. 默认情况下，JSON 只能表示 Python 内置类型的子集，不能表示自定义的类；但 pickle 可以表示大量的 Python 数据类型（可以合理使用 Python 的对象内省功能自动地表示大多数类型，复杂情况可以通过实现 specific object APIs 来解决）。


import  pickle
import numpy as np

# 写入pkl文件
# 此处'wb'和'rb'相同，都是二进制文件
x = np.array([1,2])
y = {'a':1,'b':2}
with open('y.pkl','wb') as f:
    pickle.dump(y, f)

# 读取pkl文件
# 重点是rb和r的区别，rb是打开二进制文件，r是打开文本文件
with open('y.pkl', 'rb') as f:
    data = pickle.load(f)
    print(type(data))
    print(data)

