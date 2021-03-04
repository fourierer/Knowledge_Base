### 常用Python问题



#### 一、json用法

1.json基本用法

把变量从内存中变成可存储或传输的过程称之为序列化，Python中有两个模块可以进行序列化：

（1）json: 用于字符串和python数据类型间进行转换

（2）pickle： 用于python特有的类型和python的数据类型间进行转换

Json模块提供了四个功能：dumps、dump、loads、load

pickle模块提供了四个功能：dumps、dump、loads、load

json.dumps把数据类型转换成字符串，dump把数据类型转换成字符串并存储在文件中；loads把字符串转换成数据类型，load把文件打开从字符串转换成数据类型。json是可以在不同语言之间交换数据的，而pickle只在python之间使用。json只能序列化最基本的数据类型（列表、字典、列表、字符串、数字、），但对于日期格式，类对象以及numpy数组，josn就不能序列化。pickle可以序列化所有的数据类型，包括类，函数都可以序列化。

示例代码：

```python
import json

test_dict = {'name':'Bob','age':20,'score':88}
json_str = json.dumps(test_dict) # json格式数据
# print(isinstance(json_str,str)) # True

new_dict = json.loads(json_str) # json格式数据转换为python数据类型
# print(isinstance(new_dict,dict)) # True

with open('./test_json.json', 'w') as f:
    json.dump(new_dict, f)

with open('./test_json.json', 'r') as f:
    load_dict = json.load(f)

# print(isinstance(load_dict,dict)) # True
```



2.json高阶用法

json可以将列表，字典等数据类型进行序列化，但对于一些复杂的数据类型会报错，如：

```python
import numpy as np
test_numpy = np.array([1,2])
# json_str = json.dumps(test_numpy)
# print(json_str) # 报错：TypeError: Object of type ndarray is not JSON serializable

class Test_Class(object):
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score
test_class = Test_Class('Bob', 20, 88)
# json_str = json.dumps(test_class)
# print(json_str) # TypeError: Object of type Test_Class is not JSON serializable
# print(test_class.__dict__) # {'name': 'Bob', 'age': 20, 'score': 88}，python中预置的__dict__属性，是保存类实例或对象实例的属性变量键值对字典
```

观察dumps方法的参数列表，除了第一个必须的obj参数外，dumps方法还提供了一大堆的可选参数。( https://docs.python.org/3/library/json.html#json.dumps ）。这些可选参数就是让我们来定制JSON序列化。前面的代码之所以无法把Student类实例序列化为JSON，是因为默认情况下，dumps()方法不知道如何将Student实例变为一个JSON对象。可选参数default就是把任意一个对象变成一个可序列为JSON的对象，只需要为Student专门写一个转换函数，再把函数传进去即可：

具体解决方法如下：

```python
# 测试一
import numpy as np
test_numpy = {'a':np.array([1,2])}
# json_str = json.dumps(test_numpy)
# print(json_str) # 报错：TypeError: Object of type ndarray is not JSON serializable

def numpy2list(n):
    return n.tolist()

json_str = json.dumps(test_numpy, default=numpy2list)
# print(json_str)


# 测试二
class Test_Class(object):
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score
test_class = Test_Class('Bob', 20, 88)
# json_str = json.dumps(test_class)
# print(json_str) # TypeError: Object of type Test_Class is not JSON serializable
# print(test_class.__dict__) # {'name': 'Bob', 'age': 20, 'score': 88}，python中预置的__dict__属性，是保存类实例或对象实例的属性变量键值对字典


def Test_Class2dict(d):
    # return d.__dict__
    return {'name':d.name,'age':d.age,'score':d.score}

json_str = json.dumps(test_class, default = Test_Class2dict)
# print(json_str)


# 测试三
test_dict = {'a':test_numpy,'b':test_class}
# json_str = json.dumps(test_dict)
# print(json_str) # 报错

def numpy_TestClass2json(d):
    if isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, Test_Class):
        return d.__dict__

json_str = json.dumps(test_dict, default = numpy_TestClass2json)
print(json_str) # {"a": {"a": [1, 2]}, "b": {"name": "Bob", "age": 20, "score": 88}}

```





#### 二、多进程与多线程

参考 https://www.liaoxuefeng.com/wiki/1016959663602400/1017627212385376

一般编写的Python程序是执行单任务的进程，也就是只有一个线程。如果我们要同时执行多个任务有两种解决方案：一是启动多个进程，每个进程虽然只有一个线程，但多个进程可以一块执行多个任务；二是启动一个进程，在一个进程内启动多个线程，这样多个线程也可以一块执行多个任务；三是启动多个进程，每个进程再启动多个线程，这样同时执行的任务就会更多，当然这种模型更复杂，实际很少采用。总结一下就是，多任务的实现有3种方式：

（1）多进程模式；

（2）多线程模式；

（3）多进程+多线程模式。



1.多进程



#### 三、正则表达式

参考 https://www.liaoxuefeng.com/wiki/1016959663602400/1017639890281664











