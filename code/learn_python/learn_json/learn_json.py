import json


###################### 基本用法 #######################

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


#################### 高阶用法 ######################

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

