import torch
import json
import numpy as np


a = torch.rand([5,3])
# print(a)
# print(isinstance(a, torch.Tensor)) # True

def tensor2list(t):
    if isinstance(t, torch.Tensor):
        # torch.Tensor需要先转换成numpy，然后转换成list才可以通过list序列化
        return t.numpy().tolist()

def list2tensor(l):
    if isinstance(l, list):
        l1 = np.array(l)
        l2 = torch.from_numpy(l1)
        return l2


json_str = json.dumps(a, default=tensor2list)
# print(type(json_str)) # <class 'str'>
test_list = json.loads(json_str)
# print(type(test_list)) # <class 'list'>

with open('./test.json', 'w') as f:
    json.dump(a, f, default=tensor2list)

with open('./test.json', 'r') as f:
    test_list = json.load(f)
print(type(test_list)) # list

with open('./test.json', 'r') as f:
    test_tensor = json.load(f, object_hook=list2tensor)
print(type(test_tensor)) # list, 由于列表中还嵌套列表，这里定义的list2tensor并不能满足将所有的list都转换为np.array(),所以没有成功转换为torch.Tensor
