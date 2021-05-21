a = '123'
# for i, value in enumerate(a):
    # print(i,value)


d = {'a':3,'b':4,'c':2}
print(d.items())
for key,value in d.items():
    print(key,value)
__private = 3
for index,key in enumerate(d):
    print(key)

for k, v in d.items():
    print(k,v)

def _private_1(name):
    return 'Hello, %s' % name

def _private_2(name):
    return 'Hi, %s' % name

def greeting(name):
    if len(name) > 3:
        return _private_1(name)
    else:
        return _private_2(name)

import logging
logging.basicConfig(level=logging.INFO)

logging.info('batman')
#print(10/0)


import os
logging.info('the operate system is:' + os.name)
logging.info('the detailed information of the operate system is:' + str(os.uname()))



# OrderedDict
# python中的字典一般是无序的，但是python中有个collection模块，里面自带了一个子类OrderedDict，实现了对字典对象元素的排序
import collections

print("Regular dictionary")
d = {}
d['a'] = 'A'
d['b'] = 'B'
d['c'] = 'C'
for k, v in d.items():
    print(k, v)

print("\nOrder dictionary")
d1 = collections.OrderedDict()
d1['a'] = 'A'
d1['b'] = 'B'
d1['c'] = 'C'
d1['1'] = '1'
d1['2'] = '2'
for k, v in d1.items():
    print(k, v)
'''
输出：
Regular dictionary
a A
c C
b B

Order dictionary
a A
b B
c C
1 1
2 2
'''
