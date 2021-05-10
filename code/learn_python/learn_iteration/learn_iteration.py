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



