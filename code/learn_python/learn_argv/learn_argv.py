import sys

a = sys.argv[0]
b = sys.argv[1]
c = sys.argv[0:]

# 在命令行输入 python learn_argv.py test
print(a) # learn_argv.py
print(b) # test
print(c) # ['learn_argv.py', 'test']
print(type(a)) # <class 'str'>
print(type(b)) # <class 'str'>
print(type(c)) # <class 'list'>

