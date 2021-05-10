# __import__('just_for_test')
# import just_for_test
# import just_for_test.fun # 报错
# from just_for_test import fun

# import just for test # 由于要导入的文件名是有空格的，所以直接import会报错，需要使用__import__函数
# __import__('just for test')
# __import__('just for test.fun') # 报错，在不能使用from just for test import fun的情况下如何从just for test.py中导入fun函数？可以采用下面的写法
just = __import__('just for test')
just.fun()

# fun()


