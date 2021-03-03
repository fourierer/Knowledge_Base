

def fun1(*args):
    print(args)

def fun2(a,*args):
    print('a:',a)
    print('args:',args)

def fun3(a,b,c):
    print(a,b,c)


if __name__=='__main__':
    fun1(1,2,3,4,5)
    '''
    *用于形参前面，将传的值进行打包成tuple赋给形参
    (1, 2, 3, 4, 5)
    '''
    fun2(1,2,3,4,5)
    '''
    a: 1
    args: (2, 3, 4, 5)
    '''
    fun3(*(1,2,3))
    '''
    *用于tuple或者list前面，将tuple或者list进行拆分，分别赋值给形参
    1 2 3
    '''
    # **kwargs同理，不过用于字典，详情见连接：https://lixiaoqian.blog.csdn.net/article/details/81288741