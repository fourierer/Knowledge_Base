# learning function or class decorator


###################use class to decorator function#######################
class tracer:
    def __init__(self,func):
        self.calls = 0
        self.func = func
    def __call__(self,*args):
        self.calls += 1
        print('call {} to {}'.format(self.calls,self.func.__name__))
        self.func(*args)

@tracer
# 使用类tracer来装饰函数func1，使得函数在调用时增加了计数功能；
# 相当于func1 = tracer(func1)，此时func1是类tracer的一个实例；
# 所以后续在调用func1(1,2,3)时相当于在调用类tracer中的__call__函数
def func1(a,b,c):
    print(a,b,c)

@tracer
def func2(a,b,c):
    print(a,b,c)

def main1():
    func1(1,2,3)
    # call 1 to func1
    # 1 2 3
    func1(4,5,6)
    # call 2 to func1
    # 4 5 6
    func2(7,8,9)
    # call 1 to func2
    # 7 8 9
    func2(10,11,12)
    # call 2 to func2
    # 10 11 12



###################use function to decorator class#######################


def test(a=1,b=2,c=3):
    print(a,b,c)


if __name__=='__main__':
    # main1()
    print(isinstance(tracer,type)) # True
    print(tracer.__name__) # tracer
    print(tracer)


