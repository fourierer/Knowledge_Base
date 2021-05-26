import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=3, out_dim=2):
        super(Net, self).__init__()
        self.block = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        self.A = torch.Tensor(3,5)
        self.B = nn.Parameter(self.A)
        # 或者如下定义parameter参数
        # param = nn.Parameter(self.A)
        # self.register_parameter('B', param)
        self.register_buffer('C', self.A)
    
    def forward(self, x):
        x = self.block(x)
        return x

if __name__=='__main__':
    model = Net()


    # 访问模型中的parameter类型的参数，包括参数名
    for name, param in model.named_parameters():
        print(name, type(param))
    # 输出如下，从输出结果中可以看出，网络中定义的参数都是torch.nn.parameter.Parameter类，
    # 而其中的buffer类型或者tensor类型的变量都不在模型的参数列表中
    '''
    B <class 'torch.nn.parameter.Parameter'>
    block.0.weight <class 'torch.nn.parameter.Parameter'>
    block.0.bias <class 'torch.nn.parameter.Parameter'>
    block.2.weight <class 'torch.nn.parameter.Parameter'>
    block.2.bias <class 'torch.nn.parameter.Parameter'>
    '''


    # 访问模型中的parameter参数，不包括参数名
    for param in model.parameters():
        print(param)
    # 输出如下，只有参数，没有参数名
    '''
    Parameter containing:
tensor([[1.9514e-19, 1.8314e+25, 6.9768e+22, 4.0069e+24, 2.7088e+23],
        [1.9435e-19, 7.2127e+22, 4.7428e+30, 4.6534e+33, 1.7753e+28],
        [1.3458e-14, 6.4610e+19, 2.0618e-19, 1.8545e+25, 6.9767e+22]],
       requires_grad=True)
Parameter containing:
tensor([[ 0.3812,  0.0515, -0.5280],
        [-0.0494,  0.3105, -0.1022],
        [-0.4689,  0.3737, -0.4375]], requires_grad=True)
Parameter containing:
tensor([ 0.3313, -0.3482,  0.0369], requires_grad=True)
Parameter containing:
tensor([[ 0.4688,  0.5602, -0.0133],
        [-0.1683,  0.1731,  0.4286]], requires_grad=True)
Parameter containing:
tensor([ 0.1057, -0.0223], requires_grad=True)
    '''


    # 访问模型中的buffer参数,同样有buffers()与named_buffers()，区别是是否输出buffer的变量名
    for buffer in model.named_buffers():
        print(buffer)
    
    
    # 访问模型中所有的参数，包括parameter和buffer
    print(type(model.state_dict())) # <class 'collections.OrderedDict'>
    print(model.state_dict())
    # for k, v in model.state_dict().items():
        # print(k,v)



