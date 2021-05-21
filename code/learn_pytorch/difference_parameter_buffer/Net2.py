import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.my_tensor = torch.randn(1) # 参数直接作为模型类成员变量
        self.register_buffer('my_buffer', torch.randn(1)) # 参数注册为 buffer
        self.my_param = nn.Parameter(torch.randn(1))

    def forward(self, x):
            return x

model = MyModel()
print(model.state_dict())
model.cuda()
print(model.my_tensor)
print(model.my_buffer)