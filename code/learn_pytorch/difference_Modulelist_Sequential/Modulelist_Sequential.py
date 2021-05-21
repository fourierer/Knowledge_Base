import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_dim=100, hidden_dim=50, out_dim=10):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        # x = self.layer(x)
        for f in self.layer:
            x = f(x)
        return x

if __name__=='__main__':
    model = Net()
    x = torch.ones(2,100)
    # 网络中有BN层，在输入时应当输入两个或以上的样本
    # 如果输入x = torch.ones(1,100)时会报错
    y = model(x)
    print(y)
