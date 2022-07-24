import os.path as osp
import torch.nn as nn

class LinearNet(nn.Module):
    def __init__(self, num_inputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, 1)
    def forward(self, x):
        y = self.linear(x)
        return y
