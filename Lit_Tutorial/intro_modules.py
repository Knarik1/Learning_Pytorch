import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool=True, requires_grad: bool=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        config_dict = {"requires_grad": requires_grad}

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim, **config_dict))
        print(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
        return x


# data
x = torch.rand(2,3)    
# model = MyLinear(2, 3)
print(888)    
print(torch.FloatTensor(2,3))
