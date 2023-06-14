import torch
import torch.nn as nn
import numpy as np

# 第一种方式
class GlobalMaxPoolid(nn.Module):
    def __init__(self):
        super(GlobalMaxPoolid,self).__init__()
    def forward(self, x):
        return torch.max_pool1d(x,kernel_size=x.shape[2])

a = torch.tensor(np.arange(24),dtype=torch.float).view(2,3,4).cuda()
gmp1 = GlobalMaxPoolid()
print(gmp1(a))

class GlobalAvgPoolid(nn.Module):
    def __init__(self):
        super(GlobalAvgPoolid,self).__init__()
    def forward(self, x):
        return torch.avg_pool1d(x,kernel_size=x.shape[2])

b = torch.tensor(np.arange(24),dtype=torch.float).view(2,3,4).cuda()
gmp2 = GlobalAvgPoolid()
print(gmp2(b))

class GlobalAvg_MaxPoolid(nn.Module):
    def __init__(self):
        super(GlobalAvg_MaxPoolid,self).__init__()
    def forward(self, x):
        return torch.avg_pool1d(x,kernel_size=x.shape[2]) + torch.max_pool1d(x,kernel_size=x.shape[2])

c = torch.tensor(np.arange(24),dtype=torch.float).view(2,3,4).cuda()
gmp3 = GlobalAvg_MaxPoolid()
print(gmp3(b))
