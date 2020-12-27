import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifer(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.3):
        super(Classifer2, self).__init__()
        self.drop_rt = drop_rate

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, input):
        h1 = self.fc1(input)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, self.drop_rt)
        out = self.out(h1)
        return out
