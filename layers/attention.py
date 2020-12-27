import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from setting import logger

class AttentionLayer(nn.Module):
    """
    co-attention, fuse multiple views
    """
    def __init__(self, in_features, out_features, drop_rt=0.4):
        super(AttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop_rt=0.4
        self.linear = nn.Linear(in_features, out_features)
        init.xavier_normal_(self.linear.weight)
        self.q = nn.Parameter(torch.randn(out_features))

    def forward(self, inputs):
        hidden = self.linear(inputs)
        atten = torch.matmul(hidden, self.q)
        atten_cof = F.softmax(atten, dim=1)
        atten_cof = atten_cof.reshape(atten_cof.shape[0], 1, atten_cof.shape[1])
        fuse_embed = torch.matmul(atten_cof, hidden)
        fuse_embed.squeeze_()
        fuse_embed = F.dropout(fuse_embed, self.drop_rt)
        return fuse_embed
