import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from setting import logger
from layers.sampler import node_map
from layers.attention import SelfAttentionLayer


class SageLayer(nn.Module):
    """
    single view
    """
    def __init__(self, in_dim, out_dim, drop_rate=0.4, cuda=False):
        super(SageLayer, self).__init__()
        self.drop_rt = drop_rate
        self.cuda = cuda
        self.weight = nn.Parameter(torch.FloatTensor(out_dim, in_dim * 2))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes, pre_hidden_embs, info_neighs, layer_i=1):
        _, _, unique_nodes = info_neighs
        nodes = [unique_nodes[x] for x in nodes]
        self_feats = pre_hidden_embs[nodes]
        aggregate_feats = self.message(nodes, pre_hidden_embs, info_neighs, layer_i)
        out_emb = self.update(self_feats, aggregate_feats)
        out_emb = F.dropout(out_emb, self.drop_rt)
        return out_emb

    def message(self, nodes, pre_hidden_embs, info_neighs, layer_i=1):
        """
        obtain enviroment
        """
        _, samp_neighs, unique_nodes = info_neighs
        assert len(nodes) == len(samp_neighs)
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        embed_matrix = pre_hidden_embs
        m_vector = mask.mm(embed_matrix)
        return m_vector
    
    
    def update(self, self_feats, aggregate_feats):
        """
        update hindden state
        """
        combined = torch.cat([self_feats, aggregate_feats], dim=1)
        combined = F.relu(self.weight.mm(combined.t()))
        return combined.t()
