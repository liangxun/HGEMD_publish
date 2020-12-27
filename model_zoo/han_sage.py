import torch
import torch.nn as nn
from torch.nn import init, Linear
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
import time
import random
import os

from layers.MessagePassing import SageLayer
from layers.sampler import Sampler, Sampler3
from layers.attention import AttentionLayer
from setting import logger
from utils.input_data import InputData
from model_zoo.hander import BaseHander
from model_zoo.classifer import Classifer
from model_zoo.sage import GraphSage


class HANSage(nn.Module):
    def __init__(self, num_classes, num_nodes, feat_data, feat_dim, adj_lists, adj_matrix, cuda, num_sample_tpl, num_sample_permission, embed_dim, num_layers):
        super(HANSage, self).__init__()
        self.is_cuda = cuda
        self.num_sample_tpl = num_sample_tpl
        self.num_sample_permission = num_sample_permission
        self.embed_dim = embed_dim

        adj_tpl = adj_lists['tpl']
        adj_permission = adj_lists['permission']
        mat_tpl = adj_matrix['tpl']
        mat_permission = adj_matrix['permission']

        self.encoder_tpl = GraphSage(num_classes, num_nodes, feat_data, feat_dim, adj_tpl, mat_tpl, self.is_cuda, self.num_sample_tpl, self.embed_dim, num_layers, as_view=True)
        self.encoder_permission = GraphSage(num_classes, num_nodes, feat_data, feat_dim, adj_permission, mat_permission, self.is_cuda, self.num_sample_permission, self.embed_dim, num_layers, as_view=True)

        self.atten = AttentionLayer(self.embed_dim, self.embed_dim)

        self.clf = Classifer(self.embed_dim, num_classes)
    
    def get_embedding(self, nodes):
        embed_tpl = self.encoder_tpl.get_embedding(nodes)
        embed_tpl = embed_tpl.view(embed_tpl.shape[0], 1, embed_tpl.shape[1])
        embed_permission = self.encoder_permission.get_embedding(nodes)
        embed_permission = embed_permission.view(embed_permission.shape[0], 1, embed_permission.shape[1])

        multi_embed = torch.cat((embed_tpl, embed_permission), dim=1)
        fuse_embed = self.atten(multi_embed)
        return fuse_embed
    
    def forward(self, nodes):
        fuse_embed = self.get_embedding(nodes)
        out = self.clf(fuse_embed)
        return out

    def predict(self, node, node_feat=None):
        embed_tpl = self.encoder_tpl.predict(node, node_feat)
        embed_tpl = embed_tpl.view(embed_tpl.shape[0], 1, embed_tpl.shape[1])
        embed_permission = self.encoder_permission.predict(node, node_feat)
        embed_permission = embed_permission.view(embed_permission.shape[0], 1, embed_permission.shape[1])

        multi_embed = torch.cat((embed_tpl, embed_permission), dim=1)

        fuse_embed = self.atten(multi_embed)
    
        out = self.clf(fuse_embed)
        return out.unsqueeze(0)


class HANSageHander(BaseHander):
    """
    wrapper for SupervisedGraphSage model
    """
    def __init__(self, num_class, data, args):
        self.num_class = num_class
        self.labels = data['labels']
        self.adj_lists = data['adj_lists']
        self.adj_matrix = data['adj_matrix']
        self.feat_data = data['feat_data']
        self.num_nodes, self.feat_dim = self.feat_data.shape
        self.split_seed = args.split_seed
        self.is_cuda = args.cuda
        self.view = args.view
        self.num_sample_tpl = args.num_sample_tpl
        self.num_sample_permission = args.num_sample_permission
        self.num_neighs_tpl = args.num_neighs_tpl
        self.num_neighs_permission = args.num_neighs_permission
        self.embed_dim = args.embed_dim
        self.freeze = args.freeze
        self.inputdata = InputData(self.num_nodes, self.labels, self.adj_lists, args.split_seed, args.label_rate, self.is_cuda)
        self.inst_generator = self.inputdata.gen_train_batch(batch_size=args.batch_size)
        self.train_data_loader = self.inputdata.get_train_data_load(batch_size=args.batch_size, shuffle=True)
    
    def build_model(self):
        logger.info("define model.")
        num_layers = 2
        self.model = HANSage(self.num_class, self.num_nodes, self.feat_data, self.feat_dim, self.adj_lists, self.adj_matrix, self.is_cuda, self.num_sample_tpl, self.num_sample_permission, self.embed_dim, num_layers)
        logger.info(self.model)
        if self.is_cuda:
            self.model.cuda()
        self.custom_init(self.freeze)
        self.optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, self.model.parameters()), lr=1e-3, weight_decay=1e-5)
        unbalance_alpha = torch.Tensor([0.9934, 1])
        if self.is_cuda:
            unbalance_alpha = unbalance_alpha.cuda()
        self.loss_func = nn.CrossEntropyLoss(weight=unbalance_alpha)
    
    def custom_init(self, freeze=False):
        logger.info("custom initialization. freeze={}".format(freeze))
        from setting import model_path
        import glob
        checkpoint_tpl = torch.load(glob.glob(os.path.join(model_path, 'GraphSage', "*tpl*neigh{}".format(self.num_neighs_tpl)))[0])
        tpl_state = checkpoint_tpl['state_dict']
        self.model.encoder_tpl.load_state_dict(tpl_state, strict=False)
        if freeze:
            for param in self.model.encoder_tpl.parameters():
                param.requires_grad = False

        checkpoint_permission = torch.load(glob.glob(os.path.join(model_path, 'GraphSage', "*permission*neigh{}".format(self.num_neighs_permission)))[0])
        permission_state = checkpoint_permission['state_dict']
        self.model.encoder_permission.load_state_dict(permission_state, strict=False)
        if freeze:
            for param in self.model.encoder_permission.parameters():
                param.requires_grad = False
