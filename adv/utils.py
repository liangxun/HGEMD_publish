import pickle as pkl
import torch
from sklearn import metrics
import numpy as np
import json
from model_zoo.sage import GraphSage
from model_zoo.han_sage import HANSage


# ============================= load pretrained model ===========================================
def load_model(model_path, num_class, num_nodes, feat_data, feat_dim, adj_lists, adj_matrix, cuda=False, num_sample=5, embed_dim=64, num_layers=2, as_view=False):
    """
    load GraphSage
    """
    model = GraphSage(num_class, num_nodes, feat_data, feat_dim, adj_lists, adj_matrix, cuda, num_sample, embed_dim, num_layers, as_view=False)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    if cuda is True:
        model.cuda()
    model.eval()
    return model


def load_hansage(model_path, num_class, num_nodes, feat_data, feat_dim, adj_lists, adj_matrix, cuda=False, num_sample_tpl=5, num_sample_permission=5, embed_dim=64, num_layers=2):
    """
    load HANSage 
    """
    model = HANSage(num_class, num_nodes, feat_data, feat_dim, adj_lists, adj_matrix, cuda, num_sample_tpl, num_sample_permission, embed_dim, num_layers)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    if cuda is True:
        model.cuda()
    model.eval()
    return model  

# ================================== load malwares =============================================
def load_ids(split_file, labels):
    with open(split_file, 'r') as f:
        data = json.load(f)
    ids = data['test']
    ids = list(filter(lambda n: labels[n]==1, ids)) #  only craft adversarial attack for malwares
    ids.sort()
    return ids

