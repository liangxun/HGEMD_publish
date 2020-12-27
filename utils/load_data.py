import os
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from collections import defaultdict
import setting


def view2adj_list(view):
    adj_lists = defaultdict(set)
    for i, col in enumerate(view):
        col_np = col.toarray().squeeze()
        if np.sum(col_np) > 0:
            adj_lists[i] = set(np.where(col_np>0)[0])
        else:
            adj_lists[i].add(i)
    return adj_lists


def normalize(feats):
    rowsum = np.array(np.sum(feats, axis=1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    feats = r_mat_inv.dot(feats)
    return feats


def load_view(hin_path, view_flag, neighs, pre_max):
    with open(os.path.join(hin_path, 'mat_view_{}.{}_{}.pkl'.format(view_flag, pre_max, neighs)), 'rb') as f:
        mat = pkl.load(f)
    if not os.path.exists(os.path.join(hin_path, 'adj_view_{}.{}_{}.pkl'.format(view_flag, pre_max, neighs))):
        adj = view2adj_list(mat)
        with open(os.path.join(hin_path, 'adj_view_{}.{}_{}.pkl'.format(view_flag, pre_max, neighs)), 'wb') as f:
            pkl.dump(adj, f)
    else:
        with open(os.path.join(hin_path, 'adj_view_{}.{}_{}.pkl'.format(view_flag, pre_max, neighs)), 'rb') as f:
            adj = pkl.load(f)
    return mat, adj


def load_hin(hin_path, view_flag, neighs=None, neighs_tpl=None, neighs_permission=None, pre_max=25):
    # get lables
    with open(os.path.join(hin_path, 'label.pkl'), 'rb') as f:
        labels = pkl.load(f)

    # get node's features
    with open(os.path.join(hin_path, 'feats_{}.pkl'.format(setting.feat_dim)), 'rb') as f:
        feats = pkl.load(f)

    # get adj_list and adj_matrix
    if view_flag == "multi":
        adj_list = dict()
        adj_matrix = dict()
        adj_matrix['tpl'], adj_list['tpl'] = load_view(hin_path, 'tpl', neighs if neighs_tpl is None else neighs_tpl, pre_max)
        adj_matrix['permission'], adj_list['permission'] = load_view(hin_path, 'permission', neighs if neighs_permission is None else neighs_permission, pre_max)
    else: # 'tpl' or 'permission'
        assert neighs is not None
        adj_matrix, adj_list = load_view(hin_path, view_flag, neighs, pre_max)

    return feats.toarray(), labels, adj_list, adj_matrix


if __name__ == '__main__':
    hin_path = "../data"
    feats, labels, adj_list, adj_matrix = load_hin(hin_path, "multi", neighs_tpl=5, neighs_permission=5)
    print(feats.shape)
