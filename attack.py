"""
train script
"""
import os
import torch
import argparse
import torch
import numpy as np
import random
import pickle as pkl
import json
import glob
from utils.load_data import load_hin
from adv.utils import load_model, load_hansage, load_ids
from adv.fgsm import FGSM
from adv.jsma import JSMA
import setting
from setting import logger, hin_path, model_path


parser = argparse.ArgumentParser(description="adversarial attack.")

parser.add_argument('alg', type=str, help="specify the attack algorithm.")
parser.add_argument('max_bit', type=int,help="the num of bits allowed to modify by attacker")
parser.add_argument('--model', type=str, default="GraphSage", help="the target model type")
parser.add_argument('--m_file', type=str, default=None, help="the pretrain target model")
parser.add_argument('--view', type=str, default="permission", help="choose whick view will be loaded. (tpl or permission or multi)")
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu. used in GAT')
parser.add_argument('--dropout', type=float, default=0.4, help="p for dropout layer")
parser.add_argument('--embed_dim', type=int, default=64, help="the dim of embedding vector")
parser.add_argument('--num_neighs', type=int, default=5, help="neighboors of each node during sample")
parser.add_argument('--num_neighs_tpl', type=int, default=None, help="neighboors of each node during sample")
parser.add_argument('--num_neighs_permission', type=int, default=None, help="neighboors of each node during sample")
parser.add_argument('--num_sample', type=int, default=None, help="used in sage_sample_layer")
parser.add_argument('--num_sample_tpl', type=int, default=None, help="used in sage_sample_layer")
parser.add_argument('--num_sample_permission', type=int, default=None, help="used in sage_sample_layer")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
if args.view == 'multi':
    args.model = 'HANSage'
    if args.num_neighs_tpl is None:
        args.num_neighs_tpl = args.num_neighs
    if args.num_neighs_permission is None:
        args.num_neighs_tpl = args.num_neighs
    
    if args.num_sample_tpl is None:
        args.num_sample_tpl = args.num_neighs_tpl
    if args.num_sample_permission is None:
        args.num_sample_permission = args.num_neighs_permission

elif args.model == "GraphSage":
    if args.num_sample is None:
        args.num_sample = args.num_neighs
else:
    raise("error")

SEED = 10000
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.set_device(setting.cuda_device_id)



def run(seed):
    # ========== load pretrained model ===============
    num_class = 2
    feat_data, labels, adj_lists, adj_matrix = load_hin(hin_path, args.view, args.num_neighs, args.num_neighs_tpl, args.num_neighs_permission, pre_max=setting.pre_max)
    num_nodes, feat_dim = feat_data.shape

    if args.m_file is not None:
        model_file = os.path.join(model_path, args.model, args.m_file)
    else:
        model_file = glob.glob(os.path.join(model_path, args.model, "*{}*neigh{}*".format(args.view, args.num_neighs)))[0]

    logger.info("mdoel_file: {}".format(model_file))
    if args.model == "HANSage":
        model = load_hansage(model_file, num_class, num_nodes, feat_data, feat_dim, adj_lists, adj_matrix, cuda=args.cuda, num_sample_tpl=args.num_sample_tpl, num_sample_permission=args.num_sample_permission, embed_dim=args.embed_dim, num_layers=2)
    elif args.model == "GraphSage":
        model  = load_model(model_file, num_class, num_nodes, feat_data, feat_dim, adj_lists, adj_matrix, cuda=args.cuda, num_sample=args.num_sample, embed_dim=args.embed_dim, num_layers=2, as_view=False)
    else:
        AssertionError("the model param is wrong!")
    logger.info(model)


    # ============= load data ===================
    ids = load_ids(setting.split_file_tmp.format(seed), labels)
    
    # ============= adversarial attack ==================
    if args.alg == 'fgsm':
        attacker = FGSM(model, args.max_bit, args.cuda)
    elif args.alg == 'jsma':
        attacker = JSMA(model, args.max_bit, args.cuda)
    else:
        AssertionError("alg is wrong!")
    
    r_codes = []
    iter = 0
    for id in ids:
        iter += 1
        init_x = feat_data[id]
        x = torch.Tensor([init_x])
        y = torch.LongTensor([1])
        r_code, _ = attacker.attack([id], x, y)
        logger.info("{}: id={}, r_code={}".format(iter, id, r_code))
        r_codes.append(r_code)
    score(r_codes, args.alg, args.max_bit, args.model, seed, model_file, save=True)


def score(r_codes, alg, max_bit, model, seed, model_file, save=False):
    r_codes = np.array(r_codes)
    rc_0 = np.sum(r_codes==0)
    rc_n = np.sum(r_codes==-1)
    rc_p = np.sum(r_codes>0)
    logger.info("rc_0={}\trc_n={}\trc_p={}".format(rc_0, rc_n, rc_p))
    recall = (rc_n + rc_p) / (rc_0 + rc_n + rc_p)
    recall_adv = rc_n / (rc_0 + rc_n + rc_p)
    foolrate = rc_p / (rc_n + rc_p)
    logger.info("recall={:.4f}\trecall_adv={:.4f}\tfoolrate={:.4f}".format(recall, recall_adv, foolrate))

    if save:
        with open("./adv/reports.csv",'a') as f:
            f.write("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\t{}\n".format(model, alg, max_bit, recall, recall_adv, foolrate, rc_0, rc_p, rc_n, model_file))


if __name__ == "__main__":
    seed = 0
    run(seed)
