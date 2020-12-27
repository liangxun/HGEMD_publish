"""
train script
"""
import argparse
import torch
import random
import numpy as np
import setting
from setting import logger
from utils.load_data import load_hin
from model_zoo.sage import GraphSageHandler
from model_zoo.han_sage import HANSageHander

parser = argparse.ArgumentParser(description="malware detection.")
parser.add_argument('--model',type=str, default="GraphSage", help='choose model:\n SupervisedGraphSage, UnsupervisedGraphSage, MultiGraphSage, MLP')
parser.add_argument('--view', type=str, default="permission", help="choose whick view will be loaded. (tpl or permission or multi)")
parser.add_argument('--split_seed', type=int, default=0, help="choose presaved split_file")
parser.add_argument('--label_rate', type=float, default=1, help="the rate of labeled samples in train set")

parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu. used in GAT')
parser.add_argument('--dropout', type=float, default=0.4, help="p for dropout layer")
parser.add_argument('--embed_dim', type=int, default=64, help="the dim of embedding vector")
parser.add_argument('--num_neighs', type=int, default=5, help="neighboors of each node during sample")
parser.add_argument('--num_neighs_tpl', type=int, default=None, help="neighboors of each node during sample")
parser.add_argument('--num_neighs_permission', type=int, default=None, help="neighboors of each node during sample")
parser.add_argument('--num_sample', type=int, default=None, help="used in sage_sample_layer")
parser.add_argument('--num_sample_tpl', type=int, default=None, help="used in sage_sample_layer")
parser.add_argument('--num_sample_permission', type=int, default=None, help="used in sage_sample_layer")

parser.add_argument('--epoches', type=int, default=5, help="train epoches")
parser.add_argument('--interval_eval', type=int, default=10, help="evaluation interval")
parser.add_argument('--batch_size', type=int, default=128, help="minibatch")
parser.add_argument('--seed', type=int, default=0, help="seed for random")
parser.add_argument('--freeze', type=bool, default=False, help="used in hansage to load pretrain sage")


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
if args.view == 'multi':
    args.model = 'HANSage'
    if args.num_neighs_tpl is None:
        args.num_neighs_tpl = args.num_neighs
    if args.num_neighs_permission is None:
        args.num_neighs_permission = args.num_neighs
    
    if args.num_sample_tpl is None:
        args.num_sample_tpl = args.num_neighs_tpl
    if args.num_sample_permission is None:
        args.num_sample_permission = args.num_neighs_permission
else:
    if args.num_sample is None:
        args.num_sample = args.num_neighs


SEED = 10000
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.set_device(setting.cuda_device_id)


def run(data_path, model_path, embed_path, report_file):

    logger.info("loading dataset.")
    num_class = 2
    feat_data, labels, adj_lists, adj_matrix = load_hin(data_path, args.view, args.num_neighs, args.num_neighs_tpl, args.num_neighs_permission, pre_max=setting.pre_max)
    data = dict()
    data['feat_data'] = feat_data
    data['labels'] = labels
    data['adj_lists'] = adj_lists
    data['adj_matrix'] = adj_matrix

    if args.model == "GraphSage":
        model = GraphSageHandler(num_class, data, args)
    elif args.model == "HANSage":
        model = HANSageHander(num_class, data, args)
    elif args.model == "GAT":
        model = GATHandler(num_class, data, args)
    else:
        raise("error")

    model.train(epoch=args.epoches, interval_val=args.interval_eval)

    model.save_mode(model_path, report_file)
    # model.get_embedding(embed_path)
    

if __name__ == '__main__':
    from setting import hin_path, embed_path, report_file, model_path
    run(data_path=hin_path, model_path=model_path, embed_path=embed_path, report_file=report_file)
