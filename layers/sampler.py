import random
import numpy as np


class Sampler(object):
    """
    sample method similar to GraphSAGE
    """
    def __init__(self, adj_list, self_loop=False):
        self.adj_list = adj_list
        self.layers = []
        self.self_loop = self_loop

    def add_sample_layer(self, num_sample):
        self.layers.append(num_sample)

    def sample_layer(self, nodes, num_sample):
        """
        sample for each layer
        self_loop: regard itself as neighbor
        """
        to_neighs = [self.adj_list[int(node)] for node in nodes]
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample,)) if len(to_neigh)>num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        if self.self_loop:
            for i in range(len(nodes)):
                samp_neighs[i].add(nodes[i])
        unique_nodes_list = list(set.union(*samp_neighs) | set(nodes))
        unique_nodes = dict(list(zip(unique_nodes_list, list(range(len(unique_nodes_list))))))
        return unique_nodes_list, samp_neighs, unique_nodes

    def sample(self, nodes):
        current_nodes = nodes
        nodes_layers = [(current_nodes,)]
        for num_sample in reversed(self.layers):
            lower_nodes_list, lower_samp_neighs, lower_unique_nodes = self.sample_layer(current_nodes, num_sample)
            nodes_layers.insert(0, (lower_nodes_list, lower_samp_neighs, lower_unique_nodes))
            current_nodes = lower_nodes_list

        return nodes_layers
