import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from torch.backends import cudnn
from random import sample
import math
import torch.optim as optim
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import init_model, freeze_layers, set_params
from datasets import TensorDataset


def receive_client_models(args, client_nodes, select_list, size_weights):
    client_params = []
    local_protos_list = {}

    for idx in select_list:
        client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))

    agg_weights = [size_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]

    return agg_weights, client_params, local_protos_list


def Server_update(args, central_node, client_nodes, select_list, size_weights):
    '''
    server update functions for baselines
    '''

    # receive the local models from clients
    agg_weights, client_params, local_protos_list = receive_client_models(args, client_nodes, select_list, size_weights)

    # update the global model
    if args.server_method == 'fedavg':
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)
    else:
        raise ValueError('Undefined server method...')
        
    return central_node


def fedavg(parameters, list_nums_local_data):
    fedavg_global_params = copy.deepcopy(parameters[0])
    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        fedavg_global_params[name_param] = value_global_param
    return fedavg_global_params

