from cProfile import label
import numpy as np
import torch
import torch.nn.functional as F
from utils import validate, model_parameter_vector, freeze_layers, set_params
import copy
from nodes import Node

def receive_server_model(args, client_nodes, central_node):

    for idx in range(len(client_nodes)):
        client_nodes[idx].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))

    return client_nodes

def Client_update(args, client_nodes, central_node, select_list, g_driver=None):
    '''
    client update functions
    '''
    if central_node is not None:
        client_nodes = receive_server_model(args, client_nodes, central_node)

    if args.client_method == 'local_train':
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_localTrain(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)
            
    elif 'fedbm' in args.client_method:
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedbm(args, client_nodes[i], g_driver = g_driver)
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)
    else:
        raise ValueError('Undefined client method...')

    return client_nodes, train_loss


def client_localTrain(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        
        data, target = data.cuda(), target.cuda()
        _, output_local, _ = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

    return loss/len(train_loader)

def client_fedbm(args, node, loss = 0.0, g_driver = None):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        node.optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()
        if g_driver is not None and g_driver.is_trained:
            g_driver.generator.eval()
            with torch.no_grad():
                target_ge = torch.randint(0,args.num_classes,(args.g_batchsize,)).cuda()
                _, data_ge = g_driver.generator(target_ge)
                data_ge = g_driver.aug(data_ge)
            data = torch.cat([data, data_ge])
            target = torch.cat([target, target_ge])
            _, _, feature = node.model(data)
        else:
            _, _, feature = node.model(data)
        feature = node.model.linear_fedbm(feature)
        feature_norm = feature / feature.norm(dim=-1, keepdim=True)
        
        T = args.temperature
        query_mean = feature_norm.mm(node.means.permute(1,0).float())
        covs = node.covs * T
        query_cov_query = 0.5*feature_norm.pow(2).mm(covs.permute(1,0))
        logits = query_mean + query_cov_query

        logits = logits * T
        ce_loss = F.cross_entropy(logits, target, reduction='none')
    
        key_covs = covs[target]
        jcl_loss = (0.5 * torch.sum(feature_norm.pow(2).mul(key_covs), dim=1))*T
        loss_local = ce_loss + jcl_loss
        loss_local = loss_local.mean()
        
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

    return loss/len(train_loader)
