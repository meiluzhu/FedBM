
import time
import torch
import numpy as np
import os
import copy
import gc
import pprint
import argparse
import warnings
from datasets import Data
from nodes import Node
from server_funct import Server_update
from client_funct import Client_update
from generator_driver import Generator_Driver
from utils import setup_seed, set_server_method, lr_scheduler, validate

warnings.filterwarnings('ignore')
np.set_printoptions(precision=7, suppress=True)

def generate_matchlist(node_num, ratio = 0.5):
    candidate_list = [i for i in range(node_num)]
    select_num = int(ratio * node_num)
    match_list = np.random.choice(candidate_list, select_num, replace = False).tolist()
    return match_list

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--iid', type=int, default=0,
                        help='set 1 for iid, and 0 for noniid (dir. sampling)')
    parser.add_argument('--batchsize', type=int, default=128, 
                        help="batchsize")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, 
                    help="dirichlet_alpha")
    parser.add_argument('--num_classes', type=int, default=8, 
                        help="num_classes")
    
    # System
    parser.add_argument('--device', type=str, default='0',
                        help="cuda device: {cuda, cpu}")
    parser.add_argument('--node_num', type=int, default=20, 
                        help="Number of nodes") 
    parser.add_argument('--T', type=int, default=200, 
                        help="Number of communication rounds")
    parser.add_argument('--E', type=int, default=3, 
                        help="Number of local epochs: E")
    parser.add_argument('--dataset', type=str, default='OCT',
                        help="Type of dataset") 
    parser.add_argument('--data_path', type=str, default='./',
                        help="data_path") 
    parser.add_argument('--select_ratio', type=float, default=1.0,
                    help="the ratio of client selection in each round")
    parser.add_argument('--local_model', type=str, default='CNN',
                        help='Type of local model: {CNN, ResNet20, ResNet18}')
    parser.add_argument('--exp_name', type=str, default='FirstTable',
                        help="experiment name")

    # Server function
    parser.add_argument('--server_method', type=str, default='fedavg',
                        help="FedAvg, or others")
    # Client function
    parser.add_argument('--client_method', type=str, default='local_train',
                        help="client method")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="optimizer: {sgd, adam}")
    parser.add_argument('--lr', type=float, default=0.04,  
                        help='learning rate')
    parser.add_argument('--local_wd_rate', type=float, default=5e-4,
                        help='clients local wd rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')

    parser.add_argument('--method', type=str, default='FedAvg',
                        help="method")
    parser.add_argument('--text_encoder', type=str, default='BiomedCLIP',
                        help="Text_encoder") #CLIP, BiomedCLIP

    parser.add_argument('--temperature', type=float, default=1.,
                        help="scaling") #CLIP, BiomedCLIP
    
    parser.add_argument('--g_iter', type=int, default=100,
                        help="iterations")
    parser.add_argument('--lr_g', type=float, default=0.01,
                        help="lr_g")
    parser.add_argument('--synthesis_batch_size', type=int, default=128,
                        help="synthesis_batch_size")
    parser.add_argument('--freq_g_driver', type=int, default=5,
                        help="freq_g_driver")
    parser.add_argument('--g_batchsize', type=int, default=8,
                        help="g_batchsize")
    parser.add_argument('--lambda_div', type=float, default=1.0,
                        help="diversity_weight")
    parser.add_argument('--lambda_dis', type=float, default=1,
                        help="bn_weight")
    
    args = parser.parse_args()
    
    #Ensure that each client has samples
    if args.dataset == 'Kvasir': random_seeds = [1, 3, 6]
    if args.dataset == 'OCT': random_seeds = [0, 1, 2]

    lr = args.lr
    all_acc, all_recall, all_prec, all_f1, all_auc = [],[],[],[],[]
    for random_seed in random_seeds:
        gc.collect()
        torch.cuda.empty_cache()
        args.random_seed = random_seed
        args.lr = lr
        print('starting run seed', args.random_seed)
        setup_seed(random_seed)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print('The starting time ：{}'.format(now), flush=True)
        args = set_server_method(args)
        pprint(vars(args))
    
        if args.select_ratio == 1.0:
            select_list_recorder = [[i for i in range(args.node_num)] for _ in range(args.T)]
        else:
            select_list_recorder = [generate_matchlist(args.node_num, args.select_ratio) for _ in range(args.T)]
  
        setting_name =  args.exp_name + '_' + args.dataset + '_' + args.local_model + '_nodenum' + str(args.node_num) + '_dir' + str(args.dirichlet_alpha) +'_E'+ str(args.E)  + '_C' + str(args.select_ratio) \
        + '_' + args.server_method + '_' + args.client_method + '_seed' + str(args.random_seed)
    
        root_path = './'
        output_path = 'results/'
        if not os.path.exists(os.path.join(root_path, output_path)):
            os.makedirs(os.path.join(root_path, output_path))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        data = Data(args)
        sample_size = []
        for i in range(args.node_num): 
            sample_size.append(len(data.train_loader[i]))
        size_weights = [i/sum(sample_size) for i in sample_size]
        print('size-based weights',size_weights, flush=True)

        central_node = Node(args, -1, train_loader = None, val_loader=data.val_loader, test_loader=data.test_loader)
        # initialize the client nodes
        client_nodes = {}
        for i in range(args.node_num): 
            client_nodes[i] = Node(args, i, train_loader=data.train_loaders[i], val_loader=None, test_loader=None) 
            client_nodes[i].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
            if args.method == 'FedBM': 
                client_nodes[i].model.classifier_fedbm = copy.deepcopy(central_node.model.classifier_fedbm.data)
                client_nodes[i].means = copy.deepcopy(central_node.means.data)
                client_nodes[i].covs = copy.deepcopy(central_node.covs.data)
    
        best_val_acc = 0
        best_test_acc = 0
        best_test_recall=0
        best_test_prec=0
        best_test_f1=0
        best_test_auc=0
        print(setting_name, flush=True)
        g_driver = None
        if args.method == 'FedBM':
            g_driver = Generator_Driver(num_classes=args.num_classes, iterations=args.g_iter,
                                         lr_g=args.lr_g,
                                         text_features=central_node.all_text_features,
                                         synthesis_batch_size=args.synthesis_batch_size,
                                         means=central_node.means,
                                         covs=central_node.covs,
                                         args=args)
        for rounds in range(0, args.T):
            
            print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1), flush=True)
            lr_scheduler(rounds, client_nodes, args)

            # Client selection
            select_list = select_list_recorder[rounds]
    
            # Local update
            client_nodes, train_loss = Client_update(args, client_nodes, central_node, select_list, g_driver=g_driver)
            print(args.server_method + args.client_method + ', train loss is {:.5f}'.format(train_loss), flush=True)
            
            # Server aggregation
            central_node = Server_update(args, central_node, client_nodes, select_list, size_weights)
            val_acc, val_recall, val_prec, val_f1, val_auc  = validate(args, central_node, which_dataset = 'validate')
            print(args.server_method + args.client_method + ', Val acc: {:.3f}'.format(val_acc)+ ', recall: {:.3f}'.format(val_recall)+ ', prec: {:.3f}'.format(val_prec)+ ', f1: {:.3f}'.format(val_f1)+ ', auc: {:.3f}'.format(val_auc), flush=True) 
            print(args.server_method + args.client_method + ', Test acc: {:.3f}'.format(best_test_acc)+ ', recall: {:.3f}'.format(best_test_recall)+ ', prec: {:.3f}'.format(best_test_prec)+ ', f1: {:.3f}'.format(best_test_f1)+ ', auc: {:.3f}'.format(best_test_auc), flush=True)
            print()
            if val_acc+val_recall+val_prec+val_f1+val_auc>best_val_acc:
                best_val_acc = val_acc+val_recall+val_prec+val_f1+val_auc
                best_test_acc,best_test_recall, best_test_prec, best_test_f1,best_test_auc = validate(args, central_node, which_dataset = 'test')
                print(args.server_method + args.client_method + ', Test acc: {:.3f}'.format(best_test_acc)+ ', recall: {:.3f}'.format(best_test_recall)+ ', prec: {:.3f}'.format(best_test_prec)+ ', f1: {:.3f}'.format(best_test_f1)+ ', auc: {:.3f}'.format(best_test_auc), flush=True)
                print()
                
                torch.save(central_node.model.state_dict(), os.path.join(root_path, output_path, setting_name+'_finalmodel.pth'))
                if args.method == 'FedBM':
                    torch.save(central_node.model.classifier_fedbm, os.path.join(root_path, output_path, setting_name+'_classifier_fedbm.pth'))
                    torch.save(central_node.means, os.path.join(root_path, output_path, setting_name+'_means.pth'))
                    torch.save(central_node.covs, os.path.join(root_path, output_path, setting_name+'_covs.pth'))
            
            #training generator
            if args.method == 'FedBM' and (rounds+1)%args.freq_g_driver==0:
                g_driver.train(central_node)
            
        all_acc.append(best_test_acc)
        all_recall.append(best_test_recall)
        all_prec.append(best_test_prec)
        all_f1.append(best_test_f1)
        all_auc.append(best_test_auc)
        end = time.strftime("%Y-%m-%d %H:%M:%S")
        print('The ending time ：{}'.format(end))
        
    print('===========================================================')
    print('Best test acc:', all_acc)
    print('Best test acc mean: {:.5f}'.format(np.mean(all_acc)),'Best test acc std: {:.5f}'.format(np.std(all_acc)) )

    print('Best test recall:', all_recall)
    print('Best test recall mean: {:.5f}'.format(np.mean(all_recall)),'Best test recall std: {:.5f}'.format(np.std(all_recall)) )

    print('Best test prec:', all_prec)
    print('Best test prec mean: {:.5f}'.format(np.mean(all_prec)),'Best test prec std: {:.5f}'.format(np.std(all_prec)) )

    print('Best test f1:', all_f1)
    print('Best test f1 mean: {:.5f}'.format(np.mean(all_f1)),'Best test f1 std: {:.5f}'.format(np.std(all_f1)) )

    print('Best test auc:', all_auc)
    print('Best test auc mean: {:.5f}'.format(np.mean(all_auc)),'Best test auc std: {:.5f}'.format(np.std(all_auc)) )
    print('===========================================================')
    
    
    