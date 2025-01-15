
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import random
from torch.backends import cudnn
import math
from pyhessian import hessian
from torch.optim import Optimizer
from models_dict import densenet, resnet, cnn
import copy

##############################################################################
# Tools
##############################################################################

def set_server_method(args):
    '''
    FedAvg: {'server_method': 'fedavg', 'client_method': 'local_train'}.
    FedBM: {'server_method': 'fedavg', 'client_method': 'fedbm'}.
    '''
    
    if args.method == 'FedAvg':
        args.client_method = 'local_train'
        args.server_method = 'fedavg'
    elif args.method == 'FedBM':
        args.client_method = 'fedbm'
        args.server_method = 'fedavg'
    else:
        assert False

    return args


class Model(nn.Module):
    """For classification problem"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_params(self):
        return self.state_dict()

    def get_gradients(self, dataloader):
        raise NotImplementedError


def set_params(model, model_state_dict, exclude_keys=set()):
    """
        Reference: Be careful with the state_dict[key].
        https://discuss.pytorch.org/t/how-to-copy-a-modified-state-dict-into-a-models-state-dict/64828/4.
    """
    with torch.no_grad():
        for key in model_state_dict.keys():
            if key not in exclude_keys:
                model.state_dict()[key].copy_(model_state_dict[key])
    return model

def freeze_layers(model, layers_to_freeze):
    for name, p in model.named_parameters():
        try:
            if name in layers_to_freeze:
                p.requires_grad = False
            else:
                p.requires_grad = True
        except:
            pass
    return model

class ModelWrapper(Model):
    def __init__(self, base, head, config):
        """
            head and base should be nn.module
        """
        super(ModelWrapper, self).__init__(config)

        self.base = base
        self.head = head

    def forward(self, x, return_embedding):
        feature_embedding = self.base(x)
        out = self.head(feature_embedding)
        if return_embedding:
            return feature_embedding, out
        else:
            return out


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)

def softmax_fuct(lrs):
    '''
    lrs is dict as {0:3, 1:3, 2:4}
    '''
    exp_cache = []
    softmax_lrs = {}
    for i in range(len(lrs)):
        exp_cache.append(math.exp(lrs[i]))
    
    for i in range(len(lrs)):
        softmax_lrs[i] = exp_cache[i]/sum(exp_cache)
    
    return softmax_lrs

def cos(x, y):
    fuct = nn.CosineSimilarity(dim=0)
    result = fuct(x, y)
    result = result.detach().cpu().numpy().tolist()
    return result

def get_cosGrad_matrix(gradients):
    client_num = len(gradients)
    matrix = [[0.0 for _ in range(client_num)] for _ in range(client_num)]

    for i in range(client_num):
        for j in range(client_num):
            if matrix[j][i] != 0.0:
                matrix[i][j] = matrix[j][i]
            else:
                matrix[i][j] = cos(gradients[i], gradients[j])
    
    return matrix

def model_parameter_vector(args, model):
    param = [p.view(-1) for p in model.parameters()]
    # vector = torch.concat(param, dim=0)
    vector = torch.cat(param, dim=0)
    return vector

##############################################################################
# Initialization function
##############################################################################

def init_model(model_type, args):
    
    num_classes = args.num_classes
    if model_type == 'CNN':
        if args.dataset == 'cifar10':
            model = cnn.CNNCifar10()
        else:
            model = cnn.CNNCifar100()
    elif model_type == 'ResNet18':
        model = resnet.ResNet18(num_classes)
    elif model_type == 'ResNet20':
        model = resnet.ResNet20(num_classes)
    elif model_type == 'ResNet56':
        model = resnet.ResNet56(num_classes)
    elif model_type == 'ResNet110':
        model = resnet.ResNet110(num_classes)
    elif model_type == 'WRN56_2':
        model = resnet.WRN56_2(num_classes)
    elif model_type == 'WRN56_4':
        model = resnet.WRN56_4(num_classes)
    elif model_type == 'WRN56_8':
        model = resnet.WRN56_8(num_classes)
    elif model_type == 'DenseNet121':
        model = densenet.DenseNet121(num_classes)
    elif model_type == 'DenseNet169':
        model = densenet.DenseNet169(num_classes)
    elif model_type == 'DenseNet201':
        model = densenet.DenseNet201(num_classes)
    elif model_type == 'MLP':
        model = cnn.MLP()
    elif model_type == 'LeNet5':
        model = cnn.LeNet5() 

    return model

def init_optimizer(num_id, model, args):

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)

    return optimizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

##############################################################################
# Training function
##############################################################################

def generate_matchlist(client_node, ratio = 0.5):
    candidate_list = [i for i in range(len(client_node))]
    select_num = int(ratio * len(client_node))
    match_list = np.random.choice(candidate_list, select_num, replace = False).tolist()
    return match_list

def lr_scheduler(rounds, node_list, args):
    # learning rate scheduler for decaying
    if rounds != 0:
        args.lr *= 0.99 #0.99
        for i in range(len(node_list)):
            node_list[i].args.lr = args.lr
            node_list[i].optimizer.param_groups[0]['lr'] = args.lr
    print('Learning rate={:.4f}'.format(args.lr))
    

class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)

        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                # g = g.cuda()
                if p.grad != None:
                    d_p = p.grad.data + group['mu'] * (p.data - g.data)
                    p.data.add_(d_p, alpha=-group['lr'])

##############################################################################
# Validation function
##############################################################################
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def compute_metrics(pre, gt): #D, H, W
    pred = pre.cpu().numpy()
    gt = gt.cpu().numpy()
    acc=accuracy_score(gt, pred)
    recall=recall_score(gt, pred, average='micro')
    prec = precision_score(gt, pred, average='macro')
    f1 = f1_score(gt, pred, average='macro')

    return acc, recall, prec, f1

def compute_auc(pre_scores, gt, num_classes = 8):

    pre_scores = pre_scores.cpu().numpy()
    gt = gt.cpu().numpy()
    
    gt_one_hot = np.eye(num_classes)[gt]
    auc_score = roc_auc_score(gt_one_hot, pre_scores)
    
    return auc_score
    

def validate(args, node, which_dataset = 'validate'):
    '''
    Generally, 'validate' refers to the local datasets of clients and 'local' refers to the server's testset.
    '''
    node.model.cuda().eval() 
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    elif which_dataset == 'test':
        test_loader = node.test_set
    else:
        raise ValueError('Undefined...')

    with torch.no_grad():
        preds = []
        targets = []
        pred_scores = []
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            feature, logit, out = node.model(data)
            if 'fedbm' in args.client_method:
                out = node.model.linear_fedbm(out)
                out_norm = out / out.norm(dim=-1, keepdim=True)
                query_mean = out_norm.mm(node.means.permute(1,0).float()) #N*K
                T = args.temperature
                #T = 0.07
                covs = node.covs * T
                query_cov_query = 0.5*out_norm.pow(2).mm(covs.permute(1,0))
                output = query_mean + query_cov_query
            else:
                output = logit

            pred = output.argmax(dim=1)
            pred_scores.append(output.softmax(dim=1)) # B, C
            preds.append(pred)
            targets.append(target.view_as(pred))

        pred_scores = torch.cat(pred_scores)
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        acc, recall, prec, f1 = compute_metrics(preds, targets)
        auc = compute_auc(pred_scores, targets, num_classes=args.num_classes)
        
    return acc*100, recall*100, prec*100, f1*100, auc*100



def get_text_embeddings(args, templates, classnames):

    if args.text_encoder=='CLIP': 
        '''
        ViT-L/14 768
        ViT-B/16 512
        ViT-B/32 512
        RN50 1024
        RN101 512
        RN50x4 640
        RN50x16 768
        RN50x64 1024
        ''' 
        import clip
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        clip_model.cuda()
        for params in clip_model.parameters():
            params.requires_grad_(False)
        num_temp = len(templates)
        all_text_features = []
        for c in classnames:
            prompts = [temp.format(c.replace("_", " ")) for temp in templates]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            if args.is_averge_text_embed:
                text_features = text_features.mean(0, keepdim=True)
            all_text_features.append(text_features)
        all_text_features = torch.stack(tuple(all_text_features),dim=0)#C, N, 512
        print(f"Prompt ensembling (n={num_temp}, size={all_text_features.size()})")
        all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)
        del clip_model

    elif args.text_encoder=='BiomedCLIP':
        from open_clip import create_model_from_pretrained, get_tokenizer
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        num_temp = len(templates)
        all_text_features = []
        for c in classnames:
            prompts = [temp.format(c.replace("_", " ")) for temp in templates]
            prompts = torch.cat([tokenizer(p) for p in prompts])
            text_features = model.encode_text(prompts)
            all_text_features.append(text_features)
        all_text_features = torch.stack(tuple(all_text_features),dim=0).cuda()#C, N, 512
        print(f"Prompt ensembling (n={num_temp}, size={all_text_features.size()})")
        del model
        del tokenizer
    elif args.text_encoder=='BERT':
        from transformers import BertModel, BertTokenizer
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        num_temp = len(templates)
        all_text_features = []
        for c in classnames:
            prompts = [temp.format(c.replace("_", " ")) for temp in templates]
            tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
            output = model(**tokens)
            last_hidden_state = output.last_hidden_state # N, C_o, 768
            text_features = last_hidden_state.mean(dim=1).squeeze() # N, 768
            all_text_features.append(text_features)
        all_text_features = torch.stack(tuple(all_text_features),dim=0).cuda()#C, N, 512
        print(f"Prompt ensembling (n={num_temp}, size={all_text_features.size()})")
        del model
        del tokenizer
    elif args.text_encoder=='RoBERTa':
        from transformers import AutoTokenizer, RobertaModel
        model_name = 'FacebookAI/roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name)
        num_temp = len(templates)
        all_text_features = []
        for c in classnames:
            prompts = [temp.format(c.replace("_", " ")) for temp in templates]
            tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
            output = model(**tokens)
            last_hidden_state = output.last_hidden_state # N, C_o, 768
            text_features = last_hidden_state.mean(dim=1).squeeze() # N, 768
            all_text_features.append(text_features)
        all_text_features = torch.stack(tuple(all_text_features),dim=0).cuda()#C, N, 512
        print(f"Prompt ensembling (n={num_temp}, size={all_text_features.size()})")
        del model
        del tokenizer
    else:
        assert False
    
    return all_text_features
