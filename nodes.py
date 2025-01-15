

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from datasets import DatasetSplit
from utils import init_model, PerturbedGradientDescent
from utils import init_optimizer, model_parameter_vector
from utils import get_text_embeddings


class Node(object):

    def __init__(self,args, num_id, train_loader, val_loader, test_loader):
        self.num_id = num_id
        self.args = args
        self.node_num = self.args.node_num
        self.num_classes = args.num_classes
        self.local_data = None
        self.validate_set = None
        self.test_set = None

        if args.iid == 1 or num_id == -1:
            self.validate_set, self.test_set = val_loader, test_loader
        else:
            self.local_data = train_loader
            self.sample_per_class = self.generate_sample_per_class(self.local_data)
            
        self.model = init_model(self.args.local_model, self.args).cuda()
        self.optimizer = init_optimizer(self.num_id, self.model, args)
        
        if args.method == 'FedBM':
            if num_id == -1:
                from prompt_templates import OCT_TEMPLATES, OCT_BASIC_TEMPLATES
                from prompt_templates import Kvasir_TEMPLATES, Kvasir_BASIC_TEMPLATES
                if args.dataset == 'OCT':
                    from classnames import OCT_Concepts
                    classnames = OCT_Concepts
                elif args.dataset == 'Kvasir':
                    from classnames import Kvasir_Concepts
                    classnames = Kvasir_Concepts
                else:
                    assert False
                
                if args.dataset == 'Kvasir':
                    templates = Kvasir_TEMPLATES
                elif args.dataset == 'OCT':
                    templates = OCT_TEMPLATES
                else:
                    assert False
                all_text_features = get_text_embeddings(args, templates, classnames) # C, N, D
                all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True) #C, N, D
                all_classes_mean = []
                all_classes_cov = []
                for c in range(all_text_features.shape[0]):
                    text_features = all_text_features[c] # N D
                    text_features_mean = text_features.mean(0) #512
                    text_features_cov = torch.cov(text_features.T) #512 512
                    text_features_cov = text_features_cov.diag()
                    all_classes_mean.append(text_features_mean)
                    all_classes_cov.append(text_features_cov)
                all_classes_mean = torch.stack(tuple(all_classes_mean),dim=0) # C,512
                all_classes_cov = torch.stack(tuple(all_classes_cov),dim=0) # C,512
                self.means = all_classes_mean
                self.covs = all_classes_cov
                self.model.classifier_fedbm = all_text_features
                self.all_text_features = all_text_features

    def generate_sample_per_class(self, local_data):
        sample_per_class = torch.tensor([0 for _ in range(self.num_classes)])

        for idx, (data, target) in enumerate(local_data):
            sample_per_class += torch.tensor([sum(target==i) for i in range(self.num_classes)])

        sample_per_class = torch.where(sample_per_class > 0, sample_per_class, 1)

        return sample_per_class