# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 20:59:12 2024

@author: ZML
"""
import torch
from torchvision import transforms
import torch.nn.functional as F
from models_dict.generator import Generator
import copy
from utils import init_model
import torch.nn as nn
import gc

def custom_cross_entropy(preds, target):
    return torch.mean(torch.sum(-target * preds.log_softmax(dim=-1), dim=-1))

def dv_loss(codes, samples, target, class_num=8):
    
    if len(codes.shape)!=2:
        codes = codes.view(codes.shape[0], -1)
    if len(samples.shape)!=2:
        samples = samples.view(samples.shape[0], -1)
        
    ds_loss = 0
    for c in range(class_num):
        if (target==c).sum()>1:
            codes_c = codes[target==c]
            samples_c = samples[target==c]
            samples_c = samples_c.view(samples_c.shape[0],-1)
            code_pairwise_distance = F.pdist(codes_c, p =1)
            sample_pairwise_distance = F.pdist(samples_c, p =1)
            ds_loss_c = sample_pairwise_distance / code_pairwise_distance
            eps = 1e-5
            ds_loss = ds_loss + torch.mean(1/(ds_loss_c+eps))
    ds_loss = ds_loss / class_num
    
    return ds_loss



class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):  # hook_fn(module, input, output) -> None
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def remove(self):
        self.hook.remove()



class Generator_Driver():
    def __init__(self, num_classes=8, 
                 img_size=128, 
                 iterations=100, 
                 lr_g=0.01, 
                 text_features=None,
                 synthesis_batch_size=128, 
                 means = None, 
                 covs= None,
                 args=None):
        super(Generator_Driver, self).__init__()
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.args = args
        self.text_features = text_features

        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.generator = Generator(text_features = text_features).cuda()
    
        self.aug = transforms.Compose([
            transforms.RandomCrop(128, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]) 

        self.discriminator = init_model(self.args.local_model, self.args).cuda()
        self.means = means.detach().clone()
        self.means.requires_grad = False
        self.covs = covs.detach().clone()
        self.covs.requires_grad = False
        self.is_trained = False

    def train(self, node):
        
        self.discriminator.load_state_dict(copy.deepcopy(node.model.state_dict()))
        self.discriminator.eval()
        if self.args.lambda_dis>0:
            hooks = []
            for m in self.discriminator.modules():
                if isinstance(m, nn.BatchNorm2d):
                    hooks.append(DeepInversionHook(m))
        
        self.generator.train()
        optimizer = torch.optim.Adam([
            {'params': self.generator.parameters()},
        ], lr=self.lr_g, betas=[0.5, 0.999])
        
        loss = 0
        acc = 0
        
        sem_loss = 0 
        div_loss = 0
        bn_loss = 0
        for it in range(1, self.iterations+1):
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            targets = torch.randint(0,self.num_classes,(self.synthesis_batch_size,)).cuda().detach()
            codes, inputs = self.generator(targets=targets)
            inputs_aug = self.aug(inputs)
            _, _, feature = self.discriminator(inputs_aug)
            feature = self.discriminator.linear_fedbm(feature)
            feature_norm = feature / feature.norm(dim=-1, keepdim=True) # B, 512
            
            T = self.args.temperature 
            query_mean = feature_norm.mm(self.means.permute(1,0).float()) #N*K
            covs = self.covs * T
            query_cov_query = 0.5*feature_norm.pow(2).mm(covs.permute(1,0))
            logits = query_mean + query_cov_query
    
            logits = logits * T
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            
            loss_local = 0
            key_covs = covs[targets]
            jcl_loss = (0.5 * torch.sum(feature_norm.pow(2).mul(key_covs), dim=1))*T
            semanticity_loss = ce_loss + jcl_loss
            semanticity_loss = semanticity_loss.mean() 
            loss_local = loss_local + semanticity_loss
            sem_loss = sem_loss + semanticity_loss.item()
            
            if self.args.lambda_dis>0:
                stability_loss = sum([h.r_feature for h in hooks])
                loss_local = loss_local + self.args.lambda_dis * stability_loss
                bn_loss = bn_loss + stability_loss.item()
                
            if self.args.lambda_div>0:
                diversity_loss = dv_loss(codes, inputs, targets, self.num_classes)
                loss_local = loss_local + self.args.lambda_div * diversity_loss
                div_loss = div_loss + diversity_loss.item()
            
            loss_local.backward()
            loss = loss + loss_local.item()
            optimizer.step()
            del loss_local
            if self.args.lambda_div>0:
                del diversity_loss
            if self.args.lambda_dis>0:
                del stability_loss

            preds = logits.argmax(dim=1)
            correct = preds.eq(targets.view_as(preds)).sum().item()
            acc_local = correct / self.synthesis_batch_size
            acc = acc + acc_local
            
            g_print_freq = 10
            if it % g_print_freq==0:
                loss = loss/g_print_freq
                sem_loss = sem_loss/g_print_freq
                div_loss = div_loss/g_print_freq
                bn_loss = bn_loss/g_print_freq
                acc = acc/g_print_freq
                print('Train generator'+', iteration-{:d} train loss:{:.5f}, sem loss:{:.5f}, div loss:{:.5f}, bn loss:{:.5f}, acc:{:.5f}'.format(it, loss, sem_loss, div_loss, bn_loss, acc*100), flush=True)
                loss = 0
                acc = 0
                sem_loss = 0
                div_loss = 0
                bn_loss = 0
        print()
        self.is_trained = True
        self.generator.eval()
            