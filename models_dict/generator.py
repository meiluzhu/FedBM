# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:16:31 2024

@author: ZML
"""

import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, ngf=64, img_size=128, nc=3, nl=100, text_features=None, le_emb_size=256, sbz=200):
        super(Generator, self).__init__()
        self.params = (ngf, img_size, nc, nl, text_features, le_emb_size, sbz)
        self.le_emb_size = le_emb_size
        self.embs_dists = self.construct_text_embs_dists(text_features) #text_features, C, N, D
        self.init_size = img_size // 16
        self.le_size = text_features.shape[-1]
        self.nl = nl
        self.nle = int(np.ceil(sbz/nl))
        self.sbz = sbz
        self.num_classes = text_features.shape[0]

        #self.n1 = nn.BatchNorm1d(self.le_size)
        #self.sig1 = nn.Sigmoid()
        #self.le1 = nn.ModuleList([nn.Linear(self.le_size, le_emb_size) for i in range(self.nle)])
        self.l1 = nn.Sequential(nn.Linear(self.le_size, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(2*ngf, 2*ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, 2*ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, 2*ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 128

            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)
            
    
    def construct_text_embs_dists(self, text_features):
        
        embs_dists=[]
        for c in range(text_features.shape[0]):
            g_mean = text_features[c].mean(dim = 0)
            g_cov = torch.cov(text_features[c].T)
            g_cov += 0.0001 * torch.eye(g_cov.shape[1]).cuda()
            class_embs_dist=torch.distributions.multivariate_normal.MultivariateNormal(g_mean, covariance_matrix=g_cov)
            embs_dists.append(class_embs_dist)
            
        return embs_dists
    
    def sampling_embs(self, targets):
        
        num = targets.shape[0]
        samples = torch.zeros(num, self.le_size).cuda()
        for c in range(self.num_classes):
            cnum = (targets == c).sum()
            if cnum>0:
                samples[targets == c] = self.embs_dists[c].rsample((cnum,))
            
        return samples

    def forward(self, targets=None):

        le = self.sampling_embs(targets).detach()
        '''
        # le = self.sig1(le)
        le = self.n1(le)
        v = None
        for i in range(self.nle):
            if (i+1)*self.nl > le.shape[0]:
                sle = le[i*self.nl:]
            else:
                sle = le[i*self.nl:(i+1)*self.nl]
            sv = self.le1[i](sle)
            if v is None:
                v = sv
            else:
                v = torch.cat((v, sv))
        '''
        out = self.l1(le)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return le, img

    def reinit(self):
        return Generator(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
                               self.params[5], self.params[6], self.params[7]).cuda()
        


if __name__=='__main__':

    data = torch.randn(100, 128)
    g_mean = data.mean(dim = 0)
    g_cov = torch.cov(data.T)
    eye_matrix = torch.eye(g_cov.shape[1])
    g_cov += 0.0001 * eye_matrix
    
    y = torch.randint(0,3, (20,))
    
    unknown_dis=torch.distributions.multivariate_normal.MultivariateNormal(g_mean, covariance_matrix=g_cov)        
    generated_unknown_samples = unknown_dis.rsample(((y==10).sum(),))
            
        
    
    cr = 0
    synthesis_batch_size = 20
    num_classes = 8
    s = synthesis_batch_size // num_classes
    v = synthesis_batch_size % num_classes
    target = torch.randint(num_classes, (v,))

    for i in range(s):
        tmp_label = torch.tensor(range(0, num_classes))
        target = torch.cat((tmp_label, target))

    ys = torch.zeros(synthesis_batch_size, num_classes)
    ys.fill_(cr / (num_classes - 1))
    ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))


    text_features = torch.randn(8,30, 512)
    generator = Generator(text_features = text_features)
    
    x = generator.forward(target)





