import random
import time
import warnings
import sys
import argparse
import copy
import numpy as np
import os
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from dalib.adaptation.sfda import mut_info_loss


import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from compute_features import compute_features
import torch.nn as nn

class DistillationLoss:
    def __init__(self):
        self.student_loss = nn.CrossEntropyLoss()
        self.distillation_loss = nn.KLDivLoss()
        self.temperature = 1
        self.alpha = 0.0

    def __call__(self, student_logits, teacher_logits):
        distillation_loss = self.distillation_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                                   F.softmax(teacher_logits / self.temperature, dim=1))

        # loss = (1 - self.alpha) * student_target_loss + self.alpha * distillation_loss
        return distillation_loss
    
def get_features_anchor(model, anchore_loader, device):
    model.eval()
    features = []
    targets = []
    preds = []
    first_run = True
    t_features = torch.zeros((model.fc.out_features, model.fc.in_features)).to(device)
    num_each = torch.zeros((model.fc.out_features))
    for data, target in anchore_loader:
        data = data.to(device).float()
        target = target.to(device).long()

        y_t, f_t = model(data)
        
        pred = y_t.data.max(1)[1]


        for i in range(len(t_features)):
            indices = torch.where(pred==i)[0]
            if len(indices) != 0:
                num_each[i] += len(indices)
                t_features[i] += torch.sum(
                    f_t[indices],dim=0)
    
    for i in range(len(t_features)):
        if num_each[i] != 0:
            t_features[i] = t_features[i]/num_each[i] 
    t_centroids = t_features.detach()
    t_centroids_norm = t_centroids / (t_centroids.norm(dim=1)[:, None]+1e-10)
    
    return t_centroids_norm

# def contrastive_loss(anchor_data_loader,model,inputs,targets,device):
def contrastive_loss(t_centroids,s_centroids,f_t,outputs,f_s,outputs_s,model,device,exist):
    ro = 0.99
    temprature = 0.1
    t_features = torch.zeros((model.head.out_features, model.head.in_features)).to(device)
    
    # print(outputs)
    for i in range(len(t_features)):
        if exist[i]:
            indices = torch.where(outputs==i)[0]
            if len(indices) != 0:
                t_features[i] = torch.sum(f_t[indices],dim=0)/len(indices)
    # print(s_features[0])

    t_centroids = (1-ro)*t_centroids.detach() + ro*t_features
    # print(s_centroids)
    
    s_features = torch.zeros((model.head.out_features, model.head.in_features)).to(device)
    # print(outputs)
    for i in range(len(s_features)):
        if exist[i]:
            indices = torch.where(outputs_s==i)[0]
            if len(indices) != 0:
                s_features[i] = torch.sum(f_s[indices],dim=0)/len(indices)
    # print(s_features[0])

    s_centroids = (1-ro)*s_centroids.detach() + ro*s_features

    t_centroids_norm = t_centroids / (t_centroids.norm(dim=1)[:, None]+1e-10)
    s_centroids_norm = s_centroids / (s_centroids.norm(dim=1)[:, None]+1e-10)
    res = torch.exp(torch.mm(t_centroids_norm, s_centroids_norm.transpose(0,1))/temprature)

    loss4 = -1* torch.sum(torch.log(torch.diagonal(res,0)))+torch.sum(torch.log(torch.sum(res,dim = 0)))
    
    # Loss = nn.MSELoss()
    # loss4 = 100*Loss(t_centroids_norm,s_centroids_norm)
    
    return loss4,t_centroids,s_centroids