import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from typing import Tuple, Optional, List, Dict
from torch import nn, optim
import time
import copy
from nets.models import DigitModel
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import data_utils
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as func
from torch.optim import SGD

def labels_to_one_hot(labels, num_class, device):
    # convert labels to one-hot
    labels_one_hot = torch.FloatTensor(labels.shape[0], num_class).to(device)
    labels_one_hot.zero_()
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return labels_one_hot

def src_img_synth_admm(gen_loader, src_model, args , device):

    src_model.eval()
    LAMB = torch.zeros_like(src_model.head.weight.data).to(device)
    gen_dataset = None
    gen_labels = None
    original_dataset = None
    original_labels = None
    for batch_idx, (images_s, labels_real) in enumerate(gen_loader):
        print(batch_idx,len(images_s))
        # if batch_idx == 10:
        #     break
        images_s = images_s.to(device)
        y_s,_ = src_model(images_s)
        labels_s = y_s.argmax(dim=1)
        if gen_dataset == None:
            gen_dataset = images_s
            gen_labels = labels_s
            original_dataset = images_s
            original_labels = labels_real
        else:
            gen_dataset = torch.cat((gen_dataset, images_s), 0)
            gen_labels = torch.cat((gen_labels, labels_s), 0)
            original_dataset = torch.cat((original_dataset, images_s), 0)
            original_labels = torch.cat((original_labels, labels_real), 0)

    for i in range(args.iters_admm):

        print(f'admm iter: {i}/{args.iters_admm}')

        # step1: update imgs
        for batch_idx, (images_s, labels_s) in enumerate(gen_loader):
            # if batch_idx == 10:
            #     break

    #        images_s = images_s.to(device)
    #        labels_s = labels_s.to(device)
            images_s = gen_dataset[batch_idx*args.batch:(batch_idx+1)*args.batch].clone().detach().to(device)
            labels_s = gen_labels[batch_idx*args.batch:(batch_idx+1)*args.batch].clone().detach().to(device)

            # convert labels to one-hot
            plabel_onehot = labels_to_one_hot(labels_s, 10, device)

            # init src img
            images_s.requires_grad_()
            optimizer_s = SGD([images_s], args.lr_img, momentum=args.momentum_img)
            
            for iter_i in range(args.iters_img):
                y_s, f_s = src_model(images_s)
                loss = func.cross_entropy(y_s, labels_s)
                p_s = func.softmax(y_s, dim=1)
                grad_matrix = (p_s - plabel_onehot).t() @ f_s / p_s.size(0)
                new_matrix = grad_matrix + args.param_gamma * src_model.head.weight.data
                grad_loss = torch.norm(new_matrix, p='fro') ** 2
                loss += grad_loss * args.param_admm_rho / 2
                loss += torch.trace(LAMB.t() @ new_matrix)
                
                optimizer_s.zero_grad()
                loss.backward()
                optimizer_s.step()

            # update src imgs
            gen_dataset[batch_idx*args.batch:(batch_idx+1)*args.batch] = images_s
       #     for img, path in zip(images_s.detach_().cpu(), paths):
       #         torch.save(img.clone(), path)

        # step2: update LAMB
        grad_matrix = torch.zeros_like(LAMB).to(device)
        for batch_idx, (images_s, labels_s) in enumerate(gen_loader):
            # if batch_idx == 10:
            #     break
       #     images_s = images_s.to(device)
       #     labels_s = labels_s.to(device)
            images_s = gen_dataset[batch_idx*args.batch:(batch_idx+1)*args.batch].clone().detach().to(device)
            labels_s = gen_labels[batch_idx*args.batch:(batch_idx+1)*args.batch].clone().detach().to(device)

            # convert labels to one-hot
            plabel_onehot = labels_to_one_hot(labels_s, 10, device)

            y_s, f_s = src_model(images_s)
            p_s = func.softmax(y_s, dim=1)
            grad_matrix += (p_s - plabel_onehot).t() @ f_s

        new_matrix = grad_matrix / len(gen_dataset) + args.param_gamma * src_model.head.weight.data
        LAMB += new_matrix * args.param_admm_rho

    return gen_dataset, gen_labels, original_dataset ,original_labels
