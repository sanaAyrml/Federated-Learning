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

def train_uda(trg_loader: DataLoader, src_loader: DataLoader, trg_model, domain_adv, optimizer: SGD, epoch: int, args: argparse.Namespace, device):
    
    # batch_time = AverageMeter('Time', ':5.2f')
    # losses = AverageMeter('Loss', ':6.2f')
    num_data = 0
    correct = 0
    loss_all = 0
 
    trg_model.train()
    
    preds = []
    targets = []
    first_run = True
    
    for e in range(epoch):
        src_iter = iter(src_loader)
        for i, (images_t, target) in enumerate(trg_loader):

            try:
                images_s, labels_s = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                images_s, labels_s = next(src_iter)

            images_s = images_s.to(device)
            labels_s = labels_s.to(device)
            images_t = images_t.to(device)

            y_s, f_s = trg_model(images_s)
            y_t, f_t = trg_model(images_t)

            # print(y_t.shape, f_t.shape,target.shape)
            loss = 0.
            if args.param_cls_s > 0:
                cls_loss_s = F.cross_entropy(y_s, labels_s)
                loss += cls_loss_s * args.param_cls_s

            if args.uda_type == 'dann' and  args.param_dann > 0:
                dann_loss = domain_adv(f_s, f_t)
                loss += dann_loss * args.param_dann

            if args.uda_type == 'cdan' and  args.param_cdan > 0:
                # print('here')
                cdan_loss = domain_adv(y_s, f_s, y_t, f_t)
                loss += cdan_loss * args.param_cdan

            if args.param_mi > 0:
                p_t = F.softmax(y_t, dim=1)
                mi_loss = mut_info_loss(p_t)
                loss += mi_loss * args.param_mi

            # losses.update(loss.item(), images_t.size(0))
            loss_all += loss.item()
            num_data += images_t.size(0)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = y_t.data.max(1)[1]

            correct += pred.eq(target.to(device).view(-1)).sum().item()
        
    return loss_all / len(src_iter), correct / num_data
        


            
            
def train(model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    preds = []
    targets = []
    first_run = True
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output,_ = model(x)

        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all / len(train_iter), correct / num_data


def train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    first_run = True
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output,_ = model(x)

        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all / len(train_iter), correct / num_data


def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    first_run = True
    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output,_ = model(data)

        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()

    return test_loss / len(test_loader), correct / len(test_loader.dataset)


################# Key Function ########################
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models