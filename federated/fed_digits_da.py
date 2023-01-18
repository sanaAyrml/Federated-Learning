"""
federated learning with different aggregation strategy on benchmark exp.
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import time
import copy
from nets.models import DigitModel
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import data_utils
import wandb
import os
from torch.utils.data import TensorDataset
from torchmetrics.classification import BinaryF1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau

### Domain adaptation modules import
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
from train_handler import train_uda,train,train_fedprox,test,communication

def prepare_data(args):
    # Prepare data
    transform_mnist = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_svhn = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_usps = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_synth = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnistm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # MNIST
    mnist_trainset = data_utils.DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=True,
                                              transform=transform_mnist)
    mnist_virtualset = data_utils.DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=True,
                                              transform=transform_mnist)
    mnist_testset = data_utils.DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=False,
                                             transform=transform_mnist)

    # SVHN
    svhn_trainset = data_utils.DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent, train=True,
                                             transform=transform_svhn)
    svhn_virtualset = data_utils.DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent, train=True,
                                             transform=transform_svhn)
    svhn_testset = data_utils.DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent, train=False,
                                            transform=transform_svhn)

    # USPS
    usps_trainset = data_utils.DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent, train=True,
                                             transform=transform_usps)
    usps_virtualset = data_utils.DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent, train=True,
                                             transform=transform_usps)
    usps_testset = data_utils.DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent, train=False,
                                            transform=transform_usps)

    # Synth Digits
    synth_trainset = data_utils.DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,
                                              train=True, transform=transform_synth)
    synth_virtualset = data_utils.DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,
                                              train=True, transform=transform_synth)
    synth_testset = data_utils.DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,
                                             train=False, transform=transform_synth)

    # MNIST-M
    mnistm_trainset = data_utils.DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,
                                               train=True, transform=transform_mnistm)
    mnistm_virtualset = data_utils.DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,
                                               train=True, transform=transform_mnistm)
    mnistm_testset = data_utils.DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,
                                              train=False, transform=transform_mnistm)

    trainsets = [mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset]
    virtualsets = [mnist_virtualset, svhn_virtualset, usps_virtualset, synth_virtualset, mnistm_virtualset]
    testsets = [mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset]

    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    mnist_virtual_loader = torch.utils.data.DataLoader(mnist_virtualset, batch_size=args.batch, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch, shuffle=True)
    svhn_virtual_loader = torch.utils.data.DataLoader(svhn_virtualset, batch_size=args.batch, shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch, shuffle=True)
    usps_virtual_loader = torch.utils.data.DataLoader(usps_virtualset, batch_size=args.batch, shuffle=True)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
    synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch, shuffle=True)
    synth_virtual_loader = torch.utils.data.DataLoader(synth_virtualset, batch_size=args.batch, shuffle=True)
    synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch, shuffle=True)
    mnistm_virtual_loader = torch.utils.data.DataLoader(mnistm_virtualset, batch_size=args.batch, shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    virtual_loaders = [mnist_virtual_loader, svhn_virtual_loader, usps_virtual_loader, synth_virtual_loader, mnistm_virtual_loader]
    test_loaders = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]

    return trainsets,virtualsets,testsets,train_loaders,virtual_loaders, test_loaders



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('Device:', device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to make a log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--percent', type=float, default=0.1, help='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=100, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='fedbn', help='fedavg | fedprox | fedbn | fedda')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type=str, default='../checkpoint/digits', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')
    parser.add_argument('--project_name', type=str, default='fed_digit', help='name of wandb project')
    parser.add_argument('--cuda_num', type=int, default=0, help='cuda num')
    parser.add_argument('--attack_mode', action='store_true', help='whether to make a log')
    parser.add_argument('--attack_batch', type=int, default=500, help='attack batch size')
    parser.add_argument('--weighted_loss', action='store_true', help='whether to compute loss weighted')

    parser.add_argument('--uda_type', default='cdan')
    parser.add_argument('--param_cdan', default=1, type=float)
    parser.add_argument('--param_dann', default=0., type=float)
    parser.add_argument('--param_cls_s', default=0., type=float)
    parser.add_argument('--param_mi', default=1., type=float)

    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.cuda_num) if torch.cuda.is_available() else 'cpu')
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    lr_factor = 0.3  # Learning rate decrease factor
    lr_patience = 5
    lr_threshold = 0.0001

    print('Device:', device)

    exp_folder = 'federated_medical'
    Best_Global_model = None
    Best_local_models = None

    args.save_path = os.path.join(args.save_path, exp_folder)

    log = args.log
    if log:
        log_path = os.path.join('../logs/medical/', exp_folder)
        os.environ["WANDB_API_KEY"] = 'f87c7a64e4a4c89c4f1afc42620ac211ceb0f926'
        wandb.init(project=args.project_name, entity="sanaayr", config=args)
        wandb.run.name = args.mode
        wandb.run.save()
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path, '{}.log'.format(args.mode)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))
        logfile.write('    wk_iters: {}\n'.format(args.wk_iters))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))

    server_model = DigitModel().to(device)
    loss_fun = nn.CrossEntropyLoss()

    # prepare the data
    trainsets,virtualsets,testsets,train_loaders,virtual_loaders, test_loaders = prepare_data(args)

    # name of each client dataset
    datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']

    # federated setting
    client_num = len(datasets)
    client_weights = [1 / client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('../snapshots/digits/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        else:
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(server_model, test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        resume_iter = int(checkpoint['a_iter']) + 1
        print('Resume training from epoch {}'.format(resume_iter))
    else:
        resume_iter = 0
    max_train_acc = 0
    max_test_acc = 0
    lr = args.lr
    # lr_schedulers =  [ReduceLROnPlateau(optimizer, factor=lr_factor, patience=lr_patience, threshold=lr_threshold) for optimizer]
    # start training
    patience = 0
    trainset_num_classes = 10
    domain_discri = []
    domain_adv = []
    print(models[0])
    features_dim = 512
    for client_idx in range(client_num):
        if args.uda_type == 'dann':
            domain_discri.append(DomainDiscriminator(in_feature=features_dim, hidden_size=1024).to(device))
            domain_adv.append((domain_discri[client_idx]).to(device))
        elif args.uda_type == 'cdan':
            domain_discri.append(DomainDiscriminator(features_dim * trainset_num_classes, hidden_size=1024).to(device))
            domain_adv.append(ConditionalDomainAdversarialLoss(domain_discri[client_idx], entropy_conditioning=False,
                                                          num_classes=trainset_num_classes,
                                                          features_dim= features_dim, randomized=False).to(device))

    for a_iter in range(resume_iter, args.iters):
        if a_iter > 0:
            lr = args.lr / a_iter
            if patience == 8:
                server_model = Best_Global_model
                models = Best_local_models
                patience = 0
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=lr) for idx in range(client_num)]
        if args.mode.lower() == 'fedda':
            for param in server_model.parameters():
                param.requires_grad = False
            server_model.eval()
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log: logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

            for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                if args.mode.lower() == 'fedprox':
                    if a_iter > 0:
                        train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device)
                    else:
                        train(model, train_loader, optimizer, loss_fun, client_num, device)
                if args.mode.lower() == 'fedda':
                    if a_iter > 20:
                        train_uda(trg_loader =train_loader , src_loader = virtual_loaders[client_idx], trg_model = model, 
                                  domain_adv = domain_adv[client_idx] , optimizer = optimizer , epoch = 10, args=args, device = device)
                    else:
                        train(model, train_loader, optimizer, loss_fun, client_num, device)
                else:
                    train(model, train_loader, optimizer, loss_fun, client_num, device)
                train_loss, train_acc = test(model, train_loader, loss_fun, device)
                test_loss, test_acc = test(model, test_loaders[client_idx], loss_fun, device)
                if args.log:
                    metrics = {"Train_ACC_Local_" + str(client_idx): train_acc,
                               "Test_ACC_Local_" + str(client_idx): test_acc}
                    wandb.log(metrics)

        # aggregation
        server_model, models = communication(args, server_model, models, client_weights)

        # report after aggregation
        avg_train = 0
        for client_idx in range(client_num):
            model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
            train_loss, train_acc = test(model, train_loader, loss_fun, device)
            avg_train += train_acc
            print(
                ' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], train_loss, train_acc))
            if args.log:
                logfile.write(
                    ' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx], train_loss,
                                                                                train_acc))
                metrics = {"Train_ACC_" + str(client_idx): train_acc,
                           "Train_Loss_" + str(client_idx): train_loss}
                wandb.log(metrics)
        if max_train_acc < avg_train / client_num:
            max_train_acc = avg_train / client_num
        if args.log:
            metrics = {"Train_AVG": avg_train / client_num}
            wandb.log(metrics)

        # start testing
        avg_test = 0
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            avg_test += test_acc
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
            if args.log:
                logfile.write(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_loss,
                                                                                          test_acc))
                metrics = {"Test_ACC_" + str(test_idx): test_acc,
                           "Test_Loss_" + str(test_idx): test_loss}
                wandb.log(metrics)

        if max_test_acc < avg_test / client_num:
            max_test_acc = avg_test / client_num
            Best_Global_model = server_model
            Best_local_models = models
            patience = 0
        else:
            patience += 1

        if args.log:
            metrics = {"Test_AVG": avg_test / client_num}
            wandb.log(metrics)
        print('Maximum train accuracy average is:', max_train_acc)
        print('Maximum test accuracy average is:', max_test_acc)

    # Save checkpoint
    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    if args.mode.lower() == 'fedbn':
        torch.save({
            'model_0': models[0].state_dict(),
            'model_1': models[1].state_dict(),
            'model_2': models[2].state_dict(),
            'model_3': models[3].state_dict(),
            'model_4': models[4].state_dict(),
            'server_model': server_model.state_dict(),
        }, SAVE_PATH)
    else:
        torch.save({
            'server_model': server_model.state_dict(),
        }, SAVE_PATH)

    if log:
        logfile.flush()
        logfile.close()

