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
from utils.data_utils import DomainNetDataset
import wandb
import os
from torch.utils.data import TensorDataset
from torchmetrics.classification import BinaryF1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

### Domain adaptation modules import
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
from train_handler import train_uda, train, train_fedprox, test, communication, visualize, visualize_all,fit_umap
from synthesize import src_img_synth_admm
from digit_net import ImageClassifier


def prepare_data(args,datasets,public_dataset):
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
    mnist_trainset = data_utils.DigitsDataset(data_path="data/MNIST", channels=1, percent=args.percent, train=True,
                                              transform=transform_mnist)
    mnist_virtualset = data_utils.DigitsDataset(data_path="data/MNIST", channels=1, percent=args.percent, train=True,
                                                transform=transform_mnist)
    mnist_testset = data_utils.DigitsDataset(data_path="data/MNIST", channels=1, percent=args.percent, train=False,
                                             transform=transform_mnist)

    # SVHN
    svhn_trainset = data_utils.DigitsDataset(data_path='data/SVHN', channels=3, percent=args.percent, train=True,
                                             transform=transform_svhn)
    svhn_virtualset = data_utils.DigitsDataset(data_path='data/SVHN', channels=3, percent=args.percent, train=True,
                                               transform=transform_svhn)
    svhn_testset = data_utils.DigitsDataset(data_path='data/SVHN', channels=3, percent=args.percent, train=False,
                                            transform=transform_svhn)

    # USPS
    usps_trainset = data_utils.DigitsDataset(data_path='data/USPS', channels=1, percent=args.percent, train=True,
                                             transform=transform_usps)
    usps_virtualset = data_utils.DigitsDataset(data_path='data/USPS', channels=1, percent=args.percent, train=True,
                                               transform=transform_usps)
    usps_testset = data_utils.DigitsDataset(data_path='data/USPS', channels=1, percent=args.percent, train=False,
                                            transform=transform_usps)

    # Synth Digits
    synth_trainset = data_utils.DigitsDataset(data_path='data/SynthDigits/', channels=3, percent=args.percent,
                                              train=True, transform=transform_synth)
    synth_testset = data_utils.DigitsDataset(data_path='data/SynthDigits/', channels=3, percent=args.percent,
                                             train=False, transform=transform_synth)
    # synth_virtualset = torch.utils.data.ConcatDataset([synth_trainset, synth_testset])
    synth_virtualset = data_utils.DigitsDataset(data_path='data/SynthDigits/', channels=3, percent=args.percent,
                                              train=True, transform=transform_synth)

    # MNIST-M
    mnistm_trainset = data_utils.DigitsDataset(data_path='data/MNIST_M/', channels=3, percent=args.percent,
                                               train=True, transform=transform_mnistm)
    mnistm_virtualset = data_utils.DigitsDataset(data_path='data/MNIST_M/', channels=3, percent=args.percent,
                                                 train=True, transform=transform_mnistm)
    mnistm_testset = data_utils.DigitsDataset(data_path='data/MNIST_M/', channels=3, percent=args.percent,
                                              train=False, transform=transform_mnistm)

    trainsets = []
    virtualsets = []
    testsets = []
    generatsets = []
    for dataset in datasets:
        if dataset == 'MNIST':
            trainsets.append(mnist_trainset)
            testsets.append(mnist_testset)
            if public_dataset == None:
                generatsets.append(mnist_trainset)
                virtualsets.append(mnist_virtualset)
                
        elif dataset == 'SVHN':
            trainsets.append(svhn_trainset)
            testsets.append(svhn_testset)
            if public_dataset == None:
                generatsets.append(svhn_trainset)
                virtualsets.append(svhn_virtualset)
    
        elif dataset == 'USPS':
            trainsets.append(usps_trainset)
            testsets.append(usps_testset)
            if public_dataset == None:
                generatsets.append(usps_trainset)
                virtualsets.append(usps_virtualset)
        
        elif dataset == 'MNIST-M':
            trainsets.append(mnistm_trainset)
            testsets.append(mnistm_testset)
            if public_dataset == None:
                generatsets.append(mnistm_trainset)
                virtualsets.append(mnistm_virtualset)
                
        elif dataset == 'SynthDigits':
            trainsets.append(synth_trainset)
            testsets.append(synth_testset)
            if public_dataset == None:
                generatsets.append(synth_trainset)
                virtualsets.append(synth_virtualset)
                                   
    if public_dataset != None:
        if public_dataset == 'MNIST':
            generatsets.append(mnist_trainset)
            virtualsets.append(mnist_virtualset)
                
        elif public_dataset == 'SVHN':
            generatsets.append(svhn_trainset)
            virtualsets.append(svhn_virtualset)

        elif public_dataset == 'USPS':
            generatsets.append(usps_trainset)
            virtualsets.append(usps_virtualset)
        
        elif public_dataset == 'MNIST-M':
            generatsets.append(mnistm_trainset)
            virtualsets.append(mnistm_virtualset)
                
        elif public_dataset == 'SynthDigits':
            generatsets.append(synth_trainset)
            virtualsets.append(synth_virtualset)
        

    return trainsets, virtualsets, testsets, generatsets


def prepare_domainnet(args, datasets, public_dataset):
    # Prepare data
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    data_base_path = 'data'

    clipart_trainset = DomainNetDataset(data_base_path, 'clipart', transform=transform_train)
    clipart_virtualset = DomainNetDataset(data_base_path, 'clipart', transform=transform_train)
    clipart_testset = DomainNetDataset(data_base_path, 'clipart', transform=transform_test, train=False)
    # infograph
    infograph_trainset = DomainNetDataset(data_base_path, 'infograph', transform=transform_train)
    infograph_virtualset = DomainNetDataset(data_base_path, 'infograph', transform=transform_train)
    infograph_testset = DomainNetDataset(data_base_path, 'infograph', transform=transform_test, train=False)
    # painting
    painting_trainset = DomainNetDataset(data_base_path, 'painting', transform=transform_train)
    painting_virtualset = DomainNetDataset(data_base_path, 'painting', transform=transform_train)
    painting_testset = DomainNetDataset(data_base_path, 'painting', transform=transform_test, train=False)
    # quickdraw
    quickdraw_trainset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_train)
    quickdraw_virtualset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_train)
    quickdraw_testset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_test, train=False)
    # real
    real_trainset = DomainNetDataset(data_base_path, 'real', transform=transform_train)
    real_virtualset = DomainNetDataset(data_base_path, 'real', transform=transform_train)
    real_testset = DomainNetDataset(data_base_path, 'real', transform=transform_test, train=False)
    # sketch
    sketch_trainset = DomainNetDataset(data_base_path, 'sketch', transform=transform_train)
    sketch_virtualset = DomainNetDataset(data_base_path, 'sketch', transform=transform_train)
    sketch_testset = DomainNetDataset(data_base_path, 'sketch', transform=transform_test, train=False)

    min_data_len = min(len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset),
                       len(real_trainset), len(sketch_trainset))
    min_data_len = int(min_data_len * 0.05)

    # clipart_valset = torch.utils.data.Subset(clipart_trainset, list(range(len(clipart_trainset)))[-val_len:])
    clipart_trainset = torch.utils.data.Subset(clipart_trainset, list(range(min_data_len)))

    # infograph_valset = torch.utils.data.Subset(infograph_trainset, list(range(len(infograph_trainset)))[-val_len:])
    infograph_trainset = torch.utils.data.Subset(infograph_trainset, list(range(min_data_len)))

    # painting_valset = torch.utils.data.Subset(painting_trainset, list(range(len(painting_trainset)))[-val_len:])
    painting_trainset = torch.utils.data.Subset(painting_trainset, list(range(min_data_len)))

    # quickdraw_valset = torch.utils.data.Subset(quickdraw_trainset, list(range(len(quickdraw_trainset)))[-val_len:])
    quickdraw_trainset = torch.utils.data.Subset(quickdraw_trainset, list(range(min_data_len)))

    # real_valset = torch.utils.data.Subset(real_trainset, list(range(len(real_trainset)))[-val_len:])
    real_trainset = torch.utils.data.Subset(real_trainset, list(range(min_data_len)))

    # sketch_valset = torch.utils.data.Subset(sketch_trainset, list(range(len(sketch_trainset)))[-val_len:])
    sketch_trainset = torch.utils.data.Subset(sketch_trainset, list(range(min_data_len)))


    trainsets = []
    virtualsets = []
    testsets = []
    generatsets = []
    for dataset in datasets:
        if dataset == 'clipart':
            trainsets.append(mnist_trainset)
            testsets.append(mnist_testset)
            if public_dataset == None:
                generatsets.append(mnist_trainset)
                virtualsets.append(mnist_virtualset)

        elif dataset == 'infograph':
            trainsets.append(svhn_trainset)
            testsets.append(svhn_testset)
            if public_dataset == None:
                generatsets.append(svhn_trainset)
                virtualsets.append(svhn_virtualset)

        elif dataset == 'painting':
            trainsets.append(usps_trainset)
            testsets.append(usps_testset)
            if public_dataset == None:
                generatsets.append(usps_trainset)
                virtualsets.append(usps_virtualset)

        elif dataset == 'quickdraw':
            trainsets.append(mnistm_trainset)
            testsets.append(mnistm_testset)
            if public_dataset == None:
                generatsets.append(mnistm_trainset)
                virtualsets.append(mnistm_virtualset)

        elif dataset == 'real':
            trainsets.append(synth_trainset)
            testsets.append(synth_testset)
            if public_dataset == None:
                generatsets.append(synth_trainset)
                virtualsets.append(synth_virtualset)

        elif dataset == 'sketch':
            trainsets.append(synth_trainset)
            testsets.append(synth_testset)
            if public_dataset == None:
                generatsets.append(synth_trainset)
                virtualsets.append(synth_virtualset)

    if public_dataset != None:
        if public_dataset == 'clipart':
            generatsets.append(mnist_trainset)
            virtualsets.append(mnist_virtualset)

        elif public_dataset == 'infograph':
            generatsets.append(svhn_trainset)
            virtualsets.append(svhn_virtualset)

        elif public_dataset == 'painting':
            generatsets.append(usps_trainset)
            virtualsets.append(usps_virtualset)

        elif public_dataset == 'quickdraw':
            generatsets.append(mnistm_trainset)
            virtualsets.append(mnistm_virtualset)

        elif public_dataset == 'real':
            generatsets.append(synth_trainset)
            virtualsets.append(synth_virtualset)

        elif public_dataset == 'sketch':
            generatsets.append(synth_trainset)
            virtualsets.append(synth_virtualset)

    return trainsets, virtualsets, testsets, generatsets