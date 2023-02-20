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