import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import itertools

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# import self functions
from preprocess import *
from network import *
from train_network import *


if __name__ == '__main__':
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only.")


    # data_loader
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    (len_train_label, train_label_loader) = load_data_label(
        '/home/chendh/Desktop/label',  #/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/gtFine_trainvaltest/gtFine
        'train', transform, 1, shuffle=False
    )
    (len_train_raw, train_raw_loader) = load_data_raw(
        '/home/chendh/Desktop/raw',  #/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/leftImg8bit_trainvaltest/leftImg8bit
        'train', transform, 1, shuffle=False
    )
    (len_test_label, test_label_loader) = load_data_label(
        '/home/chendh/Desktop/label',  #/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/gtFine_trainvaltest/gtFine
        'val', transform, 3, shuffle=False
    )
    (len_test_raw, test_raw_loader) = load_data_raw(
        '/home/chendh/Desktop/raw',  #/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/leftImg8bit_trainvaltest/leftImg8bit
        'val', transform, 3, shuffle=False
    )
    test_label = test_label_loader.__iter__().__next__()[0]
    test_raw = test_raw_loader.__iter__().__next__()[0]
    img_size = test_label.size()[2]
    fixed_y_ = test_label
    fixed_x_ = test_raw


    # train test
    # Define network
    G_100 = generator()
    D_100 = discriminator()
    G_100.weight_init(mean=0.0, std=0.02)
    D_100.weight_init(mean=0.0, std=0.02)
    G_100.cuda()
    D_100.cuda()
    G_100.train()
    D_100.train()

    # test
    # Report the architectures of your network
    print(G_100)
    print('Number of trainable parameters {}'.format(count_params(G_100)))

    print(D_100)
    print('Number of trainable parameters {}'.format(count_params(D_100)))

    # training
    hist_D_100_losses, hist_G_100_losses = train(
        train_label_loader, train_raw_loader, img_size, fixed_x_, fixed_y_,
        G_100, D_100, num_epochs=20, only_L1=False
    )