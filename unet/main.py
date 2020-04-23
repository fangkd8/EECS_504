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
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    (len_train_label, train_label_loader) = load_data_label(
        '/home/chendh/Desktop/label',  #/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/gtFine_trainvaltest/gtFine; /home/chendh/Desktop/label
        'train', transform, 1, shuffle=False
    )
    (len_train_raw, train_raw_loader) = load_data_raw(
        '/home/chendh/Desktop/raw',  #/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/leftImg8bit_trainvaltest/leftImg8bit; /home/chendh/Desktop/raw
        'train', transform, 1, shuffle=False
    )
    (len_test_label, test_label_loader) = load_data_label(
        '/home/chendh/Desktop/label',  #/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/gtFine_trainvaltest/gtFine; /home/chendh/Desktop/label
        'val', transform, 5, shuffle=False
    )
    (len_test_raw, test_raw_loader) = load_data_raw(
        '/home/chendh/Desktop/raw',  #/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/leftImg8bit_trainvaltest/leftImg8bit; /home/chendh/Desktop/raw
        'val', transform, 5, shuffle=False
    )
    test_label = test_label_loader.__iter__().__next__()[0]
    test_raw = test_raw_loader.__iter__().__next__()[0]
    img_size = test_label.size()[2]

    # train split
    '''
    train_label = train_label_loader.__iter__().__next__()[0]
    train_raw = train_raw_loader.__iter__().__next__()[0]
    train_fixed = torch.cat((train_label, train_raw), 3)
    print(len_train_label)
    print(len_train_raw)
    '''

    '''
    # test success 1
    train_label = iter(train_label_loader)
    train_raw = iter(train_raw_loader)
    #train_combined = iter(train_combined_loader)
    for i in range(1, 10):
        data1 = train_label.next()
        data2 = train_raw.next()
        data3 = torch.cat((data1[0], data2[0]), 3)
        #data3 = train_combined.next()

        plt.imshow(data1[0][0][0])
        plt.show()
        plt.imshow(data2[0][0][0])
        plt.show()
        #plt.imshow(data3[0][0][0])
        #plt.show()
        plt.imshow(data3[0][0])
        plt.show()
    # test success 1
    '''

    '''
    # test success 2
    for (i, j) in zip(train_label_loader, train_raw_loader):
        a, b = i
        c, d = j
        plt.imshow(a[0][0])
        plt.show()
        plt.imshow(c[0][0])
        plt.show()
    # test success 2
    '''

    # Define and report the network
    model = Unet(3, 3)
    #model.weight_init(mean=0.0, std=0.02)
    model.cuda()
    model.train()
    # Report the architectures of Unet
    print(model)
    print('Number of trainable parameters {}'.format(count_params(model)))

    # train
    (model, hist_losses) = train(
        model, train_label_loader, train_raw_loader, test_raw, test_label, num_epochs=20
    )

    # plot the loss history
    plot_loss(hist_losses)
