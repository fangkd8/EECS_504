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
import cv2
import matplotlib.image
from PIL import Image

# import self functions
from preprocess import *
from network import *
from process_network import *
from label_visualize import *
from evaluate_network import *


if __name__ == '__main__':
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only.")


    # transform for train and test
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # set training batch size
    batch_size_train = 1

    (len_train_label, train_label_loader) = load_data_labelID(
        '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/gtFine_trainvaltest/gtFine',  #/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/gtFine_trainvaltest/gtFine; /home/chendh/Desktop/label
        'train', transform, batch_size_train, shuffle=False
    )
    (len_train_raw, train_raw_loader) = load_data_raw(
        '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/leftImg8bit_trainvaltest/leftImg8bit',  #/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/leftImg8bit_trainvaltest/leftImg8bit; /home/chendh/Desktop/raw
        'train', transform, batch_size_train, shuffle=False
    )
    (len_test_label, test_label_loader) = load_data_label(
        '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/gtFine_trainvaltest/gtFine',  #/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/gtFine_trainvaltest/gtFine; /home/chendh/Desktop/label
        'val', transform, 5, shuffle=False
    )
    (len_test_raw, test_raw_loader) = load_data_raw(
        '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/leftImg8bit_trainvaltest/leftImg8bit',  #/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/leftImg8bit_trainvaltest/leftImg8bit; /home/chendh/Desktop/raw
        'val', transform, 5, shuffle=False
    )
    '''
    test_label = test_label_loader.__iter__().__next__()[0]
    test_raw = test_raw_loader.__iter__().__next__()[0]
    img_size = test_label.size()[2]
    '''
    test_label_iter = iter(test_label_loader)
    test_raw_iter = iter(test_raw_loader)
    for i in range(1, 2):
        data1 = test_label_iter.next()
        data2 = test_raw_iter.next()
    test_label = data1[0]
    test_raw = data2[0]
    #img_size = test_label.size()[2]

    train_label = train_label_loader.__iter__().__next__()[0]  #([1, 3, 1024, 2048])
    train_raw = train_raw_loader.__iter__().__next__()[0]
    #print(train_label.shape)
    #print(torch.as_tensor((train_label[0]*255), dtype=torch.uint8, device=device))


    '''
    para define whether to train or test or predict
    1->train; 2->keep training; 3->test&evaluate; 4->predict
    '''
    flag_mode = 2

    if flag_mode == 1:
        # define and report the network
        model = Unet(3, 34)
        model.cuda()
        model.train()
        # report the architectures of Unet
        print(model)
        print('Number of trainable parameters {}'.format(count_params(model)))
        # train
        (model, hist_losses) = train(
            model, train_label_loader, train_raw_loader, test_raw, test_label, num_epochs=20
        )
        # plot the loss history
        plot_loss(hist_losses)

    elif flag_mode == 2:
        # define and report the network
        model = Unet(3, 34)
        # select trained weights to keep training
        start_epoch = 105
        weights_file_path = '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Final_Project/weights/weights_epoch104.pth'
        model.load_state_dict(torch.load(weights_file_path))
        model.cuda()
        model.train()
        # report the architectures of Unet
        print(model)
        print('Number of trainable parameters {}'.format(count_params(model)))
        # train
        (model, hist_losses) = keep_train(
            model, train_label_loader, train_raw_loader, test_raw, test_label, start_epoch, num_epochs=120
        )
        # plot the loss history
        plot_loss(hist_losses)

    elif flag_mode == 3:
        # define and report the network
        model = Unet(3, 34)
        weights_file_path = '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Final_Project/weights/weights_epoch86.pth'
        img_raw_path = '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000001_032018_leftImg8bit.png'
        img_label_path = '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/gtFine_trainvaltest/gtFine/val/frankfurt/frankfurt_000001_032018_gtFine_labelIds.png'
        model.load_state_dict(torch.load(weights_file_path))
        model = model.cuda()

        # transform the input image
        transform = transforms.Compose([
            transforms.Resize((512, 1024)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        img_raw = Image.open(img_raw_path)
        img_input = transform(img_raw).unsqueeze(0).cuda()
        #print(img_input.shape)
        img_label = Image.open(img_label_path)
        predict_label = model(img_input)
        (temp_max, preds) = torch.max(predict_label.data, 1)  # preds->(1, 512, 1024)
        #print(preds.shape)
        imgPredict = pyr_up_IOU(preds[0].cpu().data.numpy())
        print(imgPredict)
        imgLabel = np.array(img_label)
        print(imgLabel)

        (iou_scores, ave) = mIOU(imgLabel, imgPredict, 34)
        print(iou_scores)
        print(ave)
        '''
        metric = SegmentationMetric(34)
        metric.addBatch(imgPredict, imgLabel)
        acc = metric.pixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        print(acc, mIoU)
        '''

    elif flag_mode == 4:
        # define and report the network
        model = Unet(3, 34)
        weights_file_path = '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Final_Project/weights/weights_epoch104.pth'
        dir_path = '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/2011_10_03/2011_10_03_drive_0027_sync/image_03/data'#'/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'#'/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Datasets/Prediction/0000000068.png'

        '''
        test->0; predict kitti->1
        '''
        flag_crop = 1
        if flag_crop == 0:
            #predict(model, weights_file_path, dir_path, flag_crop, test_raw, test_label)
            # model.eval()
            model.load_state_dict(torch.load(weights_file_path))
            model = model.cuda()
            with torch.no_grad():
                save_visualized_result_test(model, test_raw, test_label, 100)
        else:
            predict(
                model, weights_file_path, dir_path,
                flag_crop, test_raw, test_label, 0
            )
            #print(output_label)