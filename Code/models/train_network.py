import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable


# Helper function for showing result.
def process_image(img):
    return (img.cpu().data.numpy().transpose(1, 2, 0) + 1) / 2


def show_result(G, x_, y_, num_epoch):
    predict_images = G(x_)

    fig, ax = plt.subplots(x_.size()[0], 3, figsize=(10, 30))

    for i in range(x_.size()[0]):
        ax[i, 0].get_xaxis().set_visible(False)
        ax[i, 0].get_yaxis().set_visible(False)
        ax[i, 1].get_xaxis().set_visible(False)
        ax[i, 1].get_yaxis().set_visible(False)
        ax[i, 2].get_xaxis().set_visible(False)
        ax[i, 2].get_yaxis().set_visible(False)
        ax[i, 0].cla()
        ax[i, 0].imshow(process_image(x_[i]))
        ax[i, 1].cla()
        ax[i, 1].imshow(process_image(predict_images[i]))
        ax[i, 2].cla()
        ax[i, 2].imshow(process_image(y_[i]))

    plt.tight_layout()
    label_epoch = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0, label_epoch, ha='center')
    label_input = 'Input'
    fig.text(0.18, 1, label_input, ha='center')
    label_output = 'Output'
    fig.text(0.5, 1, label_output, ha='center')
    label_truth = 'Ground truth'
    fig.text(0.81, 1, label_truth, ha='center')

    plt.show()


def count_params(model):
    num_params = sum([item.numel() for item in model.parameters() if item.requires_grad])


    return num_params


BCE_loss = nn.BCELoss().cuda()
L1_loss = nn.L1Loss().cuda()


def train(train_label_loader, train_raw_loader, img_size, fixed_x_, fixed_y_, G, D, num_epochs=20, only_L1=False):
    hist_D_losses = []
    hist_G_losses = []

    # Adam optimizer
    G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    print('training start!')
    start_time = time.time()
    for epoch in range(num_epochs):
        print('Start training epoch %d' % (epoch + 1))
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        num_iter = 0
        for (train_label, train_raw) in zip(train_label_loader, train_raw_loader):
            y_, temp1 = train_label
            x_, temp2 = train_raw

            x_, y_ = x_.cuda(), y_.cuda()
            # train discriminator D
            # Compute loss of real_img
            real_out = D(x_, y_).squeeze()
            real_label_d = Variable(torch.ones(real_out.size()).cuda())
            loss_D_real = BCE_loss(real_out, real_label_d)

            # Compute loss of fake_img
            fake_img_d = G(x_)
            fake_out_d = D(x_, fake_img_d).squeeze()
            fake_label = Variable(torch.zeros(fake_out_d.size()).cuda())
            loss_D_fake = BCE_loss(fake_out_d, fake_label)

            # Loss D
            loss_D = (loss_D_real + loss_D_fake) / 2

            # Bp and optimize
            D_optimizer.zero_grad()
            loss_D.backward()
            D_optimizer.step()

            D_losses.append(loss_D)
            hist_D_losses.append(loss_D)

            # train generator G
            # Compute loss of fake_img
            fake_img_g = G(x_)

            if only_L1 == False:
                fake_out_g = D(x_, fake_img_g).squeeze()
                real_label_g = Variable(torch.ones(fake_out_g.size()).cuda())
                loss_G = BCE_loss(fake_out_g, real_label_g) + 100 * L1_loss(fake_img_g, y_)
            else:
                loss_G = 100 * L1_loss(fake_img_g, y_)

            # Bp and optimize
            G_optimizer.zero_grad()
            loss_G.backward()
            G_optimizer.step()

            G_losses.append(loss_G)
            hist_G_losses.append(loss_G)
            num_iter += 1

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - using time: %.2f' % ((epoch + 1), num_epochs, per_epoch_ptime))
        print('loss of discriminator D: %.3f' % (torch.mean(torch.FloatTensor(D_losses))))
        print('loss of generator G: %.3f' % (torch.mean(torch.FloatTensor(G_losses))))
        print('Sample Image:')
        show_result(G, Variable(fixed_x_.cuda(), volatile=True), fixed_y_, (epoch + 1))

    end_time = time.time()
    total_ptime = end_time - start_time


    return hist_D_losses, hist_G_losses