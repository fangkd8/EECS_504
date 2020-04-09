import torch
import torch.nn as nn


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.lrelu5 = nn.LeakyReLU(0.2, inplace=True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.lrelu6 = nn.LeakyReLU(0.2, inplace=True)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.lrelu7 = nn.LeakyReLU(0.2, inplace=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.relu8 = nn.ReLU(inplace=True)

        self.up9 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.d9 = nn.Dropout(0.5, inplace=False)
        self.relu9 = nn.ReLU(inplace=True)
        self.up10 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.d10 = nn.Dropout(0.5, inplace=False)
        self.relu10 = nn.ReLU(inplace=True)
        self.up11 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.d11 = nn.Dropout(0.5, inplace=False)
        self.relu11 = nn.ReLU(inplace=True)
        self.up12 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU(inplace=True)
        self.up13 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(256)
        self.relu13 = nn.ReLU(inplace=True)
        self.up14 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.bn14 = nn.BatchNorm2d(128)
        self.relu14 = nn.ReLU(inplace=True)
        self.up15 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(64)
        self.relu15 = nn.ReLU(inplace=True)
        self.up16 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        c1 = self.conv1(input)
        lr1 = self.lrelu1(c1)
        c2 = self.conv2(lr1)
        b2 = self.bn2(c2)
        lr2 = self.lrelu2(b2)
        c3 = self.conv3(lr2)
        b3 = self.bn3(c3)
        lr3 = self.lrelu3(b3)
        c4 = self.conv4(lr3)
        b4 = self.bn4(c4)
        lr4 = self.lrelu4(b4)
        c5 = self.conv5(lr4)
        b5 = self.bn5(c5)
        lr5 = self.lrelu5(b5)
        c6 = self.conv6(lr5)
        b6 = self.bn6(c6)
        lr6 = self.lrelu6(b6)
        c7 = self.conv7(lr6)
        b7 = self.bn7(c7)
        lr7 = self.lrelu7(b7)
        c8 = self.conv8(lr7)
        r8 = self.relu8(c8)

        up_9 = self.up9(r8)
        up_9_bn = self.bn9(up_9)
        up_9_d = self.d9(up_9_bn)
        merge_9 = torch.cat([up_9_d, lr7], dim=1)
        up_9_relu = self.relu9(merge_9)
        up_10 = self.up10(up_9_relu)
        up_10_bn = self.bn10(up_10)
        up_10_d = self.d10(up_10_bn)
        merge_10 = torch.cat([up_10_d, lr6], dim=1)
        up_10_relu = self.relu10(merge_10)
        up_11 = self.up11(up_10_relu)
        up_11_bn = self.bn11(up_11)
        up_11_d = self.d11(up_11_bn)
        merge_11 = torch.cat([up_11_d, lr5], dim=1)
        up_11_relu = self.relu11(merge_11)
        up_12 = self.up12(up_11_relu)
        up_12_bn = self.bn12(up_12)
        merge_12 = torch.cat([up_12_bn, lr4], dim=1)
        up_12_relu = self.relu12(merge_12)
        up_13 = self.up13(up_12_relu)
        up_13_bn = self.bn13(up_13)
        merge_13 = torch.cat([up_13_bn, lr3], dim=1)
        up_13_relu = self.relu13(merge_13)
        up_14 = self.up14(up_13_relu)
        up_14_bn = self.bn14(up_14)
        merge_14 = torch.cat([up_14_bn, lr2], dim=1)
        up_14_relu = self.relu14(merge_14)
        up_15 = self.up15(up_14_relu)
        up_15_bn = self.bn15(up_15)
        merge_15 = torch.cat([up_15_bn, lr1], dim=1)
        up_15_relu = self.relu15(merge_15)
        up16 = self.up16(up_15_relu)
        o = nn.Tanh()(up16)


        return o


class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


    # forward method
    def forward(self, input, label):
        input_new = torch.cat([input, label], dim=1)
        c1 = self.conv1(input_new)
        lr1 = self.lrelu1(c1)
        c2 = self.conv2(lr1)
        b2 = self.bn2(c2)
        lr2 = self.lrelu2(b2)
        c3 = self.conv3(lr2)
        b3 = self.bn3(c3)
        lr3 = self.lrelu3(b3)
        c4 = self.conv4(lr3)
        b4 = self.bn4(c4)
        lr4 = self.lrelu4(b4)
        c5 = self.conv5(lr4)
        x = nn.Sigmoid()(c5)


        return x