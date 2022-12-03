import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn

class Parser(nn.Module):
    def __init__(self, config): # config is not used..
        super(Parser, self).__init__()
        self.input_dim = 3
        self.class_num = 17      # (16 components + backgorund)
        self.cnum = 64

        # 128 * 128 * cnum
        self.conv1_1 = gen_conv(self.input_dim, self.cnum, 7, 1, 1,)
        self.conv1_2 = gen_conv(self.cnum, self.cnum, 3, 2, 1)
        #--self.pool1

        # 64 * 64 * cnum
        self.conv2_1 = gen_conv(self.cnum, self.cnum * 2, 3, 1, 1)
        self.conv2_2 = gen_conv(self.cnum * 2, self.cnum * 2, 3, 2, 1)
        #--self.pool2

        # 32 * 32 * cnum
        self.conv3_1  = gen_conv(self.cnum * 2, self.cnum * 4, 3, 1, 1)
        self.conv3_2 = gen_conv(self.cnum * 4, self.cnum * 4, 3, 1, 1)
        self.conv3_3 = gen_conv(self.cnum * 4, self.cnum * 4, 3, 2, 1)
        #--self.pool3

        # 16 * 16 * cnum
        self.conv4_1 = gen_conv(self.cnum * 4, self.cnum * 8, 3, 1, 1)
        self.conv4_2 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1)
        self.conv4_3 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 2, 1)
        #--self.pool4

        # 8 * 8 * cnum
        self.conv5_1 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1)
        self.conv5_2 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1)
        self.conv5_3 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 2, 1)
        #--self.pool5

        # 4 * 4 * cnum
        self.conv6_1 = gen_conv(self.cnum * 8, 4096, 3, 1, 1)


        #-- self.deconv6
        self.conv7_1  = gen_conv(4096, self.cnum * 8, 5, 1, 2)
        #-- dropout6

        # 8 * 8 * cnum
        #-- self.deconv7
        self.conv8_1 = gen_conv(self.cnum * 8, self.cnum * 8, 5, 1, 2)
        # -- dropout7

        # 16 * 16 * cnum
        #--self.deconv8
        self.conv9_1 = gen_conv(self.cnum * 8, self.cnum * 8, 5, 1, 2)
        # -- dropout8

        # 32 * 32 * cnum
        #-- self.deconv9
        self.conv10_1 = gen_conv(self.cnum * 8, self.cnum * 4, 5, 1, 2)
        # -- dropout9

        # 64 * 64 * cnum
        #--self.deconv10
        self.conv11_1 = gen_conv(self.cnum * 4, self.cnum * 2, 3, 1, 1)
        # -- dropout10

        # 128 * 128 * cnum
        #--self.deconv11
        self.conv12_1 = gen_conv(self.cnum * 2, self.cnum, 3, 1, 1)
        # -- dropout11

        # 128 * 128 * C+1 (FINAL)
        self.h_out = gen_conv(self.cnum, self.class_num, 3, 1, 1)

    def forward(self, x):

        # conv1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)


        # conv2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)


        # conv3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)

        # conv4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)

        # conv5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)

        # conv6
        x = self.conv6_1(x)
        #print(x.shape)

        # conv7
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv7_1(x)
        #print(x.shape)
        x = F.dropout2d(x)

        # conv8
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv8_1(x)
        #print(x.shape)

        # conv9
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv9_1(x)
        #print(x.shape)

        # conv10
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv10_1(x)
        #print(x.shape)

        # conv11
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv11_1(x)
        #print(x.shape)

        # conv12
        #x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv12_1(x)
        #print(x.shape)

        # h_out
        x_out = self.h_out(x)
        #print(x_out.shape)

        return x_out



def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, name='conv', weight_norm='sn', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x