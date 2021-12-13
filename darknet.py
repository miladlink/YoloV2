import torch
import torch.nn as nn
import torch.nn.functional as F


def load_conv_bn (buf, start, conv_model, bn_model):
    """" load weights on conv & bn layers """
    num_w = conv_model.weight.numel ()
    num_b = bn_model.bias.numel ()
    bn_model.bias.data.copy_ (torch.from_numpy (buf [start: start + num_b])); start = start + num_b
    bn_model.weight.data.copy_ (torch.from_numpy (buf [start: start + num_b])); start = start + num_b
    bn_model.running_mean.copy_ (torch.from_numpy (buf [start: start + num_b])); start = start + num_b
    bn_model.running_var.copy_ (torch.from_numpy (buf [start: start + num_b])); start = start + num_b
    conv_model.weight.data.copy_ (torch.from_numpy (buf [start: start + num_w]).reshape_as (conv_model.weight)); start = start + num_w
    return start


def load_conv (buf, start, conv_model):
    """ load weights on conv layer """
    num_w = conv_model.weight.numel ()
    num_b = conv_model.bias.numel ()
    conv_model.bias.data.copy_ (torch.from_numpy (buf [start: start + num_b])); start = start + num_b
    conv_model.weight.data.copy_ (torch.from_numpy (buf [start: start + num_w]).reshape_as (conv_model.weight)); start = start + num_w


class Conv2D (nn.Module):
    """ Conv2D + BatchNormalization + LeakyReLU + MaxPool2d """
    def __init__ (self, in_channels, out_channels, kernel_size, stride = 1, padding = True, activation = True, pooling = False):
        super (Conv2D, self).__init__ ()
        padding = (kernel_size - 1) // 2 if padding else 0
        self.conv = nn.Conv2d (in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.bn   = nn.BatchNorm2d (out_channels)
        self.relu = nn.LeakyReLU (0.1, inplace = True) if activation else lambda x : x
        self.pooling = pooling
 
    def forward (self, x):
        return F.max_pool2d (self.relu (self.bn (self.conv (x))), 2, 2) if self.pooling else self.relu (self.bn (self.conv (x)))


def Reorg (x, s = 2):
    B, C, H, W = x.size ()
    h, w = H // s, W // s
    x = x.view (B, C, h, s, w, s).transpose (3, 4).contiguous ()
    x = x.view (B, C, h * w, s * s).transpose (2, 3).contiguous ()
    x = x.view (B, C, s * s, h, w).transpose (1, 2).contiguous ()
    return x.view (B, s * s * C, h, w)


class Darknet (nn.Module):
    def __init__ (self):
        super (Darknet, self).__init__ ()
      
        self.main1 = nn. Sequential (     #input = (3, 608, 608)
            Conv2D (3, 32, 3, pooling = True),
 
            Conv2D (32, 64, 3, pooling = True),
 
            Conv2D (64, 128, 3),
            Conv2D (128, 64, 1),
            Conv2D (64, 128, 3, pooling = True),
 
            Conv2D (128, 256, 3),
            Conv2D (256, 128, 1),
            Conv2D (128, 256, 3, pooling = True),
 
            Conv2D (256, 512, 3),
            Conv2D (512, 256, 1),
            Conv2D (256, 512, 3),
            Conv2D (512, 256, 1),
            Conv2D (256, 512, 3))
 
        self.main2 = nn.Sequential (
            nn.MaxPool2d (2, stride = 2),
            Conv2D (512, 1024, 3),
            Conv2D (1024, 512, 1),
            Conv2D (512, 1024, 3),
            Conv2D (1024, 512, 1),
            Conv2D (512, 1024, 3))
 
    def forward (self, x):
        x1 = self.main1 (x)
        x2 = self.main2 (x1)
        return x2, x1