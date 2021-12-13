import torch
import torch.nn as nn
import numpy as np

from darknet import *





class YoloV2Net (nn.Module):
    def __init__ (self, num_anchors = 5, num_classes = 80):
        super (YoloV2Net, self).__init__ ()
        self.num_classes = num_classes
        self.anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88882, 3.52778, 9.77052, 9.16888]
        self.num_anchors = num_anchors
        self.darknet = Darknet ()
     
        self.conv1 = nn.Sequential (
             Conv2D (1024, 1024, 3),
             Conv2D (1024, 1024, 3))
     
        self.conv2 = nn.Sequential (
             Conv2D (512, 64, 1))
     
        self.conv  = nn.Sequential (
             Conv2D (1280, 1024, 3),
             nn.Conv2d (1024, self.num_anchors * (self.num_classes + 5), 1))
     
    def forward (self, x):
        x1, x2 = self.darknet (x)
        x1 = self.conv1 (x1)
        x2 = self.conv2 (x2)
        x2 = Reorg (x2)
        x = torch.cat ([x2, x1], 1)
        x = self.conv (x)
        return x


def load_weights (model, wt_file):
    """ load weights from .weights file """
    buf = np.fromfile (wt_file, dtype = np.float32)
    start = 4   
    for i in range (13):
        bn   = model.darknet.main1 [i].bn
        conv = model.darknet.main1 [i].conv
        start = load_conv_bn (buf, start, conv, bn)
    for i in range (1, 6):
        bn   = model.darknet.main2 [i].bn
        conv = model.darknet.main2 [i].conv
        start = load_conv_bn (buf, start, conv, bn)
    for i in range (2):
        start = load_conv_bn (buf, start, model.conv1 [i].conv, model.conv1 [i].bn)
 
    start = load_conv_bn (buf, start, model.conv2 [0].conv, model.conv2 [0].bn)
    start = load_conv_bn (buf, start, model.conv  [0].conv, model.conv  [0].bn)
    start = load_conv (buf, start, model.conv [1])


def load_model (weights, device):
    """ load model and it's weights """
    model = YoloV2Net ()
    if weights:
        load_weights (model, weights)
        #model.load_state_dict (torch.load (weights))
    return model.to(device)