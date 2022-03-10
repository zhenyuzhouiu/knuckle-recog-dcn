# ========================================================= 
# @ Network Architecture File: Residual Feature Network 32
# @ Target dataset: CASIA
# @ Notes: This architecture is specially designed for CASIA
#   dataset. Due to its large variation and inconsistency
#   with training dataset IITD-Left, we enlarge its receptive
#   field at first with kernel_size 7. This architecture 
#   could also be used for other dataset. But the result is
#   not as good as use RFN-128.
# =========================================================

from __future__ import division
import sys
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from net_common import ConvLayer, ResidualBlock

class ResidualFeatureNet(torch.nn.Module):
    def __init__(self):
        super(ResidualFeatureNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 16, kernel_size=7, stride=2)
        self.conv2 = ConvLayer(16,24, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(24,32, kernel_size=3, stride=1)
        self.resid1= ResidualBlock(32)
        self.resid2= ResidualBlock(32)
        self.resid3= ResidualBlock(32)
        self.resid4= ResidualBlock(32)
        self.conv4 = ConvLayer(32, 16, kernel_size=3, stride=1)
        self.conv5 = ConvLayer(16, 1, kernel_size=3, stride=1)

    def forward(self, X):
        conv1 = F.relu(self.conv1(X))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1= self.resid1(conv3)
        resid2= self.resid2(resid1)
        resid3= self.resid3(resid2)
        resid4= self.resid4(resid3)
        conv4 = F.relu(self.conv4(resid4))
        conv5 = F.relu(self.conv5(conv4))
        return conv5
