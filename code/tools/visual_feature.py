# ========================================================= 
# @ Visualize 32x32 feature output of specific input image
#   with index.
# =========================================================

from __future__ import division
import os
import sys
import time
from PIL import Image
import numpy as np
import math
import torch
from torch.autograd import Variable

from inspect import getsourcefile
import os.path as path 
from os.path import join
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

import net_common, netdef_32, netdef_128

from torchvision import transforms
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.ToTensor()])

import argparse

def mkdir_if_not_exists(*args):
    for arg in args:
        if not os.path.exists(arg):
            os.mkdir(arg)

def subfolders(src, preserve_prefix=False):
    if preserve_prefix:
        return [join(src, d) for d in os.listdir(src) if "." not in d]
    else:
        return [d for d in os.listdir(src) if "." not in d]

def subimages(src, preserve_prefix=False, ext=["JPG", "bmp", "jpg"]):
    def _hasext(f):
        for ext_ in ext:
            if ext_ in f:
                return True
        return False

    if preserve_prefix:
        return [join(src, f) for f in os.listdir(src) if _hasext(f)]
    else:
        return [f for f in os.listdir(src) if ext in f]

def calc_feats(path):
    container = np.zeros((1, 1, args.default_size, args.default_size))
    im = np.array(
            Image.open(path).convert("L").resize((args.default_size, args.default_size)),
            dtype=np.float32
            )
    container[0, 0, :, :] = im
    container /= 255.
    container = torch.from_numpy(container.astype(np.float32))
    container = container.cuda()
    container = Variable(container, requires_grad=False)
    fv = inference(container)
    return fv.cpu().data.numpy()

def calc_feats_more(*paths):
    container = np.zeros((len(paths), 1, args.default_size, args.default_size))
    for i, path in enumerate(paths):
        im = np.array(
            Image.open(path).convert("L").resize((args.default_size, args.default_size)),
            dtype=np.float32
            )
        container[i, 0, :, :] = im
    container /= 255.
    container = torch.from_numpy(container.astype(np.float32))
    container = container.cuda()
    container = Variable(container, requires_grad=False)
    fv = inference(container)
    return fv.cpu().data.numpy()

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, default="", dest="test_path")
parser.add_argument("--idx", type=int, dest='idx')
parser.add_argument("--out_path", type=str, default="", dest="out_path")
parser.add_argument("--model_path", type=str, default="", dest="model_path")
parser.add_argument("--default_size", type=int, dest="default_size", default=128)
parser.add_argument("--loss", type=str, dest="loss", default="etl")
parser.add_argument("--shift_size", type=int, dest="shift_size", default=5)

args = parser.parse_args()

if "RFN-32" in args.model_path:
    inference = netdef_32.ResidualFeatureNet()
elif "RFN-128" in args.model_path:
    inference = netdef_128.ResidualFeatureNet()
inference.load_state_dict(torch.load(args.model_path))
ShiftedLoss_ = net_common.ShiftedLoss(args.shift_size, args.shift_size)

if args.loss.lower() == "etl":
    def _loss(feats1, feats2):
        loss = ShiftedLoss_(feats1, feats2)
        if isinstance(loss, torch.autograd.Variable):
            loss = loss.data
        return loss.cpu().numpy()

inference = inference.cuda()
inference.eval()

subs = subfolders(args.test_path, preserve_prefix=True)
nsubs = len(subs)
feats_all = []
subimnames = []
for i, usr in enumerate(subs):
    subims = subimages(usr, preserve_prefix=True)
    subimnames += subims
    nims = len(subims)
    feats_all.append(calc_feats_more(*subims))
feats_all = np.concatenate(feats_all, 0)
target_feats = np.squeeze(feats_all[args.idx, :, :, :])

import scipy.io
scipy.io.savemat(args.out_path, mdict={'f': target_feats})