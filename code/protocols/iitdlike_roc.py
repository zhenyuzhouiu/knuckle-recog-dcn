# ========================================================= 
# @ Protocol File: IITD-like protocol (for the protocol
#   detail, please check [1])
#
# @ Target dataset: IITD Right, 600-subject
# @ Parameter Settings:
#       save_mmat:  whether save matching matrix or not,
#                   could be helpful for plot CMC
#
# @ Notes:  also could be used on PolyU 2D if format the
#           dataset like "session1" and "session2" under 
#           PolyU 2D data folder
# =========================================================

from __future__ import division
import os
import sys
import time
from PIL import Image
import numpy as np
import math
import torch
import argparse
import torchvision
from torch.autograd import Variable

from inspect import getsourcefile
import os.path as path 
from os.path import join
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import net_common, netdef_32, netdef_128
import efficientnet
from protocol_util import *

from torchvision import transforms
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.ToTensor()])


def calc_feats(path):
    """
    1.Read a image from given a path
    2.Normalize image from 0-255 to 0-1
    3.Get the feature map from the model with inference()
    """
    container = np.zeros((1, 3, args.default_size, args.default_size))
    im = np.array(
            Image.open(path).convert("RGB").resize((args.default_size, args.default_size)),
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
    """
    1.Read a batch of images from the given paths
    2.Normalize image from 0-255 to 0-1
    3.Get a batch of feature from the model inference()
    """
    container = np.zeros((len(paths), 3, args.default_size, args.default_size))
    for i, path in enumerate(paths):
        im = np.array(
            Image.open(path).convert('RGB').resize((args.default_size, args.default_size)),
            dtype=np.float32
            )
        # change hxwxc = cxhxw
        im = np.transpose(im, (2, 0, 1))
        container[i, :, :, :] = im
    container /= 255.
    container = torch.from_numpy(container.astype(np.float32))
    container = container.cuda()
    container = Variable(container, requires_grad=False)
    fv = inference(container)
    # traced_script_module = torch.jit.trace(inference, container)
    # traced_script_module.save("traced_450.pt")

    return fv.cpu().data.numpy()


def genuine_imposter(test_path):

    subs = subfolders(test_path, preserve_prefix=True)
    nsubs = len(subs)
    feats_all = []
    subimnames = []
    for i, usr in enumerate(subs):
        subims = subimages(usr, preserve_prefix=True)
        subimnames += subims
        nims = len(subims)
        feats_all.append(calc_feats_more(*subims))
    feats_all = torch.from_numpy(np.concatenate(feats_all, 0)).cuda()

    # nsubs-> how many subjects on the test_path
    # nims-> how many images on each of subjects' path
    # for example, for hd(1-4), matching_matrix.shape = (714x4, 714x4)
    matching_matrix = np.ones((nsubs * nims, nsubs * nims)) * 1000000
    for i in range(1, feats_all.size(0)):
        feat1 =feats_all[:-i, :, :, :]
        feat2 = feats_all[i:, :, :, :]
        # loss = _loss(feats_all[:-i, :, :, :], feats_all[i:, :, :, :])
        loss = _loss(feat1, feat2)
        matching_matrix[:-i, i] = loss
        print("[*] Pre-processing matching dict for {} / {} \r".format(i, feats_all.size(0)))
        # sys.stdout.write("[*] Pre-processing matching dict for {} / {} \r".format(i, feats_all.size(0)))
        # sys.stdout.flush()
    
    matt =  np.ones_like(matching_matrix) * 1000000
    matt[0, :] = matching_matrix[0, :]
    for i in range(1, feats_all.size(0)):
        # matching_matrix每行的数值向后移动一位
        matt[i, i:] = matching_matrix[i, :-i]
        for j in range(i):
            # matt[i, j] = matt[j, i]
            matt[i, j] = matching_matrix[j, i - j]

    # matt is the matching score of
    #   1  A1A2 A1A3 A1A4
    # A2A1  1   A2A3 A2A4
    # A3A1 A3A2  1   A3A4
    # A4A1 A4A2 A4A3  1
    print ("\n [*] Done")

    g_scores = []
    i_scores = []
    for i in range(nsubs * nims):
        start_idx = int(math.floor(i / nims))
        start_remainder = int(i % nims)
        
        argmin_idx = np.argmin(matt[i, start_idx * nims: start_idx * nims + nims])
        g_scores.append(float(matt[i, start_idx * nims + argmin_idx]))
        select = list(range(nsubs * nims))
        # remove genuine matching score
        for j in range(nims):
            select.remove(start_idx * nims + j)
        # remove imposter matching scores of same index sample on other subjects
        for j in range(nsubs):
            if j == start_idx:
                continue
            select.remove(j * nims + start_remainder)
        i_scores += list(np.min(np.reshape(matt[i, select], (-1, nims - 1)), axis=1))
        print("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        # sys.stdout.write("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        # sys.stdout.flush()
    print ("\n [*] Done")
    return np.array(g_scores), np.array(i_scores), matt

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, default="/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV3/Segmented/Session_1_128/1-104/", dest="test_path")
parser.add_argument("--out_path", type=str, default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/imageblockrotationandshifted/fkv3-rfn/d3-a5-topk12-protocol3.npy", dest="out_path")
parser.add_argument("--model_path", type=str, default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/checkpoint/fkv3(session_2_1_104)_mRFN-128-stimageblockwithgradient-losstriplet-lr0.001-subd3-subs8-angle5-a20-nna40-s3_2022-04-28-20-43/ckpt_epoch_1000.pth", dest="model_path")
parser.add_argument("--default_size", type=int, dest="default_size", default=128)
parser.add_argument("--shift_size", type=int, dest="shift_size", default=3)
parser.add_argument('--dilation_size', type=int, dest="dilation", default=3)
parser.add_argument('--subpatch_size', type=int, dest="subsize", default=8)
parser.add_argument("--rotate_angle", type=int, dest="angle", default=5)
parser.add_argument("--save_mmat", type=bool, dest="save_mmat", default=True)

args = parser.parse_args()
if "RFN-128" in args.model_path:
    inference = netdef_128.ResidualFeatureNet()
else:
    if "DeConvRFNet" in args.model_path:
        inference = netdef_128.DeConvRFNet()
    elif "EfficientNet" in args.model_path:
        inference = efficientnet.EfficientNet(width_coefficient=1, depth_coefficient=1, dropout_rate=0.2)



inference.load_state_dict(torch.load(args.model_path))
# inference = torch.jit.load("knuckle-script-polyu.pt")
# Loss = net_common.ShiftedLoss(args.shift_size, args.shift_size)
# Loss = net_common.SubShiftedLoss(args.dilation, args.subsize, topk=16)
# Loss = net_common.RIPShiftedLoss(args.dilation, args.subsize, args.angle, topk=16)
# Loss = net_common.RANDIPShiftedLoss(dilation=args.dilation, subsize=args.subsize, angle=args.angle, topk=16)
# Loss = net_common.WholeRotationShiftedLoss(args.shift_size, args.shift_size, args.angle)
Loss = net_common.ImageBlockRotationAndTranslation(i_block_size=args.subsize, i_v_shift=args.dilation, i_h_shift=args.dilation, i_angle=args.angle, i_topk=12)
Loss.cuda()
Loss.eval()
def _loss(feats1, feats2):
    loss = Loss(feats1, feats2)
    if isinstance(loss, torch.autograd.Variable):
        loss = loss.data
    return loss.cpu().numpy()

inference = inference.cuda()
inference.eval()

gscores, iscores, mmat = genuine_imposter(args.test_path)
if args.save_mmat:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores, "mmat": mmat})
else:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores})