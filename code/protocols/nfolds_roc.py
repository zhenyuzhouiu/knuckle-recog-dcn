# ========================================================= 
# @ Protocol File: All-to-All protocols
#
# @ Target dataset: CASIA
# @ Parameter Settings:
#       save_mmat:  whether save matching matrix or not,
#                   could be helpful for plot CMC
#
# =========================================================

from __future__ import division
import os
import sys
import time
from PIL import Image
import numpy as np
import torch
import argparse
from torch.autograd import Variable

from inspect import getsourcefile
import os.path as path 
from os.path import join
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

import net_common, netdef_128, netdef_32
from protocol_util import *

from torchvision import transforms
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.ToTensor()])


def calc_feats_more(*paths):
    container = np.zeros((len(paths), 1, args.default_size, args.default_size))
    for i, path in enumerate(paths):
        im = np.array(
            Image.open(path).convert("L").resize((args.default_size, args.default_size)),
            dtype=np.float32
            )
        container[i, 0, :, :] = im
    container /= 255.
    container = torch.from_numpy(container.astype(np.float32)).repeat(1, 3, 1, 1)
    container = container.cuda()
    container = Variable(container, requires_grad=False)
    fv = inference(container)
    return fv.cpu().data.numpy()

def genuine_imposter(test_path):
    subs = subfolders(test_path, preserve_prefix=True)
    feats_all = []
    feats_length = []
    nfeats = 0
    for i, usr in enumerate(subs):
        subims = subimages(usr, preserve_prefix=True, ext=["jpg"])
        nfeats += len(subims)
        feats_length.append(len(subims))
        feats_all.append(calc_feats_more(*subims))
    feats_length = np.array(feats_length)
    acc_len = np.cumsum(feats_length)
    feats_start = acc_len - feats_length

    feats_all = torch.from_numpy(np.concatenate(feats_all, 0)).cuda()
    matching_matrix = np.ones((nfeats, nfeats)) * 1e5
    for i in xrange(1, feats_all.size(0)):
        loss = _loss(feats_all[:-i, :, :, :], feats_all[i:, :, :, :])
        matching_matrix[:-i, i] = loss
        sys.stdout.write("[*] Pre-processing matching dict for {} / {} \r".format(i, feats_all.size(0)))
        sys.stdout.flush()

    mmat =  np.ones_like(matching_matrix) * 1e5
    mmat[0, :] = matching_matrix[0, :]
    for i in xrange(1, feats_all.size(0)):
        mmat[i, i:] = matching_matrix[i, :-i]
        for j in xrange(i):
            mmat[i, j] = matching_matrix[j, i - j]
    print ("\n [*] Done")

    g_scores = []
    i_scores = []
    for i in xrange(nfeats):
        subj_idx = np.argmax(acc_len > i)
        g_select = [feats_start[subj_idx] + k for k in xrange(feats_length[subj_idx])]
        g_select.remove(i)
        i_select = range(nfeats)
        for k in xrange(feats_length[subj_idx]):
            i_select.remove(feats_start[subj_idx] + k)
        g_scores += list(mmat[i, g_select])
        i_scores += list(mmat[i, i_select])
        
    print ("\n [*] Done")
    return np.array(g_scores), np.array(i_scores), feats_length, mmat

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, default="", dest="test_path")
parser.add_argument("--out_path", type=str, default="protocol3.npy", dest="out_path")
parser.add_argument("--model_path", type=str, default="", dest="model_path")
parser.add_argument("--default_size", type=int, dest="default_size", default=128)
parser.add_argument("--shift_size", type=int, dest="shift_size", default=16)
parser.add_argument("--save_mmat", type=bool, dest="save_mmat", default=False)

args = parser.parse_args()

if "RFN-32" in args.model_path:
    inference = netdef_32.ResidualFeatureNet()
elif "RFN-128" in args.model_path:
    inference = netdef_128.ResidualFeatureNet()
inference.load_state_dict(torch.load(args.model_path))
ShiftedLoss_ = net_common.ShiftedLoss(args.shift_size, args.shift_size)


def _loss(feats1, feats2):
    loss = ShiftedLoss_(feats1, feats2)
    if isinstance(loss, torch.autograd.Variable):
        loss = loss.data
    return loss.cpu().numpy()

inference = inference.cuda()
inference.eval()

# gscores = genuine_scores(args.test_path)
gscores, iscores, mmat = genuine_imposter(args.test_path)

if args.save_mmat:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores, "mmat": mmat})
else:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores})