# ========================================================= 
# @ Protocol File: Two sessions (Probe / Gallery) specially
#   designed for PolyU2D (first five for probe, second five
#   for gallery)
#
# @ Target dataset: PolyU 2D/3D contactless
# @ Parameter Settings:
#       save_mmat:  whether save matching matrix or not,
#                   could be helpful for plot CMC
# =========================================================

from __future__ import division
import os
import sys
import time
from PIL import Image
import numpy as np
import torch
import math
import argparse

from torch.autograd import Variable
from inspect import getsourcefile
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

import net_common, netdef_128, netdef_32
from protocol_util import *

from torchvision import transforms
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.ToTensor()])

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

def genuine_imposter(test_path):
    subs = subfolders(test_path, preserve_prefix=True)
    nsubs = len(subs)
    nims = 5
    feats_probe = []
    feats_gallery = []
    for i, usr in enumerate(subs):
        subims = subimages(usr, preserve_prefix=True)
        subims = sorted(subims)
        fvs = calc_feats_more(*subims)
        feats_probe.append(fvs[nims:, :, :, :])
        feats_gallery.append(fvs[:nims, :, :, :])
    feats_probe = torch.from_numpy(np.concatenate(feats_probe, 0)).cuda()
    feats_gallery = np.concatenate(feats_gallery, 0)
    feats_gallery2 = np.concatenate((feats_gallery, feats_gallery), 0)
    feats_gallery = torch.from_numpy(feats_gallery2).cuda()

    nl = nsubs * nims
    matching_matrix = np.ones((nl, nl)) * 1000000
    for i in xrange(nl):
        loss = _loss(feats_probe, feats_gallery[i: i + nl, :, :, :])
        matching_matrix[:, i] = loss
        sys.stdout.write("[*] Pre-processing matching dict for {} / {} \r".format(i, nl))
        sys.stdout.flush()
    
    for i in xrange(1, nl):
        tmp = matching_matrix[i, -i:].copy()
        matching_matrix[i, i:] = matching_matrix[i, :-i]
        matching_matrix[i, :i] = tmp
    print ("\n [*] Done")

    g_scores = []
    i_scores = []
    for i in xrange(nl):
        start_idx = int(math.floor(i / nims))
        start_remainder = int(i % nims)
        
        g_scores.append(float(np.min(matching_matrix[i, start_idx * nims: start_idx * nims + nims])))
        select = range(nl)
        for j in xrange(nims):
            select.remove(start_idx * nims + j)
        i_scores += list(np.min(np.reshape(matching_matrix[i, select], (-1, nims)), axis=1))
        sys.stdout.write("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        sys.stdout.flush()
    print ("\n [*] Done")
    return np.array(g_scores), np.array(i_scores), matching_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, default="", dest="test_path")
parser.add_argument("--out_path", type=str, default="protocol3.npy", dest="out_path")
parser.add_argument("--model_path", type=str, default="", dest="model_path")
parser.add_argument("--default_size", type=int, dest="default_size", default=128)
parser.add_argument("--shift_size", type=int, dest="shift_size", default=5)
parser.add_argument("--save_mmat", type=bool, dest="save_mmat", default=False)

args = parser.parse_args()
if "RFN-32" in args.model_path:
    inference = netdef_32.ResidualFeatureNet()
elif "RFN-128" in args.model_path:
    inference = netdef_128.ResidualFeatureNet()
else:
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

gscores, iscores, mmat = genuine_imposter(args.test_path)
if args.save_mmat:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores, "mmat": mmat})
else:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores})
