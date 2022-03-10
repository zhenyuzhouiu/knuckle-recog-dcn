# ========================================================= 
# @ Protocol File: Two sessions (Probe / Gallery)
#
# @ Target dataset: 300-subject, 35-subject
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
    session1_path = join(test_path, "session1")
    session2_path = join(test_path, "session2")

    subs_session1 = subfolders(session1_path, preserve_prefix=True)
    subs_session1 = sorted(subs_session1)
    subs_session2 = subfolders(session2_path, preserve_prefix=True)
    subs_session2 = sorted(subs_session2)
    nsubs1 = len(subs_session1)
    nsubs2 = len(subs_session2)
    assert(nsubs1 == nsubs2 and nsubs1 != 0) 

    nsubs = nsubs1
    nims = -1
    feats_probe = []
    feats_gallery = []

    for gallery, probe in zip(subs_session1, subs_session2):
        assert(os.path.basename(gallery) == os.path.basename(probe))
        im_gallery = subimages(gallery, preserve_prefix=True)
        im_probe = subimages(probe, preserve_prefix=True)
        
        nim_gallery = len(im_gallery)
        nim_probe = len(im_probe)
        if nims == -1:
            nims =  nim_gallery
            assert(nims == nim_probe) # Check if image numbers in probe equals number in gallery
        else:
            assert(nims == nim_gallery and nims == nim_probe) # Check for each folder

        probe_fv = calc_feats_more(*im_probe)
        gallery_fv = calc_feats_more(*im_gallery)
        
        feats_probe.append(probe_fv)
        feats_gallery.append(gallery_fv)
    
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