from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_npy', type=str, dest='src_npy', default='')
parser.add_argument('--nobject', type=int, dest='nobject', default=5)
args = parser.parse_args()

nobject = args.nobject
data = np.load(args.src_npy)[()]
match_dict = np.array(data['mmat'])
nsamples = np.shape(match_dict)[0]

genuine_idx = np.arange(nsamples).astype(np.float32)
genuine_idx = np.expand_dims(np.floor(genuine_idx / nobject) * nobject, -1)

min_idx = match_dict.argsort()
match_rank = min_idx[:, :1]
matching = []
for j in xrange(nobject):
    genuine_tmp = np.repeat(genuine_idx + j, 1, 1)
    matching.append(np.sum((match_rank == genuine_tmp).astype(np.int8), 1))
acc = reduce(lambda x, y: x + y, matching)
acc = np.clip(acc, 0, 1)

mismatch = np.where(acc == 0)[0]
mis_pairs = [(i, min_idx[i][0]) for i in mismatch]
print mis_pairs
