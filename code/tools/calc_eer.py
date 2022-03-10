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
args = parser.parse_args()

data = np.load(args.src_npy)[()]
g_scores = np.array(data['g_scores'])

i_scores = np.array(data['i_scores'])

i_scores = sorted(i_scores)
isize = len(i_scores)
gsize = len(g_scores)
x, y = [], []
for i, threshold in enumerate(i_scores):
    x.append(math.log10((i + 1) / isize))
    y.append(len(g_scores[g_scores < threshold]) / gsize)

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

print ('[*] Source file: {}'.format(args.src_npy))

for xi, yi in zip(x, y):
    if 1 - math.pow(10, xi) < yi:
        print ("[*] EER: {}".format(math.pow(10, xi)))
        break
print ("[*] #Genuine: {}\n[*] #Imposter: {}\n".format(len(g_scores), len(i_scores)))
