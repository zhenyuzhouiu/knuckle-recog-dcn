from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse

import os
from os.path import join

predefined_line_color = ['#FF7E0E', '#1F77B4', '#2E774B', '#4c1130', '#cc4125', '#ff33cc']

parser = argparse.ArgumentParser()
parser.add_argument('--src_npy', type=str, dest='src_npy', default='')
parser.add_argument('--compare_folder', type=str, dest='compare_folder', default='./plot/scripts/old_f3/')
parser.add_argument('--dest', type=str, dest='dest', default='')
parser.add_argument('--label', type=str, dest='label', default='npys')

args = parser.parse_args()

old_f = [d for d in os.listdir(args.compare_folder) if "." not in d]
assert(len(old_f) + 2 <= len(predefined_line_color))

def read_cmc(f_path):
    roc_txt = join(f_path, "CMC.txt")
    y = []
    with open(roc_txt, "r") as fp:
        for line in fp:
            yi = line.strip()
            y.append(float(yi))
    return y

nobject = 10
def calc_cmc(npy_path):
    data = np.load(npy_path)[()]
    match_dict = np.array(data['mmat'])
    nsamples = np.shape(match_dict)[0]

    genuine_idx = np.arange(nsamples).astype(np.float32)
    genuine_idx = np.expand_dims(np.floor(genuine_idx / nobject) * nobject, -1)

    min_idx = match_dict.argsort()

    def calc_cmc_(rank):
        match_rank = min_idx[:, :rank]
        matching = []
        for j in xrange(nobject):
            genuine_tmp = np.repeat(genuine_idx + j, rank, 1)
            matching.append(np.sum((match_rank == genuine_tmp).astype(np.int8), 1))
        acc = reduce(lambda x, y: x + y, matching)
        acc = np.clip(acc, 0, 1)
        return np.sum(acc) / np.shape(match_dict)[0]

    x, y = [], []
    for i in xrange(1, 11):
        x.append(i)
        y.append(calc_cmc_(i))

    print ("[*] Accuracy for {}: {}".format(npy_path, y[0]))
    return x, y

x1, y1 = calc_cmc(args.src_npy)

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

lines = plt.plot(x1, y1, label='')
plt.setp(lines, 'color', predefined_line_color[0], 'linewidth', 4, 'label', args.label)

for k, old_fi in enumerate(old_f):
    yi = read_cmc(join(args.compare_folder, old_fi))
    lines = plt.plot(x1, yi[:10], label='')
    plt.setp(lines, 'color', predefined_line_color[k + 1], 'linewidth', 4, 'label', old_fi)

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

plt.grid(True)
plt.xlabel(r'Rank', fontsize=18)
plt.ylabel(r'Accuracy Rate', fontsize=18)
legend = plt.legend(loc='lower right', shadow=False, prop={'size': 16})
plt.xlim(xmin=1)
plt.xlim(xmax=10)
plt.ylim(ymax=1)
plt.ylim(ymin=0.2)

ax=plt.gca()
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.0)

plt.xticks([2, 4, 6, 8, 10], fontsize=16)
plt.yticks(np.array([0.2, 0.4, 0.6, 0.8, 1]), fontsize=16)

if args.dest:
    plt.savefig(args.dest, bbox_inches='tight')
