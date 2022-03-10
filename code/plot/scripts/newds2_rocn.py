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
parser.add_argument('--compare_folder', type=str, dest='compare_folder', default='./plot/scripts/old_f2/')
parser.add_argument('--dest', type=str, dest='dest', default='')
parser.add_argument('--label', type=str, dest='label', default='npys')

args = parser.parse_args()

old_f = [d for d in os.listdir(args.compare_folder) if "." not in d]
assert(len(old_f) + 2 <= len(predefined_line_color))

def read_roc(f_path):
    roc_txt = join(f_path, "ROC.txt")
    x = []
    y = []
    with open(roc_txt, "r") as fp:
        for line in fp:
            xi, yi = line.strip().split('\t')
            if float(yi) == 0:
                continue
            x.append(math.log10(float(yi)))
            y.append(float(xi))
    return x, y

def calc_roc(npy_path):
    data = np.load(npy_path)[()]
    g_scores = np.array(data['g_scores'])
    i_scores = np.array(data['i_scores'])

    i_scores = sorted(i_scores)

    isize = len(i_scores)
    gsize = len(g_scores)
    x, y = [], []
    for i, threshold in enumerate(i_scores):
        if i % 10 != 0:
            continue
        x.append(math.log10((i + 1) / isize))
        y.append(len(g_scores[g_scores < threshold]) / gsize)

    for xi, yi in zip(x, y):
        if 1 - math.pow(10, xi) < yi:
            print ("[*] EER for {}: {}".format(npy_path, math.pow(10, xi)))
            break
    print ("[*] #Genuine: {}\n[*] #Imposter: {}".format(len(g_scores), len(i_scores)))
    return x, y

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

x1, y1 = calc_roc(args.src_npy)

lines = plt.plot(x1, y1, label='')
plt.setp(lines, 'color', predefined_line_color[0], 'linewidth', 4, 'label', args.label)

for k, old_fi in enumerate(old_f):
    xi, yi = read_roc(join(args.compare_folder, old_fi))
    lines = plt.plot(xi, yi, label='')
    plt.setp(lines, 'color', predefined_line_color[k + 1], 'linewidth', 4, 'label', old_fi)

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

plt.grid(True)
plt.xlabel(r'False Accept Rate', fontsize=18)
plt.ylabel(r'Genuine Accept Rate', fontsize=18)
legend = plt.legend(loc='lower right', shadow=False, prop={'size': 16})
plt.xlim(xmin=min(x1))
plt.xlim(xmax=0)
plt.ylim(ymax=1)
plt.ylim(ymin=0.98)


ax=plt.gca()
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.0)

plt.xticks(np.array([-5, -2, 0]), ['$10^{-5}$', '$10^{-2}$', '$10^{0}$'], fontsize=16)
plt.yticks(np.array([0.98, 0.984, 0.988, 0.992, 0.996, 1]), fontsize=16)

if args.dest:
    plt.savefig(args.dest, bbox_inches='tight')
# plt.show()
