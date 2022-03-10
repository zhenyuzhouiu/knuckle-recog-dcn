# ========================================================= 
# @ Plot File: CASIA under All-to-All protocol
# =========================================================

from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--src_npy', type=str, dest='src_npy', default='')
parser.add_argument('--dest', type=str, dest='dest', default='')
parser.add_argument('--label', type=str, dest='label', default='')
args = parser.parse_args()

if args.dest == '':
    args.dest = args.src_npy[:args.src_npy.find('.npy')] + "_cmc.pdf"

if args.label == '':
    args.label = args.src_npy

data = np.load(args.src_npy)[()]
match_dict = np.array(data['mmat'])
nsamples = np.shape(match_dict)[0]
feats_length = np.array(data['subject_len'])
acc_len = np.cumsum(feats_length)
feats_start = acc_len - feats_length

min_idx = match_dict.argsort()
print min_idx

def calc_cmc(rank):
    count = 0
    for id_ in xrange(nsamples):
        idx_this = np.argmax(acc_len > id_)
        for j in xrange(rank):
            idx = np.argmax(acc_len > min_idx[id_, j])
            if idx == idx_this:
                count+=1
                break
    return count / nsamples

x, y = [], []
for i in xrange(1, 11):
    x.append(i)
    y.append(calc_cmc(i))

print ("[*] Accuracy: {}".format(y[0]))

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

lines = plt.plot(x, y, label='')
plt.setp(lines, 'color', '#1F77B4', 'linewidth', 5, 'label', args.label)

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

plt.grid(True)
plt.xlabel(r'False Accept Rate', fontsize=18)
plt.ylabel(r'Genuine Accept Rate', fontsize=18)
legend = plt.legend(loc='lower right', shadow=False, prop={'size': 16})
plt.xlim(xmin=1)
plt.xlim(xmax=10)
plt.ylim(ymax=1)
plt.ylim(ymin=0.988)

ax=plt.gca()
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.0)

plt.xticks([2, 4, 6, 8, 10], fontsize=16)
plt.yticks(np.array([0.988, 0.99, 0.992, 0.994, 0.996, 0.998, 1]), fontsize=16)

if args.dest:
    plt.savefig(args.dest, bbox_inches='tight')
