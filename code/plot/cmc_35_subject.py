# ========================================================= 
# @ Plot File: 35 subject under two sessions
# =========================================================

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

nobjects = 5

genuine_idx = np.arange(nsamples).astype(np.float32)
genuine_idx = np.expand_dims(np.floor(genuine_idx / nobjects) * nobjects, -1)

min_idx = match_dict.argsort()

def calc_cmc(rank):
    match_rank = min_idx[:, :rank]
    matching = []
    for j in xrange(nobjects):
        genuine_tmp = np.repeat(genuine_idx + j, rank, 1)
        matching.append(np.sum((match_rank == genuine_tmp).astype(np.int8), 1))
    acc = reduce(lambda x, y: x + y, matching)
    acc = np.clip(acc, 0, 1)
    return np.sum(acc) / np.shape(match_dict)[0]

x, y = [], []
for i in xrange(1, 11):
    x.append(i)
    y.append(calc_cmc(i))
print y

print ("[*] Accuracy: {}".format(y[0]))

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

lines = plt.plot(x, y, label='')
plt.setp(lines, 'color', '#1F77B4', 'linewidth', 5, 'label', args.label)

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

plt.grid(True)
plt.xlabel(r'Rank', fontsize=18)
plt.ylabel(r'Recognition Rate', fontsize=18)
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
# plt.show()
