# ========================================================= 
# @ Plot File: 300 subject under two sessions
# =========================================================

from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse
from functools import reduce

# parser = argparse.ArgumentParser()
# parser.add_argument('--src_npy', type=str, dest='src_npy', default='')
# parser.add_argument('--dest', type=str, dest='dest', default='')
# parser.add_argument('--label', type=str, dest='label', default='')
# args = parser.parse_args()
#
# if args.dest == '':
#     args.dest = args.src_npy[:args.src_npy.find('.npy')] + "_cmc.pdf"
#
# if args.label == '':
#     args.label = args.src_npy
nobjects = [6, 6, 6, 6, 6]
src_npy = ['/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/two-session/fkv3/WRS-0.01-protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/two-session/fkv3/WRS-0.001-protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/two-session/fkv3/WS-O.O1-protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/two-session/fkv3/WS-0.001-protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/imageblockrotationandshifted/two-session/fkv3-rfn/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/wholeshifted/3d1s-rfn/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/wholeimagerotationandshifted/fkv3-efficientnet/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/wholeshifted/fkv3-rfn/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/RFN/HD/RFN-TOP14/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/RFN/HD/RFN-TOP16/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/RFN/HD/RFN/protocol3.npy']

label = ['WRS-0.01',
         'WRS-0.001',
         'WS-0.01',
         'WS-0.001',
         'IBRS',
         'RFN-128-WS',
         'EfficientNet-WRS',
         'RFN-WS',
         'RFN-128-14',
         'RFN-128-16',
         'RFN']

color = ['#000000',
         '#000080',
         '#008000',
         '#008080',
         "#c0c0c0",
         '#00ffff',
         '#800000',
         '#800080',
         '#808000',
         '#ff00ff',
         '#ff0000']
dst = '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/two-session/fkv3-ibrs-protocol3_cmc.pdf'

for n in range(4):
    data = np.load(src_npy[n], allow_pickle=True)[()]
    match_dict = np.array(data['mmat'])
    nsamples = np.shape(match_dict)[0]


    genuine_idx = np.arange(nsamples).astype(np.float32)
    genuine_idx = np.expand_dims(np.floor(genuine_idx / nobjects[n]) * nobjects[n], -1)

    min_idx = match_dict.argsort()

    def calc_cmc(rank):
        match_rank = min_idx[:, :rank]
        matching = []
        for j in range(nobjects[n]):
            genuine_tmp = np.repeat(genuine_idx + j, rank, 1)
            matching.append(np.sum((match_rank == genuine_tmp).astype(np.int8), 1))
        acc = reduce(lambda x, y: x + y, matching)
        acc = np.clip(acc, 0, 1)
        return np.sum(acc) / np.shape(match_dict)[0]

    x, y = [], []
    for i in range(1, 11):
        x.append(i)
        y.append(calc_cmc(i))
    print(y)

    print ("[*] Accuracy: {}".format(y[0]))

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    lines = plt.plot(x, y, label='')
    plt.setp(lines, 'color', color[n], 'linewidth', 1, 'label', label[n])

    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)

    plt.grid(True)
    plt.xlabel(r'Rank', fontsize=18)
    plt.ylabel(r'Recognition Rate', fontsize=18)
    legend = plt.legend(loc='lower right', shadow=False, prop={'size': 16})
    plt.xlim(xmin=1)
    plt.xlim(xmax=10)
    plt.ylim(ymax=1)
    plt.ylim(ymin=0.7)

    ax=plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)

    plt.xticks([2, 4, 6, 8, 10], fontsize=16)
    plt.yticks(np.array([0.7, 0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1]), fontsize=16)

if dst:
    plt.savefig(dst, bbox_inches='tight')
