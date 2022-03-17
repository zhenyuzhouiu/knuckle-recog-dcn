from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument('--src_npy', type=str, dest='src_npy', default='/home/zhenyuzhou/Desktop/Dissertataion/Finger-Knuckle/knuckle-recog-dcn/code/output/RFN-128/protocol3.npy')
parser.add_argument('--dest', type=str, dest='dest', default='/home/zhenyuzhou/Desktop/Dissertataion/Finger-Knuckle/knuckle-recog-dcn/code/output/hd(1-4)/protocol3_cmc.pdf')
parser.add_argument('--label', type=str, dest='label', default='RFN-128')
parser.add_argument('--save_cmc', type=bool, dest='save_cmc', default=False)
args = parser.parse_args()

if args.dest == '':
    args.dest = args.src_npy[:args.src_npy.find('.npy')] + "_cmc.pdf"

if args.label == '':
    args.label = args.src_npy

nobject = 4

src_npy = ['/home/zhenyuzhou/Desktop/Dissertataion/Finger-Knuckle/knuckle-recog-dcn/code/output/hd(1-4)/claknet/protocol3.npy',
           '/home/zhenyuzhou/Desktop/Dissertataion/Finger-Knuckle/knuckle-recog-dcn/code/output/hd(1-4)/dclaknet/protocol3.npy',
           '/home/zhenyuzhou/Desktop/Dissertataion/Finger-Knuckle/knuckle-recog-dcn/code/output/hd(1-4)/ctnet/protocol3.npy',
           '/home/zhenyuzhou/Desktop/Dissertataion/Finger-Knuckle/knuckle-recog-dcn/code/output/hd(1-4)/rfn/protocol3.npy']
label = ['CLAKNet',
         'DCLAKNet',
         'CTNet',
         'RFN-128']

color = ['#DC143C',
         '#0000FF',
         '#00FF00',
         '#FFA500']
dst = '/home/zhenyuzhou/Desktop/Dissertataion/Finger-Knuckle/knuckle-recog-dcn/code/output/RFN-128/protocol3_cmc.pdf'

for n in range(4):
    data = np.load(src_npy[n], allow_pickle=True)[()]
    match_dict = np.array(data['mmat'])
    nsamples = np.shape(match_dict)[0]

    genuine_idx = np.arange(nsamples).astype(np.float32)
    genuine_idx = np.expand_dims(np.floor(genuine_idx / nobject) * nobject, -1)

    min_idx = match_dict.argsort()

    def calc_cmc(rank):
        match_rank = min_idx[:, :rank]
        matching = []
        for j in range(nobject):
            genuine_tmp = np.repeat(genuine_idx + j, rank, 1)
            matching.append(np.sum((match_rank == genuine_tmp).astype(np.int8), 1))
        acc = reduce(lambda x, y: x + y, matching)
        acc = np.clip(acc, 0, 1)
        return np.sum(acc) / np.shape(match_dict)[0]


    x, y = [], []
    for i in range(1, 11):
        x.append(i)
        y.append(calc_cmc(i))

    print ("[*] Accuracy: {}".format(y[0]))

    if args.save_cmc:
        import scipy.io
        scipy.io.savemat(args.src_npy[:args.src_npy.find('.npy')] + "_cmc.mat", mdict={'r': x, 'ac': y})

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    lines = plt.plot(x, y, label='')
    plt.setp(lines, 'color', color[n], 'linewidth', 5, 'label', label[n])

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

if args.dest:
    plt.savefig(args.dest, bbox_inches='tight')
