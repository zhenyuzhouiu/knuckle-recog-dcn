# ========================================================= 
# @ Plot File: IITD under IITD-like protocol
# =========================================================

from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

from plotroc_basic import *

parser = argparse.ArgumentParser()
parser.add_argument('--src_npy', type=str, dest='src_npy', default='/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/tnet_16/protocol3.npy')
parser.add_argument('--dest', type=str, dest='dest', default='/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/tnet_16/protocol3_roc.pdf')
parser.add_argument('--label', type=str, dest='label', default='RFN-128')

args = parser.parse_args()

if args.dest == '':
    args.dest = args.src_npy[:args.src_npy.find('.npy')] + "_roc.pdf"

if args.label == '':
    args.label = args.src_npy


src_npy = ['/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/tnet_16/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/ctnet/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/dclka/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/rfn-128/protocol3.npy']
label = ["TNet-16",
         'CTNet',
         'DCLKANet',
         'RFN-128']

color = ['#DC143C',
         '#0000FF',
         '#00FF00',
         '#FFA500']
dst = '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/protocols.pdf'

for i in range(4):
    data = np.load(src_npy[i], allow_pickle=True)[()]
    g_scores = np.array(data['g_scores'])
    i_scores = np.array(data['i_scores'])

    print ('[*] Source file: {}'.format(args.src_npy))
    print ('[*] Target output file: {}'.format(args.dest))
    print ("[*] #Genuine: {}\n[*] #Imposter: {}".format(len(g_scores), len(i_scores)))

    x, y = calc_coordinates(g_scores, i_scores)
    print ("[*] EER: {}".format(calc_eer(x, y)))

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    lines = plt.plot(x, y, label='')
    plt.setp(lines, 'color', color[i], 'linewidth', 5, 'label', label[i])

    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)

    plt.grid(True)
    plt.xlabel(r'False Accept Rate', fontsize=18)
    plt.ylabel(r'Genuine Accept Rate', fontsize=18)
    legend = plt.legend(loc='lower right', shadow=False, prop={'size': 16})
    plt.xlim(xmin=min(x))
    plt.xlim(xmax=0)
    plt.ylim(ymax=1)
    plt.ylim(ymin=0)

    ax=plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)

    plt.xticks(np.array([-4, -2, 0]), ['$10^{-4}$', '$10^{-2}$', '$10^{0}$'], fontsize=16)
    plt.yticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]), fontsize=16)

if args.dest:
    plt.savefig(args.dest, bbox_inches='tight')