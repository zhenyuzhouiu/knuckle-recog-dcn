# ========================================================= 
# .npy(python) format to .mat(MATLAB) format
# =========================================================

from __future__ import division
import numpy as np
import math
import argparse
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument('--src_npy', type=str, dest='src_npy', default='')
args = parser.parse_args()


data = np.load(args.src_npy)[()]
g_scores = np.array(data['g_scores'])
i_scores = np.array(data['i_scores'])

scipy.io.savemat(args.src_npy[:args.src_npy.find('.npy')] + ".mat", mdict={'g': np.array(g_scores), 'i': np.array(i_scores)})
