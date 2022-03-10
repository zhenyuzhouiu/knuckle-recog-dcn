import os
import sys
import time

from inspect import getsourcefile
from os.path import join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, default="", dest="test_path")
parser.add_argument("--idx", type=int)
args = parser.parse_args()

def subfolders(src, preserve_prefix=False):
    if preserve_prefix:
        return [join(src, d) for d in os.listdir(src) if "." not in d]
    else:
        return [d for d in os.listdir(src) if "." not in d]

def subimages(src, preserve_prefix=False, ext=["JPG", "bmp", "jpg"]):
    def _hasext(f):
        for ext_ in ext:
            if ext_ in f:
                return True
        return False

    if preserve_prefix:
        return [join(src, f) for f in os.listdir(src) if _hasext(f)]
    else:
        return [f for f in os.listdir(src) if ext in f]

subs = subfolders(args.test_path, preserve_prefix=True)
nsubs = len(subs)
subimnames = []
for i, usr in enumerate(subs):
    subims = subimages(usr, preserve_prefix=True)
    subimnames += subims

print ("Image directory for {} is {}".format(args.idx, subimnames[args.idx]))
