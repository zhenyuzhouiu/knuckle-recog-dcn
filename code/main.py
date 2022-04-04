# ========================================================= 
# @ Main File for PolyU Project: Online Contactless Palmprint
#   Identification using Deep Learning
# =========================================================

import argparse
import os
import shutil
import scipy.misc
import datetime
from model import Model
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def build_parser():
    parser = argparse.ArgumentParser()

    # Checkpoint Options
    parser.add_argument('--logdir', type=str, dest='logdir', default='./runs')

    parser.add_argument('--checkpoint_dir', type=str,
                        dest='checkpoint_dir', default='./checkpoint/')
    parser.add_argument('--db_prefix', dest='db_prefix', default='fkv3')
    parser.add_argument('--checkpoint_interval', type=int, dest='checkpoint_interval',
                        default=20)
    
    # Dataset Options
    parser.add_argument('--train_path', type=str, dest='train_path', default='/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/Database/Segmented/Session_2_128')

    # Training Strategy
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=8)
    parser.add_argument('--epochs', type=int, dest='epochs', default=3000)
    parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=1e-3)
    
    # Training Logging Interval
    parser.add_argument('--log_interval', type=int, dest='log_interval', default=1)
    # Pre-defined Options
    parser.add_argument('--shifttype', type=str, dest='shifttype', default='shifted')
    parser.add_argument('--losstype', type=str, dest='losstype', default='triplet')
    parser.add_argument('--alpha', type=float, dest='alpha', default=100)
    parser.add_argument('--nnalpha', type=float, dest='nnalpha', default=40)
    parser.add_argument('--model', type=str, dest='model', default="RFN-128")
    parser.add_argument('--shifted_size', type=int, dest='shifted_size', default=3)
    parser.add_argument('--dilation_size', type=int, dest="dilation", default=3)
    parser.add_argument('--subpatch_size', type=int, dest="subsize", default=8)
    parser.add_argument('--rotate_angle', type=int, dest="angle", default=5)

    # fine-tuning
    parser.add_argument('--start_ckpt', type=str, dest='start_ckpt', default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/checkpoint/FKV1_a10s3mRFN-128_2022-03-10-23-03-41/ckpt_epoch_1280.pth")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    this_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    args.checkpoint_dir = os.path.join(
        args.checkpoint_dir,
        "{}_m{}-st{}-loss{}-lr{}-subd{}-subs{}-angle{}-a{}-nna{}-s{}_{}".format(
            args.db_prefix,
            args.model,
            args.shifttype,
            args.losstype,
            float(args.learning_rate),
            int(args.dilation),
            int(args.subsize),
            int(args.angle),
            int(args.alpha),
            int(args.nnalpha),
            int(args.shifted_size),
            this_datetime
        )
    )

    args.logdir = os.path.join(
        args.logdir,
        "{}_m{}-st{}-loss{}-lr{}-subd{}-subs{}-angle{}-a{}-nna{}-s{}_{}".format(
            args.db_prefix,
            args.model,
            args.shifttype,
            args.losstype,
            float(args.learning_rate),
            int(args.dilation),
            int(args.subsize),
            int(args.angle),
            int(args.alpha),
            int(args.nnalpha),
            int(args.shifted_size),
            this_datetime
        )

    )

    print("[*] Target Checkpoint Path: {}".format(args.checkpoint_dir))
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    print("[*] Target Logdir Path: {}".format(args.logdir))
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    writer = SummaryWriter(log_dir=args.logdir)
    model_ = Model(args, writer=writer)
    if args.losstype == "triplet":
        model_.triplet_train(args)
    else:
        model_.quadruplet_train(args)


if __name__ == "__main__":
    main()
