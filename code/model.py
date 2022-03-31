from __future__ import division
import os
import sys
import time
import numpy as np
import torch
import torchvision.utils
from torch.autograd import Variable
from torch.optim import Adam

import netdef_128, netdef_32, net_common

from data_factory import Factory
import util
import json

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Model(object):
    def __init__(self, args, writer):
        self.writer = writer
        self.batch_size = args.batch_size
        self.train_loader, self.dataset_size = self._build_dataset_loader(args)

        if args.shifttype == "shifted":
            self.inference, self.loss = self._shift_model(args)
        else:
            self.inference, self.loss = self._subshift_model(args)

        self.optimizer = torch.optim.Adagrad(self.inference.parameters(), args.learning_rate)


    def _build_dataset_loader(self, args):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = Factory(args.train_path, transform=transform, valid_ext=['.bmp', '.jpg', '.JPG'], train=True, losstype=args.losstype)
        util.Logging("Successfully Load {} as training dataset...".format(args.train_path))
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        examples = iter(train_loader)
        example_data, example_target = examples.next()
        example_anchor = example_data[:, 0:3, : , :]
        example_positive = example_data[:, 3:6, :, :]
        example_negative = example_data[:, 6:9, :, :]
        anchor_grid = torchvision.utils.make_grid(example_anchor)
        self.writer.add_image(tag="anchor", img_tensor=anchor_grid)
        positive_grid = torchvision.utils.make_grid(example_positive)
        self.writer.add_image(tag="positive", img_tensor=positive_grid)
        negative_grid = torchvision.utils.make_grid(example_negative)
        self.writer.add_image(tag="negative", img_tensor=negative_grid)

        return train_loader, len(train_dataset)

    def exp_lr_scheduler(self, epoch, lr_decay=0.5, lr_decay_epoch=100):
        if epoch % lr_decay_epoch == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_decay

    def _subshift_model(self, args):
        if args.model == "RFN-32":
            inference = netdef_32.ResidualFeatureNet()
        else:
            if args.model == "RFN-128":
                inference = netdef_128.ResidualFeatureNet()
            elif args.model == "TNet_16":
                inference = netdef_128.TNet_16()
            else:
                if args.model == "TNet_8":
                    inference = netdef_128.TNet_8()
                elif args.model == "CTNet":
                    inference = netdef_128.CTNet()
                else:
                    if args.model == "DCLAKNet":
                        inference = netdef_128.DCLAKNet()
                    elif args.model == "CLAKNet":
                        inference = netdef_128.CLAKNet()
                    else:
                        if args.model == "DeepCLAKNet":
                            inference = netdef_128.DeepCLAKNet()
                        elif args.model == "MultiCLAKNet":
                            inference = netdef_128.MultiCLAKNet()

        examples = iter(self.train_loader)
        example_data, example_target = examples.next()
        data = example_data.view(-1, 3, example_data.size(2), example_data.size(3))
        self.writer.add_graph(inference, data[0,:,:,:].unsqueeze(0))

        loss = net_common.SubShiftedLoss(args.dilation, args.subsize)
        util.Logging("Successfully building sub-shifted triplet loss")
        inference.cuda()
        return inference, loss


    def _shift_model(self, args):
        if args.model == "RFN-32":
            inference = netdef_32.ResidualFeatureNet()
        else:
            if args.model == "RFN-128":
                inference = netdef_128.ResidualFeatureNet()
            elif args.model == "TNet_16":
                inference = netdef_128.TNet_16()
            else:
                if args.model == "TNet_8":
                    inference = netdef_128.TNet_8()
                elif args.model == "CTNet":
                    inference = netdef_128.CTNet()
                else:
                    if args.model == "DCLAKNet":
                        inference = netdef_128.DCLAKNet()
                    elif args.model == "CLAKNet":
                        inference = netdef_128.CLAKNet()
                    else:
                        if args.model == "DeepCLAKNet":
                            inference = netdef_128.DeepCLAKNet()
                        elif args.model == "MultiCLAKNet":
                            inference = netdef_128.MultiCLAKNet()

        examples = iter(self.train_loader)
        example_data, example_target = examples.next()
        data = example_data.view(-1, 3, example_data.size(2), example_data.size(3))
        self.writer.add_graph(inference, data[0,:,:,:].unsqueeze(0))

        loss = net_common.ShiftedLoss(args.shifted_size, args.shifted_size)
        util.Logging("Successfully building shifted triplet loss")
        inference.cuda()
        return inference, loss

    def quadruplet_train(self, args):
        epoch_steps = len(self.train_loader)
        train_loss = 0
        start_epoch = ''.join(x for x in os.path.basename(args.start_ckpt) if x.isdigit())
        if start_epoch:
            start_epoch = int(start_epoch) + 1
            self.load(args.start_ckpt)
        else:
            start_epoch = 1

        for e in range(start_epoch, args.epochs + start_epoch):
            self.exp_lr_scheduler(e, lr_decay_epoch=300)
            self.inference.train()
            agg_loss = 0.
            count = 0
            for batch_id, (x, _) in enumerate(self.train_loader):
                count += len(x)
                self.optimizer.zero_grad()
                x = x.cuda()

                x = Variable(x, requires_grad=False)

                if "RFN-32" in args.model:
                    # the RFN-32 will use the 3 channels image as the input image
                    fms = self.inference(x.view(-1, 3, x.size(2), x.size(3)).repeat(1, 3, 1, 1))
                else:
                    fms = self.inference(x.view(-1, 3, x.size(2), x.size(3)))

                fms = fms.view(x.size(0), -1, fms.size(2), fms.size(3))

                anchor_fm = fms[:, 0, :, :].unsqueeze(1)
                pos_fm = fms[:, 1, :, :].unsqueeze(1)
                nn_pair = int((fms.size(1) - 2)/2)
                if nn_pair == 1:
                    neg_fm = fms[:, 2, :, :].unsqueeze(1)
                    neg_neg = fms[:, 3, :, :].unsqueeze(1)
                else:
                    neg_fm = fms[:, 2:2+nn_pair, :, :].contiguous()
                    neg_neg = fms[:, 2+nn_pair:, :, :].contiguous()


                nneg = neg_fm.size(1)
                neg_fm = neg_fm.view(-1, 1, neg_fm.size(2), neg_fm.size(3))
                neg_neg = neg_neg.view(-1, 1, neg_neg.size(2), neg_neg.size(3))
                an_loss = self.loss(anchor_fm.repeat(1, nneg, 1, 1).view(-1, 1, anchor_fm.size(2), anchor_fm.size(3)),
                                    neg_fm)
                an_loss = an_loss.view((-1, nneg)).min(1)[0]
                nn_loss = self.loss(neg_fm, neg_neg)
                nn_loss = nn_loss.view((-1, nneg)).min(1)[0]
                ap_loss = self.loss(anchor_fm, pos_fm)

                sstl = ap_loss - an_loss + args.alpha
                sstl = torch.clamp(sstl, min=0)
                nnsstl = ap_loss - nn_loss + args.nnalpha
                nnsstl = torch.clamp(nnsstl, min=0)
                loss = torch.sum(sstl + nnsstl) / args.batch_size

                loss.backward()
                self.optimizer.step()

                # agg_loss += loss.data[0]
                agg_loss += loss.data
                train_loss += loss.item()

                if e % args.log_interval == 0:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\t {:.6f}".format(
                        time.ctime(), e, count, self.dataset_size, agg_loss / (batch_id + 1)
                    )
                    print(mesg)

                if batch_id % 5 == 0:
                    self.writer.add_scalar("loss",
                                           scalar_value=train_loss,
                                           global_step=(e * epoch_steps + batch_id))
                    train_loss = 0

            if args.checkpoint_dir is not None and e % args.checkpoint_interval == 0:
                self.save(args.checkpoint_dir, e)

        self.writer.close()

    def triplet_train(self, args):
        epoch_steps = len(self.train_loader)
        train_loss = 0
        start_epoch = ''.join(x for x in os.path.basename(args.start_ckpt) if x.isdigit())
        if start_epoch:
            start_epoch = int(start_epoch) + 1
            self.load(args.start_ckpt)
        else:
            start_epoch = 1
    
        for e in range(start_epoch, args.epochs + start_epoch):
            self.exp_lr_scheduler(e, lr_decay_epoch=300)
            self.inference.train()
            agg_loss = 0.
            count = 0
            for batch_id, (x, _) in enumerate(self.train_loader):
                count += len(x)
                self.optimizer.zero_grad()
                x = x.cuda()
                
                x = Variable(x, requires_grad=False)

                if "RFN-32" in args.model:
                    # the RFN-32 will use the 3 channels image as the input image
                    fms = self.inference(x.view(-1, 3, x.size(2), x.size(3)).repeat(1, 3, 1, 1))
                else:
                    fms = self.inference(x.view(-1, 3, x.size(2), x.size(3)))

                fms = fms.view(x.size(0), -1, fms.size(2), fms.size(3))

                anchor_fm = fms[:, 0, :, :].unsqueeze(1)
                pos_fm = fms[:, 1, :, :].unsqueeze(1)
                neg_fm = fms[:, 2:, :, :].contiguous()

                nneg = neg_fm.size(1)
                neg_fm = neg_fm.view(-1, 1, neg_fm.size(2), neg_fm.size(3))
                an_loss = self.loss(anchor_fm.repeat(1, nneg, 1, 1).view(-1, 1, anchor_fm.size(2), anchor_fm.size(3)), neg_fm)
                an_loss = an_loss.view((-1, nneg)).min(1)[0]
                ap_loss = self.loss(anchor_fm, pos_fm)

                sstl = ap_loss - an_loss + args.alpha
                sstl = torch.clamp(sstl, min=0)
                loss = torch.sum(sstl) / args.batch_size

                loss.backward()
                self.optimizer.step()

                # agg_loss += loss.data[0]
                agg_loss += loss.data
                train_loss += loss.item()

                if e % args.log_interval == 0:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\t {:.6f}".format(
                        time.ctime(), e, count, self.dataset_size, agg_loss / (batch_id + 1)
                    )
                    print(mesg)

                if batch_id % 5 == 0:
                    self.writer.add_scalar("loss",
                                           scalar_value=train_loss,
                                           global_step=(e*epoch_steps + batch_id))
                    train_loss = 0

            
            if args.checkpoint_dir is not None and e % args.checkpoint_interval == 0:
                self.save(args.checkpoint_dir, e)

        self.writer.close()


    def save(self, checkpoint_dir, e):
        self.inference.eval()
        self.inference.cpu()
        ckpt_model_filename = os.path.join(checkpoint_dir, "ckpt_epoch_" + str(e) + ".pth")
        torch.save(self.inference.state_dict(), ckpt_model_filename)
        self.inference.cuda()
        self.inference.train()

    def load(self, checkpoint_dir):
        self.inference.load_state_dict(torch.load(checkpoint_dir))
        self.inference.cuda()
