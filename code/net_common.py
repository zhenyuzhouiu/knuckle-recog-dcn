from __future__ import division
import sys

import cv2
import numpy
import torch
import numpy as np
import torchvision.ops
from torch.autograd import Variable
import torch.nn.functional as F
import math


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class DeconvResBlock(torch.nn.Module):
    def __init__(self, channels):
        super(DeconvResBlock, self).__init__()
        self.conv1 = DeformableConv2d2v(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.BatchNorm2d(num_features=channels)
        self.conv2 = DeformableConv2d2v(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.BatchNorm2d(num_features=channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class RIPShiftedLoss(torch.nn.Module):
    def __init__(self, dilation, subsize, angle, topk=-1):
        super(RIPShiftedLoss, self).__init__()
        self.dilation = dilation
        self.subsize = subsize
        self.angle = angle
        self.topk = topk

    def mse_loss(self, src, target):
        if isinstance(src, torch.autograd.Variable):
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.data.nelement() * src.size(0)
        else:
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.nelement() * src.size(0)

    def rotate_mse_loss(self, src, target, mask):
        se = (src - target) ** 2
        mask_se = se * mask
        sum_se = mask_se.view(src.size(0), -1).sum(1)
        sum = mask.view(mask.size(0), -1).sum(1)
        mse = sum_se / sum
        return mse

    def forward(self, fm1, fm2):
        # C * H * W
        bs, _, h, w = fm1.size()

        dilation_list = [h, w, self.dilation]
        # sort() function doesn't have return value
        dilation_list.sort()
        self.dilation = dilation_list[0]
        subsize_list = [h, w, self.subsize]
        subsize_list.sort()
        self.subsize = subsize_list[0]

        # when the shift size: dilation is equal to 0
        min_dist = torch.ones(bs).cuda() * sys.float_info.max
        if isinstance(fm1, torch.autograd.Variable):
            min_dist = Variable(min_dist, requires_grad=False)

        if self.dilation == 0:
            dist = self.mse_loss(fm1, fm2).cuda()
            min_dist, _ = torch.min(torch.stack([min_dist, dist]), 0)
            return min_dist

        # when the shift size: dilation is not equal to 0
        min_dist = torch.zeros(bs).cuda()
        if isinstance(fm1, torch.autograd.Variable):
            min_dist = Variable(min_dist, requires_grad=False)

        for sub_x in range(0, fm1.size(-1), self.subsize):
            for sub_y in range(0, fm1.size(-2), self.subsize):
                ref1 = fm1[:, :, sub_y:sub_y + self.subsize, sub_x:sub_x + self.subsize]

                s_sub_min_dist = torch.ones(bs).cuda() * sys.float_info.max
                if isinstance(fm1, torch.autograd.Variable):
                    s_sub_min_dist = Variable(s_sub_min_dist, requires_grad=False)

                r_sub_min_dist = torch.ones(bs).cuda() * sys.float_info.max
                if isinstance(fm1, torch.autograd.Variable):
                    r_sub_min_dist = Variable(r_sub_min_dist, requires_grad=False)

                for dw in range(-self.dilation, self.dilation + 1):
                    for dh in range(-self.dilation, self.dilation + 1):
                        if sub_y + dh < 0:
                            if sub_x + dw < 0:
                                ref2 = fm2[:, :, 0:self.subsize, 0:self.subsize]
                            elif sub_x + dw + self.subsize > w:
                                ref2 = fm2[:, :, 0:self.subsize, fm2.size(-1) - self.subsize:fm2.size(-1)]
                            else:
                                ref2 = fm2[:, :, 0:self.subsize, sub_x + dw:sub_x + self.subsize + dw]
                        elif sub_y + dh + self.subsize > h:
                            if sub_x + dw < 0:
                                ref2 = fm2[:, :, fm2.size(-2) - self.subsize:fm2.size(-2), 0:self.subsize]
                            elif sub_x + dw + self.subsize > w:
                                ref2 = fm2[:, :, fm2.size(-2) - self.subsize:fm2.size(-2),
                                       fm1.size(-1) - self.subsize:fm1.size(-1)]
                            else:
                                ref2 = fm2[:, :, fm2.size(-2) - self.subsize:fm2.size(-2),
                                       sub_x + dw:sub_x + self.subsize + dw]
                        else:
                            if sub_x + dw < 0:
                                ref2 = fm2[:, :, sub_y + dh:sub_y + self.subsize + dh, 0:self.subsize]
                            elif sub_x + dw + self.subsize > w:
                                ref2 = fm2[:, :, sub_y + dh:sub_y + self.subsize + dh,
                                       fm1.size(-1) - self.subsize:fm1.size(-1)]
                            else:
                                ref2 = fm2[:, :, sub_y + dh:sub_y + self.subsize + dh,
                                       sub_x + dw:sub_x + self.subsize + dw]

                        sub_dist = self.mse_loss(ref1, ref2).cuda()
                        s_sub_min_dist, _ = torch.min(torch.stack([s_sub_min_dist.squeeze(), sub_dist.squeeze()]), 0)

                for a in range(-self.angle, self.angle + 1):
                    s_bs, s_c, s_h, s_w = ref1.size()
                    mask = torch.ones([s_w, s_h])
                    ref2 = fm2[:, :, sub_y:sub_y + self.subsize, sub_x:sub_x + self.subsize]
                    rotate_matrix = cv2.getRotationMatrix2D(center=[s_w / 2, s_h / 2], angle=a, scale=1)
                    ref2 = torch.squeeze(ref2, dim=1)
                    ref2 = ref2.cpu().detach().numpy()
                    ref2 = ref2.transpose(1, 2, 0)
                    r_ref2 = np.zeros(ref2.shape)
                    for n_g in range(s_bs):
                        r_ref2_n = cv2.warpAffine(ref2[:, :, n_g], M=rotate_matrix, dsize=(s_w, s_h))
                        r_ref2[:, :, n_g] = r_ref2_n

                    # r_ref2 = cv2.warpAffine(ref2, M=rotate_matrix, dsize=(s_w, s_h))

                    r_ref2 = torch.from_numpy(r_ref2).cuda()
                    r_ref2 = r_ref2.permute(2, 0, 1).unsqueeze(1)
                    mask = mask.data.numpy()
                    r_mask = cv2.warpAffine(mask, M=rotate_matrix, dsize=(s_w, s_h))
                    r_mask = torch.from_numpy(r_mask).cuda()
                    r_mask = r_mask.unsqueeze(0).unsqueeze(0).repeat(s_bs, s_c, 1, 1)

                    r_sub_dist = self.rotate_mse_loss(ref1, r_ref2, r_mask)
                    r_sub_min_dist, _ = torch.min(torch.stack([r_sub_min_dist.squeeze(), r_sub_dist.squeeze()]), 0)

                sub_min_dist, _ = torch.min(torch.stack([r_sub_min_dist.squeeze(), s_sub_min_dist.squeeze()]), 0)

                sub_min_dist = torch.reshape(sub_min_dist, [1, sub_min_dist.size(0)])
                if sub_x == 0 and sub_y == 0:
                    topk_dist = sub_min_dist
                else:
                    topk_dist = torch.vstack([topk_dist, sub_min_dist])

        for i in range(topk_dist.size(1)):
            dist = topk_dist[:, i]
            sorted_d, indices = torch.sort(dist)
            if self.topk >= 0 and self.topk <= topk_dist.size(0):
                min_dist[i] = torch.sum(sorted_d[0:self.topk])
            else:
                min_dist[i] = torch.sum(dist)

        return min_dist.squeeze()


class RANDIPShiftedLoss(torch.nn.Module):
    def __init__(self, dilation, subsize, angle, topk=-1):
        super(RANDIPShiftedLoss, self).__init__()
        self.dilation = dilation
        self.subsize = subsize
        self.angle = angle
        self.topk = topk

    def mse_loss(self, src, target):
        if isinstance(src, torch.autograd.Variable):
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.data.nelement() * src.size(0)
        else:
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.nelement() * src.size(0)

    def rotate_mse_loss(self, src, target, mask):
        se = (src - target) ** 2
        mask_se = se * mask
        sum_se = mask_se.view(src.size(0), -1).sum(1)
        sum = mask.view(mask.size(0), -1).sum(1)
        mse = sum_se / sum
        return mse

    def forward(self, fm1, fm2):
        # C * H * W
        bs, _, h, w = fm1.size()

        dilation_list = [h, w, self.dilation]
        # sort() function doesn't have return value
        dilation_list.sort()
        self.dilation = dilation_list[0]
        subsize_list = [h, w, self.subsize]
        subsize_list.sort()
        self.subsize = subsize_list[0]

        # when the shift size: dilation is equal to 0
        min_dist = torch.ones(bs, device=fm1.device) * sys.float_info.max
        if fm1.requires_grad:
            min_dist = Variable(min_dist, requires_grad=True)
        else:
            min_dist = Variable(min_dist, requires_grad=False)
        # if isinstance(fm1, torch.autograd.Variable):
        #     min_dist = Variable(min_dist, requires_grad=False)

        if self.dilation == 0:
            dist = self.mse_loss(fm1, fm2).to(fm1.device)
            min_dist, _ = torch.min(torch.stack([min_dist, dist]), 0)
            return min_dist

        # when the shift size: dilation is not equal to 0
        min_dist = torch.zeros(bs).to(fm1.device)
        if fm1.requires_grad:
            min_dist = Variable(min_dist, requires_grad=True)
        else:
            min_dist = Variable(min_dist, requires_grad=False)

        rands_dist = torch.zeros(bs).to(fm1.device)
        if fm1.requires_grad:
            rands_dist = Variable(rands_dist, requires_grad=True)
        else:
            rands_dist = Variable(rands_dist, requires_grad=False)

        for sub_x in range(0, fm1.size(-1), self.subsize):
            for sub_y in range(0, fm1.size(-2), self.subsize):
                ref1 = fm1[:, :, sub_y:sub_y + self.subsize, sub_x:sub_x + self.subsize]

                sub_min_dist = torch.ones(bs).cuda() * sys.float_info.max
                if fm1.requires_grad:
                    sub_min_dist = Variable(sub_min_dist, requires_grad=True)
                else:
                    sub_min_dist = Variable(sub_min_dist, requires_grad=False)

                for dw in range(-self.dilation, self.dilation + 1):
                    for dh in range(-self.dilation, self.dilation + 1):
                        if sub_y + dh < 0:
                            if sub_x + dw < 0:
                                ref2 = fm2[:, :, 0:self.subsize, 0:self.subsize]
                            elif sub_x + dw + self.subsize > w:
                                ref2 = fm2[:, :, 0:self.subsize, fm2.size(-1) - self.subsize:fm2.size(-1)]
                            else:
                                ref2 = fm2[:, :, 0:self.subsize, sub_x + dw:sub_x + self.subsize + dw]
                        elif sub_y + dh + self.subsize > h:
                            if sub_x + dw < 0:
                                ref2 = fm2[:, :, fm2.size(-2) - self.subsize:fm2.size(-2), 0:self.subsize]
                            elif sub_x + dw + self.subsize > w:
                                ref2 = fm2[:, :, fm2.size(-2) - self.subsize:fm2.size(-2),
                                       fm1.size(-1) - self.subsize:fm1.size(-1)]
                            else:
                                ref2 = fm2[:, :, fm2.size(-2) - self.subsize:fm2.size(-2),
                                       sub_x + dw:sub_x + self.subsize + dw]
                        else:
                            if sub_x + dw < 0:
                                ref2 = fm2[:, :, sub_y + dh:sub_y + self.subsize + dh, 0:self.subsize]
                            elif sub_x + dw + self.subsize > w:
                                ref2 = fm2[:, :, sub_y + dh:sub_y + self.subsize + dh,
                                       fm1.size(-1) - self.subsize:fm1.size(-1)]
                            else:
                                ref2 = fm2[:, :, sub_y + dh:sub_y + self.subsize + dh,
                                       sub_x + dw:sub_x + self.subsize + dw]

                        for a in range(-self.angle, self.angle + 1):
                            s_bs, s_c, s_h, s_w = ref1.size()
                            # ref2 = fm2[:, :, sub_y:sub_y + self.subsize, sub_x:sub_x + self.subsize]
                            M = cv2.getRotationMatrix2D(center=[s_w / 2, s_h / 2], angle=a, scale=1)
                            a_ref2 = torch.squeeze(ref2, dim=1)
                            a_ref2 = a_ref2.detach().cpu().numpy()
                            a_ref2 = a_ref2.transpose(1, 2, 0)
                            if s_bs > 512:
                                num_chuncks = s_bs // 512
                                num_reminder = s_bs % 512
                                r_ref2 = np.zeros(a_ref2.shape)
                                for nc in range(num_chuncks):
                                    nc_ref2 = a_ref2[:, :, 0 + nc * 512:512 + nc * 512]
                                    r_nc_ref2 = cv2.warpAffine(nc_ref2, M=M, dsize=[s_w, s_h])
                                    r_ref2[:, :, 0 + nc * 512:512 + nc * 512] = r_nc_ref2
                                if num_reminder > 0:
                                    nc_ref2 = a_ref2[:, :, 512 + nc * 512:]
                                    r_nc_ref2 = cv2.warpAffine(nc_ref2, M=M, dsize=[s_w, s_h])
                                    if r_nc_ref2.ndim == 2:
                                        r_nc_ref2 = numpy.expand_dims(r_nc_ref2, axis=-1)
                                    r_ref2[:, :, 512 + nc * 512:] = r_nc_ref2
                            else:
                                r_ref2 = cv2.warpAffine(a_ref2, M=M, dsize=[s_w, s_h])
                            if r_ref2.ndim == 2:
                                r_ref2 = numpy.expand_dims(r_ref2, axis=-1)
                            r_ref2 = torch.from_numpy(r_ref2).to(fm1.device)
                            r_ref2 = r_ref2.permute(2, 0, 1).unsqueeze(1)

                            mask = np.ones([s_h, s_w])
                            r_mask = cv2.warpAffine(mask, M=M, dsize=[s_w, s_h])
                            r_mask = torch.from_numpy(r_mask).to(fm1.device)
                            r_mask = r_mask.unsqueeze(0).unsqueeze(0).repeat(s_bs, s_c, 1, 1)

                            rands_dist = self.rotate_mse_loss(ref1, r_ref2, r_mask)
                            sub_min_dist, _ = torch.min(torch.stack([sub_min_dist.squeeze(), rands_dist.squeeze()]), 0)

                min_dist = min_dist + sub_min_dist
        #         sub_min_dist = torch.reshape(sub_min_dist, [1, sub_min_dist.size(0)])
        #         if sub_x == 0 and sub_y == 0:
        #             topk_dist = sub_min_dist
        #         else:
        #             topk_dist = torch.vstack([topk_dist, sub_min_dist])
        #
        # for i in range(topk_dist.size(1)):
        #     dist = topk_dist[:, i]
        #     sorted_d, indices = torch.sort(dist)
        #     if 0 <= self.topk <= topk_dist.size(0):
        #         min_dist[i] = torch.sum(sorted_d[0:self.topk])
        #     else:
        #         min_dist[i] = torch.sum(dist)

        return min_dist.squeeze()


class MultiProcessRANDIPShiftedLoss(torch.nn.Module):
    def __init__(self, dilation, subsize, angle, topk=-1):
        super(RANDIPShiftedLoss, self).__init__()
        self.dilation = dilation
        self.subsize = subsize
        self.angle = angle
        self.topk = topk

    def mse_loss(self, src, target):
        if isinstance(src, torch.autograd.Variable):
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.data.nelement() * src.size(0)
        else:
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.nelement() * src.size(0)

    def rotate_mse_loss(self, src, target, mask):
        se = (src - target) ** 2
        mask_se = se * mask
        sum_se = mask_se.view(src.size(0), -1).sum(1)
        sum = mask.view(mask.size(0), -1).sum(1)
        mse = sum_se / sum
        return mse

    def forward(self, fm1, fm2):
        # C * H * W
        bs, _, h, w = fm1.size()

        dilation_list = [h, w, self.dilation]
        # sort() function doesn't have return value
        dilation_list.sort()
        self.dilation = dilation_list[0]
        subsize_list = [h, w, self.subsize]
        subsize_list.sort()
        self.subsize = subsize_list[0]

        # when the shift size: dilation is equal to 0
        min_dist = torch.ones(bs, device=fm1.device) * sys.float_info.max
        if fm1.requires_grad:
            min_dist = Variable(min_dist, requires_grad=True)
        else:
            min_dist = Variable(min_dist, requires_grad=False)
        # if isinstance(fm1, torch.autograd.Variable):
        #     min_dist = Variable(min_dist, requires_grad=False)

        if self.dilation == 0:
            dist = self.mse_loss(fm1, fm2).to(fm1.device)
            min_dist, _ = torch.min(torch.stack([min_dist, dist]), 0)
            return min_dist

        # when the shift size: dilation is not equal to 0
        min_dist = torch.zeros(bs).to(fm1.device)
        if fm1.requires_grad:
            min_dist = Variable(min_dist, requires_grad=True)
        else:
            min_dist = Variable(min_dist, requires_grad=False)

        rands_dist = torch.zeros(bs).to(fm1.device)
        if fm1.requires_grad:
            rands_dist = Variable(rands_dist, requires_grad=True)
        else:
            rands_dist = Variable(rands_dist, requires_grad=False)

        for sub_x in range(0, fm1.size(-1), self.subsize):
            for sub_y in range(0, fm1.size(-2), self.subsize):
                ref1 = fm1[:, :, sub_y:sub_y + self.subsize, sub_x:sub_x + self.subsize]

                sub_min_dist = torch.ones(bs).cuda() * sys.float_info.max
                if fm1.requires_grad:
                    sub_min_dist = Variable(sub_min_dist, requires_grad=True)
                else:
                    sub_min_dist = Variable(sub_min_dist, requires_grad=False)

                for dw in range(-self.dilation, self.dilation + 1):
                    for dh in range(-self.dilation, self.dilation + 1):
                        if sub_y + dh < 0:
                            if sub_x + dw < 0:
                                ref2 = fm2[:, :, 0:self.subsize, 0:self.subsize]
                            elif sub_x + dw + self.subsize > w:
                                ref2 = fm2[:, :, 0:self.subsize, fm2.size(-1) - self.subsize:fm2.size(-1)]
                            else:
                                ref2 = fm2[:, :, 0:self.subsize, sub_x + dw:sub_x + self.subsize + dw]
                        elif sub_y + dh + self.subsize > h:
                            if sub_x + dw < 0:
                                ref2 = fm2[:, :, fm2.size(-2) - self.subsize:fm2.size(-2), 0:self.subsize]
                            elif sub_x + dw + self.subsize > w:
                                ref2 = fm2[:, :, fm2.size(-2) - self.subsize:fm2.size(-2),
                                       fm1.size(-1) - self.subsize:fm1.size(-1)]
                            else:
                                ref2 = fm2[:, :, fm2.size(-2) - self.subsize:fm2.size(-2),
                                       sub_x + dw:sub_x + self.subsize + dw]
                        else:
                            if sub_x + dw < 0:
                                ref2 = fm2[:, :, sub_y + dh:sub_y + self.subsize + dh, 0:self.subsize]
                            elif sub_x + dw + self.subsize > w:
                                ref2 = fm2[:, :, sub_y + dh:sub_y + self.subsize + dh,
                                       fm1.size(-1) - self.subsize:fm1.size(-1)]
                            else:
                                ref2 = fm2[:, :, sub_y + dh:sub_y + self.subsize + dh,
                                       sub_x + dw:sub_x + self.subsize + dw]

                        for a in range(-self.angle, self.angle + 1):
                            s_bs, s_c, s_h, s_w = ref1.size()
                            # ref2 = fm2[:, :, sub_y:sub_y + self.subsize, sub_x:sub_x + self.subsize]
                            M = cv2.getRotationMatrix2D(center=[s_w / 2, s_h / 2], angle=a, scale=1)
                            a_ref2 = torch.squeeze(ref2, dim=1)
                            a_ref2 = a_ref2.detach().cpu().numpy()
                            a_ref2 = a_ref2.transpose(1, 2, 0)
                            if s_bs > 512:
                                num_chuncks = s_bs // 512
                                num_reminder = s_bs % 512
                                r_ref2 = np.zeros(a_ref2.shape)
                                for nc in range(num_chuncks):
                                    nc_ref2 = a_ref2[:, :, 0 + nc * 512:512 + nc * 512]
                                    r_nc_ref2 = cv2.warpAffine(nc_ref2, M=M, dsize=[s_w, s_h])
                                    r_ref2[:, :, 0 + nc * 512:512 + nc * 512] = r_nc_ref2
                                if num_reminder > 0:
                                    nc_ref2 = a_ref2[:, :, 512 + nc * 512:]
                                    r_nc_ref2 = cv2.warpAffine(nc_ref2, M=M, dsize=[s_w, s_h])
                                    if r_nc_ref2.ndim == 2:
                                        r_nc_ref2 = numpy.expand_dims(r_nc_ref2, axis=-1)
                                    r_ref2[:, :, 512 + nc * 512:] = r_nc_ref2
                            else:
                                r_ref2 = cv2.warpAffine(a_ref2, M=M, dsize=[s_w, s_h])
                            if r_ref2.ndim == 2:
                                r_ref2 = numpy.expand_dims(r_ref2, axis=-1)
                            r_ref2 = torch.from_numpy(r_ref2).to(fm1.device)
                            r_ref2 = r_ref2.permute(2, 0, 1).unsqueeze(1)

                            mask = np.ones([s_h, s_w])
                            r_mask = cv2.warpAffine(mask, M=M, dsize=[s_w, s_h])
                            r_mask = torch.from_numpy(r_mask).to(fm1.device)
                            r_mask = r_mask.unsqueeze(0).unsqueeze(0).repeat(s_bs, s_c, 1, 1)

                            rands_dist = self.rotate_mse_loss(ref1, r_ref2, r_mask)
                            sub_min_dist, _ = torch.min(torch.stack([sub_min_dist.squeeze(), rands_dist.squeeze()]), 0)

                min_dist = min_dist + sub_min_dist
        #         sub_min_dist = torch.reshape(sub_min_dist, [1, sub_min_dist.size(0)])
        #         if sub_x == 0 and sub_y == 0:
        #             topk_dist = sub_min_dist
        #         else:
        #             topk_dist = torch.vstack([topk_dist, sub_min_dist])
        #
        # for i in range(topk_dist.size(1)):
        #     dist = topk_dist[:, i]
        #     sorted_d, indices = torch.sort(dist)
        #     if 0 <= self.topk <= topk_dist.size(0):
        #         min_dist[i] = torch.sum(sorted_d[0:self.topk])
        #     else:
        #         min_dist[i] = torch.sum(dist)

        return min_dist.squeeze()


class SubShiftedLoss(torch.nn.Module):
    def __init__(self, dilation, subsize, topk=-1):
        super(SubShiftedLoss, self).__init__()
        self.dilation = dilation
        self.subsize = subsize
        self.topk = topk

    def mse_loss(self, src, target):
        if isinstance(src, torch.autograd.Variable):
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.data.nelement() * src.size(0)
        else:
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.nelement() * src.size(0)

    def forward(self, fm1, fm2):
        # C * H * W
        bs, _, h, w = fm1.size()

        dilation_list = [h, w, self.dilation]
        # sort() function doesn't have return value
        dilation_list.sort()
        self.dilation = dilation_list[0]
        subsize_list = [h, w, self.subsize]
        subsize_list.sort()
        self.subsize = subsize_list[0]

        # when the shift size: dilation is equal to 0
        min_dist = torch.ones(bs).cuda() * sys.float_info.max
        if isinstance(fm1, torch.autograd.Variable):
            min_dist = Variable(min_dist, requires_grad=False)

        if self.dilation == 0:
            dist = self.mse_loss(fm1, fm2).cuda()
            min_dist, _ = torch.min(torch.stack([min_dist, dist]), 0)
            return min_dist

        # when the shift size: dilation is not equal to 0
        min_dist = torch.zeros(bs).cuda()
        if isinstance(fm1, torch.autograd.Variable):
            min_dist = Variable(min_dist, requires_grad=False)

        for sub_x in range(0, fm1.size(-1), self.subsize):
            for sub_y in range(0, fm1.size(-2), self.subsize):
                ref1 = fm1[:, :, sub_y:sub_y + self.subsize, sub_x:sub_x + self.subsize]

                sub_min_dist = torch.ones(bs).cuda() * sys.float_info.max
                if isinstance(fm1, torch.autograd.Variable):
                    sub_min_dist = Variable(sub_min_dist, requires_grad=False)

                for dw in range(-self.dilation, self.dilation + 1):
                    for dh in range(-self.dilation, self.dilation + 1):
                        if sub_y + dh < 0:
                            if sub_x + dw < 0:
                                ref2 = fm2[:, :, 0:self.subsize, 0:self.subsize]
                            elif sub_x + dw + self.subsize > w:
                                ref2 = fm2[:, :, 0:self.subsize, fm2.size(-1) - self.subsize:fm2.size(-1)]
                            else:
                                ref2 = fm2[:, :, 0:self.subsize, sub_x + dw:sub_x + self.subsize + dw]
                        elif sub_y + dh + self.subsize > h:
                            if sub_x + dw < 0:
                                ref2 = fm2[:, :, fm2.size(-2) - self.subsize:fm2.size(-2), 0:self.subsize]
                            elif sub_x + dw + self.subsize > w:
                                ref2 = fm2[:, :, fm2.size(-2) - self.subsize:fm2.size(-2),
                                       fm1.size(-1) - self.subsize:fm1.size(-1)]
                            else:
                                ref2 = fm2[:, :, fm2.size(-2) - self.subsize:fm2.size(-2),
                                       sub_x + dw:sub_x + self.subsize + dw]
                        else:
                            if sub_x + dw < 0:
                                ref2 = fm2[:, :, sub_y + dh:sub_y + self.subsize + dh, 0:self.subsize]
                            elif sub_x + dw + self.subsize > w:
                                ref2 = fm2[:, :, sub_y + dh:sub_y + self.subsize + dh,
                                       fm1.size(-1) - self.subsize:fm1.size(-1)]
                            else:
                                ref2 = fm2[:, :, sub_y + dh:sub_y + self.subsize + dh,
                                       sub_x + dw:sub_x + self.subsize + dw]

                        sub_dist = self.mse_loss(ref1, ref2).cuda()
                        sub_min_dist, _ = torch.min(torch.stack([sub_min_dist.squeeze(), sub_dist.squeeze()]), 0)

                if sub_min_dist.ndim == 0:
                    temp = sub_min_dist
                    sub_min_dist = torch.ones(bs).cuda() * sys.float_info.max
                    if isinstance(fm1, torch.autograd.Variable):
                        sub_min_dist = Variable(sub_min_dist, requires_grad=False)
                    sub_min_dist[0] = temp
                sub_min_dist = torch.reshape(sub_min_dist, [1, sub_min_dist.size(0)])
                if sub_x == 0 and sub_y == 0:
                    topk_disk = sub_min_dist
                else:
                    topk_disk = torch.vstack([topk_disk, sub_min_dist])

        for i in range(topk_disk.size(1)):
            dist = topk_disk[:, i]
            sorted_d, indices = torch.sort(dist)
            if 0 <= self.topk <= topk_disk.size(0):
                min_dist[i] = torch.sum(sorted_d[0:self.topk])
            else:
                min_dist[i] = torch.sum(dist)

        return min_dist.squeeze()


class WholeRotationShiftedLoss(torch.nn.Module):
    def __init__(self, hshift, vshift, angle):
        super(WholeRotationShiftedLoss, self).__init__()
        self.hshift = hshift
        self.vshift = vshift
        self.angle = angle

    def rotate_mse_loss(self, src, target, mask):
        # if isinstance(src, torch.autograd.Variable):
        #     return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.data.nelement() * src.size(0)
        # else:
        #     return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.nelement() * src.size(0)
        se = (src - target) ** 2
        mask_se = se * mask
        sum_se = mask_se.view(src.size(0), -1).sum(1)
        sum = mask.view(src.size(0), -1).sum(1)
        mse = sum_se / sum
        return mse

    def mse_loss(self, src, target):
        if isinstance(src, torch.autograd.Variable):
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.data.nelement() * src.size(0)
        else:
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.nelement() * src.size(0)

    def forward(self, fm1, fm2):
        # C * H * W
        bs, _, h, w = fm1.size()
        if w < self.hshift:
            self.hshift = self.hshift % w
        if h < self.vshift:
            self.vshift = self.vshift % h

        # min_dist shape (bs, )  & sys.float_info.max is to get the max float value
        # min_dist save the maximal float value
        min_dist = torch.ones(bs, device=fm1.device) * sys.float_info.max
        # if isinstance(fm1, torch.autograd.Variable):
        #     min_dist = Variable(min_dist, requires_grad=True)
        if fm1.requires_grad:
            min_dist = Variable(min_dist, requires_grad=True)
        else:
            min_dist = Variable(min_dist, requires_grad=False)

        if self.hshift == 0 and self.vshift == 0:
            dist = self.mse_loss(fm1, fm2).to(fm1.device)
            min_dist, _ = torch.min(torch.stack([min_dist, dist]), 0)
            return min_dist

        for bh in range(-self.hshift, self.hshift + 1):
            for bv in range(-self.vshift, self.vshift + 1):
                if bh >= 0:
                    ref1, ref2 = fm1[:, :, :, :w - bh], fm2[:, :, :, bh:]
                else:
                    ref1, ref2 = fm1[:, :, :, -bh:], fm2[:, :, :, :w + bh]

                if bv >= 0:
                    ref1, ref2 = ref1[:, :, :h - bv, :], ref2[:, :, bv:, :]
                else:
                    ref1, ref2 = ref1[:, :, -bv:, :], ref2[:, :, :h + bv, :]

                for theta in range(-self.angle, self.angle + 1):
                    overlap_bs, overlap_c, overlap_h, overlap_w = ref1.size()
                    M = cv2.getRotationMatrix2D(center=[overlap_w / 2, overlap_h / 2], angle=theta, scale=1)
                    ref2 = torch.squeeze(ref2, dim=1)
                    n_ref2 = ref2.detach().cpu().numpy()
                    n_ref2 = n_ref2.transpose(1, 2, 0)
                    if overlap_bs > 512:
                        num_chuncks = overlap_bs // 512
                        num_reminder = overlap_bs % 512
                        r_ref2 = np.zeros(n_ref2.shape)
                        for nc in range(num_chuncks):
                            nc_ref2 = n_ref2[:, :, 0 + nc * 512:512 + nc * 512]
                            r_nc_ref2 = cv2.warpAffine(nc_ref2, M=M, dsize=[overlap_w, overlap_h])
                            r_ref2[:, :, 0 + nc * 512:512 + nc * 512] = r_nc_ref2
                        if num_reminder > 0:
                            nc_ref2 = n_ref2[:, :, 512 + nc * 512:]
                            r_nc_ref2 = cv2.warpAffine(nc_ref2, M=M, dsize=[overlap_w, overlap_h])
                            if r_nc_ref2.ndim == 2:
                                r_nc_ref2 = numpy.expand_dims(r_nc_ref2, axis=-1)
                            r_ref2[:, :, 512 + nc * 512:] = r_nc_ref2
                    else:
                        r_ref2 = cv2.warpAffine(n_ref2, M=M, dsize=[overlap_w, overlap_h])
                    # r_ref2 = rotate(n_ref2, angle=theta, reshape=False)

                    if r_ref2.ndim == 2:
                        r_ref2 = numpy.expand_dims(r_ref2, axis=-1)
                    r_ref2 = torch.from_numpy(r_ref2).to(fm1.device)
                    r_ref2 = r_ref2.permute(2, 0, 1).unsqueeze(1)

                    mask = np.ones([overlap_h, overlap_w])
                    r_mask = cv2.warpAffine(mask, M=M, dsize=[overlap_w, overlap_h])
                    # r_mask = rotate(mask, angle=theta, reshape=False)
                    r_mask = torch.from_numpy(r_mask).to(fm1.device)
                    r_mask = r_mask.unsqueeze(0).unsqueeze(0).repeat(overlap_bs, overlap_c, 1, 1)

                    dist = self.rotate_mse_loss(ref1, r_ref2, r_mask).to(fm1.device)

                    min_dist, _ = torch.min(torch.stack([min_dist.squeeze(), dist.squeeze()]), 0)
        return min_dist.squeeze()


def generate_theta(i_radian, i_tx, i_ty, i_batch_size, i_h, i_w, i_dtype):
    theta = torch.tensor([[math.cos(i_radian), math.sin(-i_radian) * i_h / i_w, i_tx],
                          [math.sin(i_radian) * i_w / i_h, math.cos(i_radian), i_ty]],
                         dtype=i_dtype).unsqueeze(0).repeat(i_batch_size, 1, 1)
    return theta


def mse_loss(i_fm1, i_fm2, i_mask):
    # the input feature map shape is (bs, 1, h, w)
    square_err = torch.mul(torch.pow((i_fm1 - i_fm2), 2), i_mask)
    mean_se = square_err.view(i_fm1.size(0), -1).sum(1) / i_mask.view(i_fm1.size(0), -1).sum(1)
    return mean_se


class ImageBlockRotationAndTranslation(torch.nn.Module):
    def __init__(self, i_block_size, i_v_shift, i_h_shift, i_angle, i_topk):
        super(ImageBlockRotationAndTranslation, self).__init__()
        self.block_size = i_block_size
        self.v_shift = i_v_shift
        self.h_shift = i_h_shift
        self.angle = i_angle
        self.topk = i_topk

    def forward(self, i_fm1, i_fm2):
        b, c, h, w = i_fm1.shape
        if self.training:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=True, device=i_fm1.device)
        else:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=False, device=i_fm1.device)

        n_affine = 0
        for sub_x in range(0, w, self.block_size):
            for sub_y in range(0, h, self.block_size):
                sub_fm1 = i_fm1[:, :, sub_y:sub_y + self.block_size, sub_x:sub_x + self.block_size]

                sub_affine = 0
                if self.training:
                    sub_min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=True, device=i_fm1.device)
                else:
                    sub_min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=False, device=i_fm1.device)

                for dx in range(-self.h_shift, self.h_shift + 1):
                    for dy in range(-self.v_shift, self.v_shift + 1):
                        if sub_y + dy < 0:
                            if sub_x + dx < 0:
                                sub_fm2 = i_fm2[:, :, 0:self.block_size, 0:self.block_size]
                            elif sub_x + dx + self.block_size > w:
                                sub_fm2 = i_fm2[:, :, 0:self.block_size, w - self.block_size:w]
                            else:
                                sub_fm2 = i_fm2[:, :, 0:self.block_size, sub_x + dx:sub_x + self.block_size + dx]
                        elif sub_y + dy + self.block_size > h:
                            if sub_x + dx < 0:
                                sub_fm2 = i_fm2[:, :, h - self.block_size:h, 0:self.block_size]
                            elif sub_x + dx + self.block_size > w:
                                sub_fm2 = i_fm2[:, :, h - self.block_size:h,
                                          w - self.block_size:w]
                            else:
                                sub_fm2 = i_fm2[:, :, h - self.block_size:h,
                                          sub_x + dx:sub_x + self.block_size + dx]
                        else:
                            if sub_x + dx < 0:
                                sub_fm2 = i_fm2[:, :, sub_y + dy:sub_y + self.block_size + dy, 0:self.block_size]
                            elif sub_x + dx + self.block_size > w:
                                sub_fm2 = i_fm2[:, :, sub_y + dy:sub_y + self.block_size + dy,
                                          w - self.block_size:w]
                            else:
                                sub_fm2 = i_fm2[:, :, sub_y + dy:sub_y + self.block_size + dy,
                                          sub_x + dx:sub_x + self.block_size + dx]

                        for a in range(-self.angle, self.angle + 1):
                            sub_fm2_b, sub_fm2_c, sub_fm2_h, sub_fm2_w = sub_fm2.shape
                            mask = torch.ones_like(sub_fm2, dtype=i_fm1.dtype, device=i_fm1.device)
                            radian_a = a * math.pi / 180.
                            theta = generate_theta(radian_a, 0, 0, sub_fm2_b, sub_fm2_h, sub_fm2_w, i_fm1.dtype)
                            grid = F.affine_grid(theta, sub_fm2.size(), align_corners=True).to(i_fm1.device)
                            r_sub_fm2 = F.grid_sample(sub_fm2, grid, align_corners=True)
                            r_mask = F.grid_sample(mask, grid, align_corners=True)
                            sub_mean_se = mse_loss(sub_fm1, r_sub_fm2, r_mask)
                            if sub_affine == 0:
                                sub_min_dist = sub_mean_se
                            else:
                                sub_min_dist = torch.vstack([sub_min_dist, sub_mean_se])
                            sub_affine += 1

                sub_min_dist, _ = torch.min(sub_min_dist, dim=0)
                if n_affine == 0:
                    min_dist = sub_min_dist
                else:
                    min_dist = torch.vstack([min_dist, sub_min_dist])
                n_affine += 1
        if self.training:
            min_dist = torch.sum(min_dist, dim=0)
        else:
            min_dist,_ = torch.topk(min_dist, k=self.topk, dim=0, largest=False)
            min_dist = torch.sum(min_dist, dim=0)

        return min_dist


class ShiftedLoss(torch.nn.Module):
    def __init__(self, hshift, vshift, topk=-1):
        super(ShiftedLoss, self).__init__()
        self.hshift = hshift
        self.vshift = vshift
        self.topk = topk

    def mse_loss(self, src, target):
        if isinstance(src, torch.autograd.Variable):
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.data.nelement() * src.size(0)
        else:
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.nelement() * src.size(0)

    def forward(self, fm1, fm2):
        # C * H * W
        bs, _, h, w = fm1.size()
        if w < self.hshift:
            self.hshift = self.hshift % w
        if h < self.vshift:
            self.vshift = self.vshift % h

        # min_dist shape (bs, )  & sys.float_info.max is to get the max float value
        # min_dist save the maximal float value
        min_dist = torch.ones(bs).cuda() * sys.float_info.max
        if isinstance(fm1, torch.autograd.Variable):
            min_dist = Variable(min_dist, requires_grad=False)

        if self.hshift == 0 and self.vshift == 0:
            dist = self.mse_loss(fm1, fm2).cuda()
            min_dist, _ = torch.min(torch.stack([min_dist, dist]), 0)
            return min_dist

        for bh in range(-self.hshift, self.hshift + 1):
            for bv in range(-self.vshift, self.vshift + 1):
                if bh >= 0:
                    ref1, ref2 = fm1[:, :, :, :w - bh], fm2[:, :, :, bh:]
                else:
                    ref1, ref2 = fm1[:, :, :, -bh:], fm2[:, :, :, :w + bh]

                if bv >= 0:
                    ref1, ref2 = ref1[:, :, :h - bv, :], ref2[:, :, bv:, :]
                else:
                    ref1, ref2 = ref1[:, :, -bv:, :], ref2[:, :, :h + bv, :]
                dist = self.mse_loss(ref1, ref2).cuda()
                min_dist, _ = torch.min(torch.stack([min_dist.squeeze(), dist.squeeze()]), 0)
        return min_dist.squeeze()


class ExtendedTrippletLoss(torch.nn.Module):
    def __init__(self, hshift, vshift, margin_dist):
        super(ExtendedTrippletLoss, self).__init__()
        self.shifted_loss = ShiftedLoss(hshift, vshift)
        self.margin_dist = margin_dist

    def forward(self, anchor, pos, neg):
        dist = self.shifted_loss(anchor, pos) - self.shifted_loss(anchor, neg) + self.margin_dist
        loss = torch.clamp(dist, min=0)
        return torch.sum(loss) / anchor.size(0)


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        if isinstance(kernel_size, int):
            reflection_padding = kernel_size // 2
        elif isinstance(kernel_size, tuple):
            assert (len(kernel_size) == 2)
            # (paddingLeft, paddingRight, paddingTop, paddingBottom)
            reflection_padding = (kernel_size[1], kernel_size[1], kernel_size[0], kernel_size[0])
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.UpsamplingNearest2d(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class DeformableConv2d1v(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
        super(DeformableConv2d1v, self).__init__()

        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = torch.nn.Conv2d(in_channels,
                                           2 * kernel_size * kernel_size,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=self.padding,
                                           bias=True)

        torch.nn.init.constant_(self.offset_conv.weight, 0.)
        torch.nn.init.constant_(self.offset_conv.bias, 0.)

        self.regular_conv = torch.nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)

        x = torchvision.ops.deform_conv2d(input=x.float(),
                                          offset=offset.float(),
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=(self.padding, self.padding),
                                          stride=self.stride,
                                          )
        return x


class DeformableConv2d2v(torch.nn.Module):
    """
    OFFICIAL: https://github.com/msracver/Deformable-ConvNets/tree/master/DCNv2_op
    MMDETECTION: https://github.com/open-mmlab/mmdetection/tree/master/configs/dcnv2
    CHENGDAZHI: https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d2v, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = torch.nn.Conv2d(in_channels,
                                           2 * kernel_size[0] * kernel_size[1],
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=self.padding,
                                           bias=True)

        torch.nn.init.constant_(self.offset_conv.weight, 0.)
        torch.nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = torch.nn.Conv2d(in_channels,
                                              1 * kernel_size[0] * kernel_size[1],
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=self.padding,
                                              bias=True)

        torch.nn.init.constant_(self.modulator_conv.weight, 0.)
        torch.nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = torch.nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 1. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(torch.nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(torch.nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # linear projection of flattened patches (Embedding)
        self.proj = torch.nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else torch.nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(torch.nn.Module):
    """
        MultiHead(Q, K, V) = Concat(HEAD1,...,HEADh)W0
        where HEAD1 = Attention(QWQi, kWKi, VWVi)
    """

    def __init__(self,
                 dim,  # input token dimension
                 num_heads=8,
                 qkv_bias=False,  # whether add bias for qkv
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        # multi-head
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # you can also use 3 different fc layer to calculate qkv
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop_ratio)
        # MultiHead(Q, K, V) = Concat(HEAD1,...,HEADh)W0
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # num_patches + 1 the 1 is the class token
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # softmax for each row
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(torch.nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=torch.nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(torch.nn.Module):
    """
    Encoder Block
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=torch.nn.GELU,
                 norm_layer=torch.nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else torch.nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp_ration = 4, the first fc layer will output 4 * input_dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LKA(torch.nn.Module):
    def __init__(self, dim):
        super(LKA, self).__init__()
        # depth-wise convolution
        self.conv0 = torch.nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        # depth-wise dilation convolution
        self.conv_spatial = torch.nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # channel convolution (1x1 convolution)
        self.conv1 = torch.nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x
