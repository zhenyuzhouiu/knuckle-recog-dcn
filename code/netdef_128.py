# ========================================================= 
# @ Network Architecture File: Residual Feature Network 128
# @ Target dataset: IITD, PolyU 2D/3D Contactless, CASIA, 
#   300 Subject, 600 Subject, 35 Subject
# =========================================================

from __future__ import division
import sys
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import net_common
from functools import partial
from collections import OrderedDict
import math
from net_common import ConvLayer, ResidualBlock, DeformableConv2d1v, DeformableConv2d2v, LKA


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.zeros_(m.bias)
        torch.nn.init.ones_(m.weight)


class ResidualFeatureNet(torch.nn.Module):
    def __init__(self):
        super(ResidualFeatureNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.resid3 = ResidualBlock(128)
        self.resid4 = ResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = F.relu(self.conv4(resid4))
        conv5 = F.relu(self.conv5(conv4))

        return conv5



class DRFNet(torch.nn.Module):
    def __init__(self):
        super(DRFNet, self).__init__()
        # Initial convolution layers
        self.conv1 = DeformableConv2d2v(1, 32, kernel_size=5, stride=2)
        self.conv2 = DeformableConv2d2v(32, 64, kernel_size=3, stride=2)
        self.conv3 = DeformableConv2d2v(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.resid3 = ResidualBlock(128)
        self.resid4 = ResidualBlock(128)
        self.conv4 = DeformableConv2d2v(128, 64, kernel_size=3, stride=1)
        self.conv5 = DeformableConv2d2v(64, 1, kernel_size=1, stride=1)

    def forward(self, X):
        conv1 = F.relu(self.conv1(X))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid2(resid1)
        resid3 = self.resid3(resid2)
        resid4 = self.resid4(resid3)
        conv4 = F.relu(self.conv4(resid4))
        conv5 = F.relu(self.conv5(conv4))
        return conv5


class CTNet(torch.nn.Module):
    def __init__(self):
        super(CTNet, self).__init__()
        self.conv1 = ConvLayer(3, )


class TNet_16(torch.nn.Module):
    """
       depth: the number of attention layer
       representation_size: the number of final MLP Head's Pre-Logits node,
                            if the value is none, then the head will not construct the Pre-Logits layer
       """

    def __init__(self, img_size=128, patch_size=16, in_c=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=net_common.PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(TNet_16, self).__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(torch.nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or torch.nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # trainable position embedding
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = torch.nn.Dropout(p=drop_ratio)

        # drop_path_ratio is the number layer of Encoder Block
        # torch.linspace(start, end, steps)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = torch.nn.Sequential(*[
            net_common.Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                             norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.conv = torch.nn.Conv2d(embed_dim, 1, 1, 1)
        # Weight init
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 64, 768]

        # directly element add
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, 8, 8)
        x = self.conv(x)
        # x = F.relu(x)

        return x


class TNet_8(torch.nn.Module):
    """
         depth: the number of attention layer
         representation_size: the number of final MLP Head's Pre-Logits node,
                              if the value is none, then the head will not construct the Pre-Logits layer
         """

    def __init__(self, img_size=128, patch_size=8, in_c=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=net_common.PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(TNet_8, self).__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(torch.nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or torch.nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # trainable position embedding
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = torch.nn.Dropout(p=drop_ratio)

        # drop_path_ratio is the number layer of Encoder Block
        # torch.linspace(start, end, steps)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = torch.nn.Sequential(*[
            net_common.Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                             norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.conv = torch.nn.Conv2d(embed_dim, 1, 1, 1)
        # Weight init
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 256, 768]

        # directly element add
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, 16, 16)
        x = self.conv(x)
        x = F.relu(x)

        return x


class CTNet(torch.nn.Module):
    def __init__(self, norm_layer=None, act_layer=None):
        super(CTNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)

        norm_layer = norm_layer or partial(torch.nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or torch.nn.GELU
        self.patch_embed1 = net_common.PatchEmbed(img_size=64, patch_size=8, in_c=128, embed_dim=64)
        num_patches1 = self.patch_embed1.num_patches
        self.pos_embed1 = torch.nn.Parameter(torch.zeros(1, num_patches1, 64))
        self.pos_drop1 = torch.nn.Dropout(p=0.)
        self.blocks1 = torch.nn.Sequential(*[
            net_common.Block(dim=64, num_heads=2, mlp_ratio=4, qkv_bias=True,
                             qk_scale=None,
                             drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                             norm_layer=norm_layer, act_layer=act_layer)
            for i in range(1)
        ])
        self.norm1 =norm_layer(64)
        self.conv3 = torch.nn.Conv2d(64, 1, 1, 1)


        torch.nn.init.trunc_normal_(self.pos_embed1, std=0.02)

        self.apply(_init_vit_weights)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.patch_embed1(x)
        x = self.pos_drop1(x + self.pos_embed1)
        x = self.blocks1(x)
        x = self.norm1(x)

        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, 8, 8)
        x = self.conv3(x)
        return x

class OldCTNet(torch.nn.Module):
    def __init__(self, norm_layer=None, act_layer=None):
        super(CTNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.maxpooling = torch.nn.MaxPool2d(2, 2)

        norm_layer = norm_layer or partial(torch.nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or torch.nn.GELU
        self.patch_embed1 = net_common.PatchEmbed(img_size=64, patch_size=2, in_c=64, embed_dim=256)
        num_patches1 = self.patch_embed1.num_patches
        self.pos_embed1 = torch.nn.Parameter(torch.zeros(1, num_patches1, 256))
        self.pos_drop1 = torch.nn.Dropout(p=0.)
        self.blocks1 = torch.nn.Sequential(*[
            net_common.Block(dim=256, num_heads=2, mlp_ratio=4, qkv_bias=True,
                             qk_scale=None,
                             drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                             norm_layer=norm_layer, act_layer=act_layer)
            for i in range(1)
        ])
        self.norm1 =norm_layer(256)

        self.conv = torch.nn.Conv2d(256, 1, 1, 1)

        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.patch_embed2 = net_common.PatchEmbed(img_size=16, patch_size=2, in_c=128, embed_dim=128)
        num_patches2 = self.patch_embed2.num_patches
        self.pos_embed2 = torch.nn.Parameter(torch.zeros(1, num_patches2, 128))
        self.pos_drop2 = torch.nn.Dropout(p=0.)
        self.blocks2 = torch.nn.Sequential(*[
            net_common.Block(dim=128, num_heads=1, mlp_ratio=4, qkv_bias=True,
                             qk_scale=None,
                             drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                             norm_layer=norm_layer, act_layer=act_layer)
            for i in range(1)
        ])
        self.norm2 = norm_layer(128)

        self.conv4 = torch.nn.Conv2d(128, 1, 1, 1)

        torch.nn.init.trunc_normal_(self.pos_embed1, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed2, std=0.02)

        # self.apply(_init_vit_weights)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.patch_embed1(x)
        x = self.pos_drop1(x + self.pos_embed1)
        x = self.blocks1(x)
        x = self.norm1(x)

        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, 32, 32)
        y = F.relu(self.conv(x))

        # x = F.relu(self.conv3(x))
        #
        # x = self.patch_embed2(x)
        # x = self.pos_drop2(x + self.pos_embed2)
        # x = self.blocks2(x)
        # x = self.norm2(x)
        #
        # B, HW, C = x.shape
        # x = x.transpose(1,2).view(B, C, 8, 8)
        #
        # x = F.relu(self.conv4(x))
        return y


class DCLAKNet(torch.nn.Module):
    def __init__(self):
        super(DCLAKNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.lka1= LKA(dim=128)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.lka2 = LKA(dim=128)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.lka1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.lka2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x


class DConvAttentionCorr(torch.nn.Module):
    def __init__(self):
        super(DConvAttentionCorr, self).__init__()
        self.dconv1 = net_common.DeformableConv2d2v(in_channels=3, out_channels=64)
        self.dconv2 = net_common.DeformableConv2d2v(in_channels=64, out_channels=128)




