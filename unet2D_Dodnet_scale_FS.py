import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
affine_par = True
import functools

import sys, os
##################################################################################################
from dataclasses import dataclass
#from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from typing import Any, Dict, List, Optional, Tuple, Union
#from typing import Literal

import einops as E
from universeg.nn import CrossConv2d
from universeg.nn import reset_conv2d_parameters
from universeg.nn import Vmap, vmap
from universeg.validation import (Kwargs, as_2tuple, size2t,
                         #validate_arguments,
                         validate_arguments_init)

from pydantic import validate_arguments
##################################################################################################






in_place = True

class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size=(3,3,3), stride=(1,1,1), padding=1, dilation=1, bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)




class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.gn1 = nn.GroupNorm(16, inplanes)
        # self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=(1,1,1),
        #                         dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=stride, padding=(1,1),dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=in_place)

        self.gn2 = nn.GroupNorm(16, planes)
        # self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=(1,1,1),
        #                         dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=(1,1),dilation=1, bias=False)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class unet2D(nn.Module):
    def __init__(self, layers, num_classes=3, weight_std = False):
        self.inplanes = 128
        self.weight_std = weight_std
        super(unet2D, self).__init__()

        # self.conv1 = conv3x3x3(3, 32, stride=[1, 1, 1], weight_std=self.weight_std)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=(1,1),dilation=1, bias=False)

        # self.add_0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1))
        self.add_0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(2, 2))
        self.add_1 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(4, 4))

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            #conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=1, padding=(0,0),dilation=1, bias=False)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=(2, 2))
        self.upsamplex4 = nn.Upsample(scale_factor=(4, 4))

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1))
        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1))
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1))
        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))

        self.x1_resb_add0 = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))
        self.x1_resb_add1 = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))


        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            # nn.Conv3d(32, 8, kernel_size=1)
            nn.Conv2d(32, 8, kernel_size=(1, 1))
        )

        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool2d((1,1))
        )

        # self.controller_task = nn.Conv2d(256 + 6, 90, kernel_size=1, stride=1, padding=0)   #### change the channel
        # self.controller_scale = nn.Conv2d(256 + 3, 72, kernel_size=1, stride=1, padding=0)
        #self.controller_trilinear = nn.Conv2d(256 * 6 * 4, 162, kernel_size=1, stride=1, padding=0)
        ################################################################################################
        self.controller_trilinear = nn.Conv2d(256 * 7 * 4, 162, kernel_size=1, stride=1, padding=0)
        ###############################################################################################
        ################################################################################################
        #in_ch = (3, 4)
        block_kws = dict(cross_kws=dict(nonlinearity=None))
        self.block = CrossBlock((3, 4), 32, 32, **block_kws)
        self.block0 = CrossBlock((32, 32), 32, 32, **block_kws)
        self.block1 = CrossBlock((64, 64), 64, 64, **block_kws)
        self.block2 = CrossBlock((128, 128), 128, 128, **block_kws)
        self.block3 = CrossBlock((256, 256), 256, 256, **block_kws)
        # self.block4 = CrossBlock(in_ch, cross_ch, conv_ch, **block_kws)
        ################################################################################################

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                # conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                #           weight_std=self.weight_std),
                nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=stride, padding=(0, 0), dilation=1, bias=False)
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        #task_encoding = torch.zeros(size=(N, 6))   #### change the channel
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i].long()]=1
        return task_encoding.cuda()

    def encoding_scale(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 4))   #### change the channel
        for i in range(N):
            task_encoding[i, task_id[i].long()]=1
        return task_encoding.cuda()

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward_v0(self, input, task_id, scale_id):
        x = self.conv1(input)

        x = self.layer0(x)
        skip0 = x

        x = self.layer1(x)
        skip1 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        skip3 = x

        x = self.layer4(x)

        x = self.fusionConv(x)

        # generate conv filters for classification layer
        task_encoding = self.encoding_task(task_id)
        # print(task_id)
        task_encoding.unsqueeze_(2).unsqueeze_(2)#.unsqueeze_(2)

        scale_encoding = self.encoding_scale(scale_id)
        scale_encoding.unsqueeze_(2).unsqueeze_(2)

        x_feat = self.GAP(x)

        #x_cond = torch.zeros((len(scale_encoding), 256 * 6 * 4, 1, 1)).cuda().float()
        ####################################################################
        x_cond = torch.zeros((len(scale_encoding), 256 * 7 * 4, 1, 1)).cuda().float()
        ####################################################################
        for xi in range(len(x_feat)):
            now_x_feat = x_feat[xi].squeeze(-1).squeeze(-1)
            now_task_encoding = task_encoding[xi].squeeze(-1).squeeze(-1)
            now_scale_encoding = scale_encoding[xi].squeeze(-1).squeeze(-1)
            x_cond[xi] = torch.outer(torch.outer(now_x_feat, now_task_encoding).view(1, -1).squeeze(0), now_scale_encoding).view(1,-1).unsqueeze(-1).unsqueeze(-1)

        params = self.controller_trilinear(x_cond)
        params.squeeze_(-1).squeeze_(-1)
        # x_cond_task = torch.cat([x_feat, task_encoding], 1)
        # params_task = self.controller_task(x_cond_task)
        # params_task.squeeze_(-1).squeeze_(-1)#.squeeze_(-1)
        #
        # x_cond_scale = torch.cat([x_feat, scale_encoding], 1)
        # params_scale = self.controller_scale(x_cond_scale)
        # params_scale.squeeze_(-1).squeeze_(-1)  # .squeeze_(-1)

        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)

        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)         # (32, 128, 256)

        head_inputs = self.precls_conv(x)

        feature_maps = head_inputs

        N, _, H, W = head_inputs.size()
        head_inputs = head_inputs.reshape(1, -1, H, W)

        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*2)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(2)

        weights, biases = self.parse_dynamic_params(params, 8, weight_nums, bias_nums)

        logits = self.heads_forward(head_inputs, weights, biases, N)

        logits = logits.reshape(-1, 2, H, W)

        return logits, x_feat

#####################################################################################
    def forward(self, input, support_images, support_labels, task_id, scale_id):

        #target_image,        # (B, 1, H, W)
        #support_images,      # (B, S, 1, H, W)
        #support_labels,      # (B, S, 1, H, W)
        #
        #target = E.rearrange(input, "B 1 H W -> B 1 1 H W")
        #"B C H W -> B 1 C H W"
        target = torch.unsqueeze(input, dim=1)
        support = torch.cat([support_images, support_labels], dim=2)

        pass_through = []

        #x = self.conv1(input)
        #######################################################
        target, support = self.block(target, support)
        #target = vmap(self.conv1, target)
        #support = vmap(self.conv1, support)
        #y = self.conv1(support)



        #x = self.layer0(x)
        #skip0 = x

        ########################################################
        target = vmap(self.layer0, target)
        support = vmap(self.layer0, support)
        target, support = self.block0(target, support)
        target_skip0 = target
        support_skip0 = support
        ########################################################

        #x = self.layer1(x)
        #skip1 = x

        ########################################################
        #y1 = self.layer1(torch.squeeze(support, dim=0))
        target = vmap(self.layer1, target)
        support = vmap(self.layer1, support)
        target, support = self.block1(target, support)
        target_skip1 = target
        support_skip1 = support
        ########################################################


        #x = self.layer2(x)
        #skip2 = x

        ########################################################
        target = vmap(self.layer2, target)
        support = vmap(self.layer2, support)
        target, support = self.block2(target, support)
        target_skip2 = target
        support_skip2 = support
        ########################################################



        #x = self.layer3(x)
        #skip3 = x

        ########################################################
        target = vmap(self.layer3, target)
        support = vmap(self.layer3, support)
        target, support = self.block3(target, support)
        target_skip3 = target
        support_skip3 = support
        ########################################################



        #x = self.layer4(x)

        ########################################################
        target = vmap(self.layer4, target)
        support = vmap(self.layer4, support)
        #target, support = self.block4(target, support)

        ########################################################

        #x = self.fusionConv(x)

        ########################################################
        target = vmap(self.fusionConv, target)
        support = vmap(self.fusionConv, support)
        ########################################################
        # generate conv filters for classification layer
        task_encoding = self.encoding_task(task_id)
        # print(task_id)
        task_encoding.unsqueeze_(2).unsqueeze_(2)#.unsqueeze_(2)

        scale_encoding = self.encoding_scale(scale_id)
        scale_encoding.unsqueeze_(2).unsqueeze_(2)

        #x_feat = self.GAP(x)

        #############################################################################
        target_feat = vmap(self.GAP, target)
        support_feat = vmap(self.GAP, support)
        #############################################################################
        #x_cond = torch.zeros((len(scale_encoding), 256 * 6 * 4, 1, 1)).cuda().float()
        ####################################################################
        #x_cond = torch.zeros((len(scale_encoding), 256 * 7 * 4, 1, 1)).cuda().float()
        target_cond = torch.zeros((len(scale_encoding), 256 * 7 * 4, 1, 1)).cuda().float()
        ####################################################################
        # for xi in range(len(x_feat)):
        #     now_x_feat = x_feat[xi].squeeze(-1).squeeze(-1)
        #     now_task_encoding = task_encoding[xi].squeeze(-1).squeeze(-1)
        #     now_scale_encoding = scale_encoding[xi].squeeze(-1).squeeze(-1)
        #     x_cond[xi] = torch.outer(torch.outer(now_x_feat, now_task_encoding).view(1, -1).squeeze(0), now_scale_encoding).view(1,-1).unsqueeze(-1).unsqueeze(-1)
        #
        # params = self.controller_trilinear(x_cond)
        # params.squeeze_(-1).squeeze_(-1)
        # x_cond_task = torch.cat([x_feat, task_encoding], 1)
        # params_task = self.controller_task(x_cond_task)
        # params_task.squeeze_(-1).squeeze_(-1)#.squeeze_(-1)
        #
        # x_cond_scale = torch.cat([x_feat, scale_encoding], 1)
        # params_scale = self.controller_scale(x_cond_scale)
        # params_scale.squeeze_(-1).squeeze_(-1)  # .squeeze_(-1)
        ########################################################################
        target_feat = torch.squeeze(target_feat, dim=0)
        for ti in range(len(target_feat)):
            now_target_feat = target_feat[ti].squeeze(-1).squeeze(-1)
            now_task_encoding = task_encoding[ti].squeeze(-1).squeeze(-1)
            now_scale_encoding = scale_encoding[ti].squeeze(-1).squeeze(-1)
            target_cond[ti] = torch.outer(torch.outer(now_target_feat, now_task_encoding).view(1, -1).squeeze(0), now_scale_encoding).view(1,-1).unsqueeze(-1).unsqueeze(-1)
            #target_cond[ti] = vmap()
        params = self.controller_trilinear(target_cond)
        params.squeeze_(-1).squeeze_(-1)
        ########################################################################
        # x8
        #x = self.upsamplex2(x)
        #x = x + skip3
        #x = self.x8_resb(x)
        ################################################################################
        target = vmap(self.upsamplex2, target)
        support = vmap(self.upsamplex2, support)

        target = target + target_skip3
        support = support + target_skip3


        target = vmap(self.x8_resb, target)
        support = vmap(self.x8_resb, support)

        target, support = self.block2(target, support)
        ################################################################################
        # x4
        #x = self.upsamplex2(x)
        #x = x + skip2
        #x = self.x4_resb(x)
        ################################################################################
        target = vmap(self.upsamplex2, target)
        support = vmap(self.upsamplex2, support)

        target = target + target_skip2
        support = support + target_skip2
        target = vmap(self.x4_resb, target)
        support = vmap(self.x4_resb, support)

        target, support = self.block1(target, support)
        ################################################################################
        # x2
        #x = self.upsamplex2(x)
        #x = x + skip1
        #x = self.x2_resb(x)
        ################################################################################
        target = vmap(self.upsamplex2, target)
        support = vmap(self.upsamplex2, support)

        target = target + target_skip1
        support = support + target_skip1
        target = vmap(self.x2_resb, target)
        support = vmap(self.x2_resb, support)

        target, support = self.block0(target, support)
        ################################################################################
        # x1
        #x = self.upsamplex2(x)
        #x = x + skip0
        #x = self.x1_resb(x)         # (32, 128, 256)
        ################################################################################
        target = vmap(self.upsamplex2, target)
        support = vmap(self.upsamplex2, support)

        target = target + target_skip0
        support = support + target_skip0
        target = vmap(self.x1_resb, target)
        support = vmap(self.x1_resb, support)

        target, support = self.block0(target, support)
        ################################################################################
        #head_inputs = self.precls_conv(x)
        ################################################################################
        #head_inputs = self.precls_conv(target)
        head_inputs = vmap(self.precls_conv, target)
        ################################################################################

        feature_maps = head_inputs

        ################################################################################
        head_inputs = torch.squeeze(head_inputs, dim=0)
        ################################################################################
        N, _, H, W = head_inputs.size()

        head_inputs = head_inputs.reshape(1, -1, H, W)

        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*2)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(2)

        weights, biases = self.parse_dynamic_params(params, 8, weight_nums, bias_nums)

        logits = self.heads_forward(head_inputs, weights, biases, N)

        logits = logits.reshape(-1, 2, H, W)

        return logits, target_feat
    def forward_v1(self, input, support_images, support_labels, task_id, scale_id):
        x = self.conv1(input)

        x = self.layer0(x)
        skip0 = x

        x = self.layer1(x)
        skip1 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        skip3 = x

        x = self.layer4(x)

        x = self.fusionConv(x)

        # generate conv filters for classification layer
        task_encoding = self.encoding_task(task_id)
        # print(task_id)
        task_encoding.unsqueeze_(2).unsqueeze_(2)#.unsqueeze_(2)

        scale_encoding = self.encoding_scale(scale_id)
        scale_encoding.unsqueeze_(2).unsqueeze_(2)

        x_feat = self.GAP(x)

        #x_cond = torch.zeros((len(scale_encoding), 256 * 6 * 4, 1, 1)).cuda().float()
        ####################################################################
        x_cond = torch.zeros((len(scale_encoding), 256 * 7 * 4, 1, 1)).cuda().float()
        ####################################################################
        for xi in range(len(x_feat)):
            now_x_feat = x_feat[xi].squeeze(-1).squeeze(-1)
            now_task_encoding = task_encoding[xi].squeeze(-1).squeeze(-1)
            now_scale_encoding = scale_encoding[xi].squeeze(-1).squeeze(-1)
            x_cond[xi] = torch.outer(torch.outer(now_x_feat, now_task_encoding).view(1, -1).squeeze(0), now_scale_encoding).view(1,-1).unsqueeze(-1).unsqueeze(-1)

        params = self.controller_trilinear(x_cond)
        params.squeeze_(-1).squeeze_(-1)
        # x_cond_task = torch.cat([x_feat, task_encoding], 1)
        # params_task = self.controller_task(x_cond_task)
        # params_task.squeeze_(-1).squeeze_(-1)#.squeeze_(-1)
        #
        # x_cond_scale = torch.cat([x_feat, scale_encoding], 1)
        # params_scale = self.controller_scale(x_cond_scale)
        # params_scale.squeeze_(-1).squeeze_(-1)  # .squeeze_(-1)

        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)

        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)         # (32, 128, 256)

        head_inputs = self.precls_conv(x)

        feature_maps = head_inputs

        N, _, H, W = head_inputs.size()
        head_inputs = head_inputs.reshape(1, -1, H, W)

        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*2)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(2)

        weights, biases = self.parse_dynamic_params(params, 8, weight_nums, bias_nums)

        logits = self.heads_forward(head_inputs, weights, biases, N)

        logits = logits.reshape(-1, 2, H, W)

        return logits, x_feat

#####################################################################################

def UNet2D(num_classes=1, weight_std=False):
    print("Using DynConv 8,8,2")
    model = unet2D([1, 2, 2, 2, 2], num_classes, weight_std)
    return model




def get_nonlinearity(nonlinearity: Optional[str]) -> nn.Module:
    if nonlinearity is None:
        return nn.Identity()
    if nonlinearity == "Softmax":
        # For Softmax, we need to specify the channel dimension
        return nn.Softmax(dim=1)
    if hasattr(nn, nonlinearity):
        return getattr(nn, nonlinearity)()
    raise ValueError(f"nonlinearity {nonlinearity} not found")


@validate_arguments_init
@dataclass(eq=False, repr=False)
class ConvOp(nn.Sequential):

    in_channels: int
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode="zeros",
            bias=True,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)

        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )


@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossOp(nn.Module):

    in_channels: size2t
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()

        self.cross_conv = CrossConv2d(
            in_channels=as_2tuple(self.in_channels),
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)

        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )

    def forward(self, target, support):
        interaction = self.cross_conv(target, support).squeeze(dim=1)

        if self.nonlinearity is not None:
            interaction = vmap(self.nonlin, interaction)

        new_target = interaction.mean(dim=1, keepdims=True)

        return new_target, interaction


@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossBlock(nn.Module):

    in_channels: size2t
    cross_features: int
    conv_features: Optional[int] = None
    cross_kws: Optional[Dict[str, Any]] = None
    conv_kws: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__init__()

        conv_features = self.conv_features or self.cross_features
        cross_kws = self.cross_kws or {}
        conv_kws = self.conv_kws or {}

        self.cross = CrossOp(self.in_channels, self.cross_features, **cross_kws)
        self.target = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))
        self.support = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))

    def forward(self, target, support):
        target, support = self.cross(target, support)
        target = self.target(target)
        support = self.support(support)
        return target, support
