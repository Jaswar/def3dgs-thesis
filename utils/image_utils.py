#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2, mask):
    if mask.dim() == 2:  # no batching and channel dimension
        mask = mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:  # masks are batched but no channel dimension
        mask = mask.unsqueeze(1)
    error_mask = (img1 * mask - img2 * mask) ** 2
    error_mask = error_mask.view(img1.shape[0], -1)
    mse = error_mask.sum(1, keepdim=True)
    mask = mask.view(mask.shape[0], -1)
    regularizer = mask.sum(1, keepdim=True) * 3  # 3 is the number of channels
    mse = mse / regularizer
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

    # mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    # return 20 * torch.log10(1.0 / torch.sqrt(mse))
