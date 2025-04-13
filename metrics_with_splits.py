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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from torch import Tensor
from lpips import LPIPS
import warnings
from typing import List, Optional, Tuple, Union

import json
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import cv2 as cv


lpips_fn = LPIPS(net='vgg', spatial=True).cuda()
def lpips(img1, img2, mask):
    if np.sum(mask) == 0:
        return None
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float().cuda()
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float().cuda()
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float().cuda()
    _lpips = lpips_fn(img1 * 2 - 1, img2 * 2 - 1)
    _lpips = _lpips[mask == 1].mean()
    return _lpips.cpu().item()


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)

def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out

def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        mask (torch.Tensor): boolean mask same size as X and Y

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if mask is not None:
        # Interpolate the mask to match the size of ssim_map and cs_map
        mask_resized = F.interpolate(mask, size=ssim_map.shape[-2:], mode='bilinear')

        # Flatten the ssim_map, cs_map, and resized mask
        ssim_map_flat = torch.flatten(ssim_map, 2)
        cs_map_flat = torch.flatten(cs_map, 2)
        mask_flat = torch.flatten(mask_resized, 2)

        # Apply the mask to the flattened ssim_map and cs_map
        masked_ssim_map = torch.masked_select(ssim_map_flat, mask_flat.bool())
        masked_cs_map = torch.masked_select(cs_map_flat, mask_flat.bool())

        # Compute the mean of the masked ssim_map and cs_map
        masked_ssim_per_channel = masked_ssim_map.mean(-1)
        masked_cs = masked_cs_map.mean(-1)

        return masked_ssim_per_channel, masked_cs
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim_torch(
    X: Tensor,
    Y: Tensor,
    data_range: float = 1.0,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
    mask: Optional[Tensor] = None,
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
        mask (torch.Tensor): boolean mask same size as X and Y

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    if mask is not None and mask.shape != X.shape:
        raise ValueError(f"Input mask should have the same dimensions as input images, but got {mask.shape} and {X.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)
        if mask is not None:
            mask = mask.squeeze(dim=d)

    #if mask is not None:
    #    assert size_average is True, "per channel ssim is not available if mask exist"
    #    margin = win_size // 2
    #    mask = mask[..., margin:-margin, margin:-margin]

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K, mask=mask)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    #if mask is not None:
    #    return ssim_per_channel
    #elif size_average:
    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ssim(img1, img2, mask):
    if np.sum(mask) == 0:
        return None
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float()
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float()
    mask = torch.tensor(mask).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float()
    return ssim_torch(img1, img2, mask=mask).item()


def psnr(img1, img2, mask):
    if np.sum(mask) == 0:
        return None
    img1 = np.reshape(img1, (-1, ))
    img2 = np.reshape(img2, (-1, ))
    mask = np.reshape(mask, (-1, ))
    mask = np.repeat(mask, 3)
    mask = mask > 0
    img1 = img1[mask]
    img2 = img2[mask]
    mse = ((img1 - img2) ** 2).mean()
    return 20 * np.log10(1.0 / np.sqrt(mse))


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = cv.imread(renders_dir / fname)
        gt = cv.imread(gt_dir / fname)
        renders.append(render / 255.0)
        gts.append(gt / 255.0)
        image_names.append(fname)
    return renders, gts, image_names

def is_static(index, splits):
    for i, split in enumerate(splits):
        if index >= split[0] and index <= split[1]:
            return i % 2 == 0
    raise ValueError('Index not in any split')

def compute_metrics(renders, gts, masks, testing_indices, splits, static_eval):
    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        test_idx = testing_indices[idx]
        if static_eval and is_static(test_idx, splits):
            ssims.append(ssim(renders[idx], gts[idx], masks[idx]))
            psnrs.append(psnr(renders[idx], gts[idx], masks[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], masks[idx]))
        elif not static_eval and not is_static(test_idx, splits):
            ssims.append(ssim(renders[idx], gts[idx], masks[idx]))
            psnrs.append(psnr(renders[idx], gts[idx], masks[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], masks[idx]))
    
    ssims = torch.tensor(ssims).mean()
    psnrs = torch.tensor(psnrs).mean()
    lpipss = torch.tensor(lpipss).mean()
    return ssims, psnrs, lpipss


def get_masks(path):
    masks = sorted(os.listdir(path))
    masks = [cv.imread(os.path.join(path, mask))[:, :, 0] != 25 for mask in masks]
    return masks

def evaluate(source_paths, model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    static_ssims = []
    static_psnrs = []
    static_lpipss = []
    dynamic_ssims = []
    dynamic_psnrs = []
    dynamic_lpipss = []
    assert len(source_paths) == len(model_paths), f'Number of source paths ({len(source_paths)}) must match number of model paths ({len(model_paths)})'
    for (source_path, model_path) in zip(source_paths, model_paths):
        print("Scene:", model_path)
        full_dict[model_path] = {}
        per_view_dict[model_path] = {}
        full_dict_polytopeonly[model_path] = {}
        per_view_dict_polytopeonly[model_path] = {}

        with open(os.path.join(source_path, 'split', 'phase_frame_index.txt'), 'r') as f:
            splits = f.read().split('\n')
        splits = [list(map(int, split.split(','))) for split in splits]
        with open(os.path.join(source_path, 'frames.txt'), 'r') as f:
            frames_indices = f.read().strip().split('-')
        frames_indices = (int(frames_indices[0]), int(frames_indices[1]))
        all_indices = list(range(frames_indices[0], frames_indices[1] + 1))
        testing_indices = all_indices[1::2]
        masks = get_masks(os.path.join(source_path, 'hand_masks'))[1::2]
        test_dir = Path(model_path) / "test"

        for method in os.listdir(test_dir):
            if not method.startswith("ours"):
                continue
            if not method.endswith('40000'):
                continue
            print("Method:", method)

            full_dict[model_path][method] = {}
            per_view_dict[model_path][method] = {}
            full_dict_polytopeonly[model_path][method] = {}
            per_view_dict_polytopeonly[model_path][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            static_ssim, static_psnr, static_lpips = compute_metrics(renders, gts, masks, testing_indices, splits, static_eval=True)
            dynamic_ssim, dynamic_psnr, dynamic_lpips = compute_metrics(renders, gts, masks, testing_indices, splits, static_eval=False)

            print("Static evaluation:")
            print("  SSIM : {:>12.7f}".format(static_ssim.item(), ".5"))
            print("  PSNR : {:>12.7f}".format(static_psnr.item(), ".5"))
            print("  LPIPS: {:>12.7f}".format(static_lpips.item(), ".5"))
            print("")
            print("Dynamic evaluation:")
            print("  SSIM : {:>12.7f}".format(dynamic_ssim.item(), ".5"))
            print("  PSNR : {:>12.7f}".format(dynamic_psnr.item(), ".5"))
            print("  LPIPS: {:>12.7f}".format(dynamic_lpips.item(), ".5"))
            print("")

            static_ssims.append(static_ssim)
            static_psnrs.append(static_psnr)
            static_lpipss.append(static_lpips)
            dynamic_ssims.append(dynamic_ssim)
            dynamic_psnrs.append(dynamic_psnr)
            dynamic_lpipss.append(dynamic_lpips)

    print("Static evaluation:")
    print("  SSIM : {:>12.7f}".format(torch.tensor(static_ssims).mean().item(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(static_psnrs).mean().item(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(static_lpipss).mean().item(), ".5"))
    print("")
    print("Dynamic evaluation:")
    print("  SSIM : {:>12.7f}".format(torch.tensor(dynamic_ssims).mean().item(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(dynamic_psnrs).mean().item(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(dynamic_lpipss).mean().item(), ".5"))
    print("")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--source_paths', '-s', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.source_paths, args.model_paths)
