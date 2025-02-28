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
from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def is_static(index, splits):
    for i, split in enumerate(splits):
        if index >= split[0] and index <= split[1]:
            return i % 2 == 0
    raise ValueError('Index not in any split')

def compute_metrics(renders, gts, testing_indices, splits, static_eval):
    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        test_idx = testing_indices[idx]
        if static_eval and is_static(test_idx, splits):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        elif not static_eval and not is_static(test_idx, splits):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
    
    ssims = torch.tensor(ssims).mean()
    psnrs = torch.tensor(psnrs).mean()
    lpipss = torch.tensor(lpipss).mean()
    return ssims, psnrs, lpipss


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
        try:
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

            test_dir = Path(model_path) / "test"

            for method in os.listdir(test_dir):
                if not method.startswith("ours"):
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

                static_ssim, static_psnr, static_lpips = compute_metrics(renders, gts, testing_indices, splits, static_eval=True)
                dynamic_ssim, dynamic_psnr, dynamic_lpips = compute_metrics(renders, gts, testing_indices, splits, static_eval=False)

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
        except Exception as e:
            print(e)
            print("Unable to compute metrics for model", model_path)

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
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--source_paths', '-s', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.source_paths, args.model_paths)
