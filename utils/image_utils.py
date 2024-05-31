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

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def warpped_depth(depth):
    return torch.where(depth < 10.0, depth / 10.0, 2.0 - 10 / depth) / 2.0

def warpped_depth_inv(depth):
    return torch.where(depth < 0.5, 2 * depth * 10.0, 10 / (1.0 - depth) / 2)

def psnr_with_mask(img1, img2, mask):
    # 创建一个mask，其中非零元素对应的位置为True，零元素对应的位置为False
    mask_bool = mask.repeat(3, 1, 1) != 0

    # 使用mask选择img1和img2中的元素，然后计算他们的平方差值
    mse = ((img1[mask_bool] - img2[mask_bool]) ** 2).mean()

    # 计算PSNR
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

    return psnr