from fastapi import Path
import imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import numpy as np
import torch
from pathlib import Path

def im2gray(ref_img):
    if ref_img.shape[-1] == 4:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGRA2RGB)
    elif ref_img.shape[-1] == 3:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    assert ref_img.shape[-1] == 3
    return ref_img

img_list = list(Path("submit").rglob("*.png")) + list(Path("submit").rglob("*.jpg"))
psnr_list, ssim_list = [], []

for name in img_list:
    im = cv2.imread(str(name))
    im = im2gray(im)
    ref_img, _, res_img = torch.chunk(torch.from_numpy(im), 3, -2)
    ref_img, res_img = ref_img.numpy(), res_img.numpy()

    psnr = peak_signal_noise_ratio(ref_img, res_img)
    ssim = structural_similarity(ref_img, res_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
    print(f"{str(name):30}, PSNR {psnr:7.4f}, SSIM {ssim:7.4f}")
    psnr_list.append(psnr), ssim_list.append(ssim)
    
print(f"pnsr avg {sum(psnr_list)/len(psnr_list):7.4f}, ssim avg {sum(ssim_list)/len(ssim_list)}")