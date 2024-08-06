import numpy as np
from skimage import io
import torch

from utils.loss_utils import ssim,l1_loss
from lpipsPyTorch import lpips
from utils.image_utils import psnr

import os

def evaluate_quality(render_path, gt_path):

    # Load two images: original and reconstructed
    original = io.imread(gt_path)
    render = io.imread(render_path)


    mask = original[:,:,3] > 0
    render_filtered = render * mask[:,:,None]


    print(np.transpose(original[:,:,:3],(2,0,1)).shape)
    print(np.transpose(render_filtered,(2,0,1)).shape)

    original = torch.tensor(np.transpose(original[:,:,:3],(2,0,1)), dtype= torch.float32, device='cuda')
    render_filtered = torch.tensor(np.transpose(render_filtered,(2,0,1)), dtype= torch.float32, device='cuda')


    # Calculate PSNR
    psnr_value_ = psnr( original,render_filtered).detach().cpu().numpy()

    # Calculate SSIM
    ssim_value_ = ssim(render_filtered, original).detach().cpu().numpy()


    # Calculate LPIPS
    lpips_value_ = lpips(render_filtered, original, net_type='vgg').detach().cpu().numpy()


    # Calculate L1 loss
    l1_value_ = l1_loss(render_filtered, original)

    ssim_value = torch.tensor(ssim_value_).mean()
    psnr_value = torch.tensor(psnr_value_).mean()
    lpips_value = torch.tensor(lpips_value_).mean()
    l1_value = l1_value_.detach().cpu().numpy()


    print("  SSIM : {:>12.7f}".format(ssim_value, ".5"))
    print("  PSNR : {:>12.7f}".format(psnr_value, ".5"))
    print("  LPIPS: {:>12.7f}".format(lpips_value, ".5"))
    print("  L1   : {:>12.7f}".format(l1_value, ".5"))

    return ssim_value, psnr_value, lpips_value, l1_value



if __name__ == "__main__":
    render_path = "/home/yanni/Thesis/gaussian-fabrication/3dgs_lego_train/mitsuba/render/z_z_high_spp/spec_000_xyz5e-3sqrt_0.01relu1d0.01_0.0025_surface_AS_AS_factor_xyz_10k_new/3000_optimize_64.png"
    gt_path = "/home/yanni/Thesis/gaussian-fabrication/lego/train/r_64.png"

    print("============================== 64 ==============================")
    ssim64,psnr64, lpips64,l164 = evaluate_quality(render_path, gt_path)

    render_path = "/home/yanni/Thesis/gaussian-fabrication/3dgs_lego_train/mitsuba/render/z_z_high_spp/spec_000_xyz5e-3sqrt_0.01relu1d0.01_0.0025_surface_AS_AS_factor_xyz_10k_new/3000_optimize_69.png"
    gt_path = "/home/yanni/Thesis/gaussian-fabrication/lego/train/r_69.png"

    print("============================== 69 ==============================")
    ssim69,psnr69, lpips69, l169 = evaluate_quality(render_path, gt_path)

    render_path = "/home/yanni/Thesis/gaussian-fabrication/3dgs_lego_train/mitsuba/render/z_z_high_spp/spec_000_xyz5e-3sqrt_0.01relu1d0.01_0.0025_surface_AS_AS_factor_xyz_10k_new/3000_optimize_73.png"
    gt_path = "/home/yanni/Thesis/gaussian-fabrication/lego/train/r_73.png"

    print("============================== 73 ==============================")
    ssim73,psnr73, lpips73,l173 = evaluate_quality(render_path, gt_path)

    render_path = "/home/yanni/Thesis/gaussian-fabrication/3dgs_lego_train/mitsuba/render/z_z_high_spp/spec_000_xyz5e-3sqrt_0.01relu1d0.01_0.0025_surface_AS_AS_factor_xyz_10k_new/3000_optimize_95.png"
    gt_path = "/home/yanni/Thesis/gaussian-fabrication/lego/train/r_95.png"

    print("============================== 95 ==============================")
    ssim95,psnr95, lpips95,l195 = evaluate_quality(render_path, gt_path)


    print("================================================================")

    print("SSIM: ", (ssim64+ssim69+ssim73+ssim95)/4.0)
    print("PSNR: ", (psnr64+psnr69+psnr73+psnr95)/4.0)
    print("LPIPS: ", (lpips64+lpips69+lpips73+lpips95)/4.0)
    print("L1: ", (l164+l169+l173+l195)/4.0)