import numpy as np
from skimage import io
import torch

from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr



if __name__ == "__main__":

    render_path = "/home/yanni/Thesis/gaussian-fabrication/3dgs_lego_train/mitsuba/render/z_correct_xyz_sqrt_albedo_spectral/xyz5e-3sqrt_0.001relu1d0.01_1500_AS/view1_1.png"
    # render_path = "3dgs_lego_train/mitsuba/render/z_correct_xyz_sqrt_albedo/xyz5e-3sqrt_0.01relu1d0.015_1500_A_light2/3000_optimize_73.png"
    gt_path = "/home/yanni/Thesis/gaussian-fabrication/lego/train/r_73.png"

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
    psnr_value = psnr(render_filtered, original).detach().cpu().numpy()

    # Calculate SSIM
    ssim_value = ssim(render_filtered, original).detach().cpu().numpy()


    # Calculate LPIPS
    lpips_value = lpips(render_filtered, original, net_type='vgg').detach().cpu().numpy()

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssim_value).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnr_value).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpips_value).mean(), ".5"))