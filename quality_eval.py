import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io
import lpips
import torch



if __name__ == "__main__":

    render_path = "/home/yanni/Thesis/gaussian-fabrication/3dgs_lego_train/mitsuba/render/z_correct_xyz_sqrt_albedo_spectral/xyz5e-3sqrt_0.001relu1d0.01_1500_AS/3000_optimize_73.png"
    # render_path = "3dgs_lego_train/mitsuba/render/z_correct_xyz_sqrt_albedo/xyz5e-3sqrt_0.01relu1d0.015_1500_A_light2/3000_optimize_73.png"
    gt_path = "/home/yanni/Thesis/gaussian-fabrication/lego/train/r_73.png"

    # Load two images: original and reconstructed
    original = io.imread(gt_path)
    render = io.imread(render_path)


    mask = original[:,:,3] > 0
    render_filtered = render * mask[:,:,None]


    print(original[:,:,:3].shape)
    print(render_filtered.shape)    


    # Calculate PSNR
    psnr_value = psnr(original[:,:,:3], render_filtered)

    # Calculate SSIM
    ssim_value, _ = ssim(original[:,:,:3], render_filtered, full=True, channel_axis=2)


    # Calculate LPIPS
    lpips_model = lpips.LPIPS(net='alex')
    original_tensor = torch.tensor(original[:,:,:3], dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0
    render_tensor = torch.tensor(render_filtered, dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0
    lpips_value = lpips_model(original_tensor, render_tensor).item()

    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")
    print(f"LPIPS: {lpips_value}")