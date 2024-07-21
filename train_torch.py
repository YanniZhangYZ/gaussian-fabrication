import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import fov2focal, getWorld2View, getWorld2View2
from ink_intrinsics import Ink
from gsplat import project_gaussians
from gsplat import rasterize_gaussians
import numpy as np
from PIL import Image   
import torch.nn.functional as F
import torchvision
from utils.loss_utils import l1_loss, ssim
from train import prepare_output_and_logger
import matplotlib.pyplot as plt
from scene.cameras import Camera
from blob_factor_model import ExtinctionModel

# from mitsuba_conversion import debug_plot_g


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# Load ink intrinsic data
INK = Ink()


def debug_plot(pos, debug_color, debug_alpha, path):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Voxel Data')
    print("Ploting voxel representation")
    rgba = np.concatenate((debug_color.reshape(-1, 3), debug_alpha.reshape(-1, 1)), axis=1)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=rgba, s=0.4)
    plt.savefig(path)


def ink_to_RGB(mix, effective_trans, H, W):
    '''
    mix: (H*W,6) array of ink mixtures
    output: (3,H,W) array of RGB values
    
    '''
    # mix: (H*W,6) array of ink mixtures
    # C, H, W = mix.shape
    # mix = mix.permute(1,2,0).view(-1,C)

    # no transparent and white ink
    # mix = mix[:,:4]
    mix =  mix[:,:5]
    # no white ink
    # assert mix[:,4].sum() == 0.0, "The white ink should be 0.0"
    # mix = torch.cat([mix[:,:4], mix[:,5:]], dim=1)

    # assert mix.shape[1] == 5, "The ink mix should have 5 channels"


    if (mix < 0.0).any():
        mask = torch.nonzero(mix < 0.0)
        print(mask)
        print(mix[mask[0]])
        # print(temp[mask[0]])
        assert False, "Negative ink concentration inside ink_to_RGB"
    
    # mix: (H*W,6) array of ink mixtures
    # K
    mix_K = mix @ INK.absorption_matrix[:5]
    # mix_K = mix @ INK.absorption_matrix
    # mix_K = mix @ torch.cat((INK.absorption_matrix[:4], INK.absorption_matrix[5:]), dim=0)
    # S
    mix_S = mix @ INK.scattering_matrix[:5] + 1e-8
    # mix_S = mix @ INK.scattering_matrix + 1e-8
    # mix_S = mix @ torch.cat((INK.scattering_matrix[:4], INK.scattering_matrix[5:]), dim=0) + 1e-8


    #equation 2
    # NOTE: Here we add 1e-8 just to make sure it is differentiable (sqrt(0) has no grad)
    R_mix = 1 + mix_K / mix_S - torch.sqrt((mix_K / mix_S)**2 + 2 * mix_K / mix_S + 1e-8)

    if torch.isnan(torch.sqrt( (mix_K / mix_S)**2 + 2 * mix_K / mix_S)).any():
        temp = (mix_K / mix_S)**2 + 2 * mix_K / mix_S
        mask = torch.nonzero(torch.isnan(torch.sqrt(temp)))
        # print(R_mix.shape)
        print(mask)
        print(temp[mask[0]])
        assert False, "sqrt negative value has nan"
    
    # # manual linear interpolation, for each element in the row vector of R_mix, we compute the mean between the two values and insert it in between
    # R_mix = torch.cat([R_mix, (R_mix[:,1:] + R_mix[:,:-1]) / 2], dim=1)
    R_mix = torch.cat([R_mix, (R_mix[:,1:] + R_mix[:,:-1]) / 2], dim=1)
    R_mix = torch.cat([R_mix, torch.zeros((R_mix.shape[0], INK.w_num - R_mix.shape[1]), dtype=torch.float, device= 'cuda')], dim=1)
    assert (R_mix[:,71] == 0.0).all(), "The 71th element should be 0.0"


    with torch.no_grad():
        # equation 3 - 5
        x_D56 = INK.x_observer * INK.sampled_illuminant_D65 * INK.w_delta
        # x_D56 /= np.sum(x_D56)
        y_D56 = INK.y_observer * INK.sampled_illuminant_D65 * INK.w_delta
        # y_D56 /= np.sum(y_D56)
        z_D56 = INK.z_observer * INK.sampled_illuminant_D65 * INK.w_delta
        # z_D56 /= np.sum(z_D56)

        D56_light_intensity = torch.tensor([x_D56.sum(), y_D56.sum(), z_D56.sum()], device="cuda") / INK.w_num
    

    X = R_mix @ x_D56
    Y = R_mix @ y_D56
    Z = R_mix @ z_D56

    X = X / INK.w_num
    Y = Y / INK.w_num
    Z = Z / INK.w_num

    XYZ = torch.stack([X,Y,Z],axis=1).T

    # Convert XYZ to sRGB, Equation 7
    with torch.no_grad():
        sRGB_matrix = torch.tensor([[3.2406, -1.5372, -0.4986],
                                [-0.9689, 1.8758, 0.0415],
                                [0.0557, -0.2040, 1.0570]], device="cuda")
    sRGB = (sRGB_matrix @ XYZ).T

    # Apply gamma correction to convert linear RGB to sRGB
    mask = sRGB > 0.0031308
    sRGB[~mask] = sRGB[~mask] * 12.92
    sRGB[mask] = 1.055 * torch.pow(sRGB[mask], 1 / 2.4) - 0.055
    # sRGB = torch.where(sRGB <= 0.0031308,
    #                 12.92 * sRGB,
    #                 1.055 * torch.pow(sRGB, 1 / 2.4) - 0.055)


    sRGB = sRGB.view(H, W, 3).permute(2,0,1) + effective_trans * D56_light_intensity.view(3,1,1)
    sRGB = torch.clip(sRGB,0,1)

    # sRGB = torch.clip(sRGB,0,1).view(H, W, 3).permute(2,0,1)

    if torch.isnan(sRGB).any():
        temp = sRGB.clone().detach()
        mask = torch.nonzero(torch.isnan(temp))
        # print(R_mix.shape)
        print(mask)
        print(temp[mask[0]])
        assert False, "sRGB has nan"


    return sRGB


def ink_to_spectral_albedo(mix, H, W):
    '''
    mix: (H*W,6) array of ink mixtures
    output: (3,H,W) array of RGB values
    
    '''
    # mix: (H*W,6) array of ink mixtures
    # C, H, W = mix.shape
    # mix = mix.permute(1,2,0).view(-1,C)

    # no transparent and white ink
    # mix = mix[:,:4]
    mix =  mix[:,:5]
    # no white ink
    # assert mix[:,4].sum() == 0.0, "The white ink should be 0.0"
    # mix = torch.cat([mix[:,:4], mix[:,5:]], dim=1)

    # assert mix.shape[1] == 5, "The ink mix should have 5 channels"


    if (mix < 0.0).any():
        mask = torch.nonzero(mix < 0.0)
        print(mask)
        print(mix[mask[0]])
        # print(temp[mask[0]])
        assert False, "Negative ink concentration inside ink_to_RGB"
    
    # mix: (H*W,6) array of ink mixtures
    # K
    mix_K = mix @ INK.absorption_matrix[:5]
    # mix_K = mix @ INK.absorption_matrix
    # mix_K = mix @ torch.cat((INK.absorption_matrix[:4], INK.absorption_matrix[5:]), dim=0)
    # S
    mix_S = mix @ INK.scattering_matrix[:5] + 1e-8
    # mix_S = mix @ INK.scattering_matrix + 1e-8
    # mix_S = mix @ torch.cat((INK.scattering_matrix[:4], INK.scattering_matrix[5:]), dim=0) + 1e-8


    #equation 2
    # NOTE: Here we add 1e-8 just to make sure it is differentiable (sqrt(0) has no grad)
    R_mix = mix_S / (mix_S + mix_K)

    # # manual linear interpolation, for each element in the row vector of R_mix, we compute the mean between the two values and insert it in between
    # R_mix = torch.cat([R_mix, (R_mix[:,1:] + R_mix[:,:-1]) / 2], dim=1)
    R_mix = torch.cat([R_mix, (R_mix[:,1:] + R_mix[:,:-1]) / 2], dim=1)
    R_mix = torch.cat([R_mix, torch.zeros((R_mix.shape[0], INK.w_num - R_mix.shape[1]), dtype=torch.float, device= 'cuda')], dim=1)
    assert (R_mix[:,71] == 0.0).all(), "The 71th element should be 0.0"


    with torch.no_grad():
        # equation 3 - 5
        x_D56 = INK.x_observer * INK.sampled_illuminant_D65 * INK.w_delta
        # x_D56 /= np.sum(x_D56)
        y_D56 = INK.y_observer * INK.sampled_illuminant_D65 * INK.w_delta
        # y_D56 /= np.sum(y_D56)
        z_D56 = INK.z_observer * INK.sampled_illuminant_D65 * INK.w_delta
        # z_D56 /= np.sum(z_D56)
        D56_light_intensity = torch.tensor([x_D56.sum(), y_D56.sum(), z_D56.sum()], device="cuda") / INK.w_num

    X = R_mix @ x_D56
    Y = R_mix @ y_D56
    Z = R_mix @ z_D56

    X = X / INK.w_num
    Y = Y / INK.w_num
    Z = Z / INK.w_num

    XYZ = torch.stack([X,Y,Z],axis=1).T

    # Convert XYZ to sRGB, Equation 7
    with torch.no_grad():
        sRGB_matrix = torch.tensor([[3.2406, -1.5372, -0.4986],
                                [-0.9689, 1.8758, 0.0415],
                                [0.0557, -0.2040, 1.0570]], device="cuda")
    sRGB = (sRGB_matrix @ XYZ).T

    # Apply gamma correction to convert linear RGB to sRGB
    mask = sRGB > 0.0031308
    sRGB[~mask] = sRGB[~mask] * 12.92
    sRGB[mask] = 1.055 * torch.pow(sRGB[mask], 1 / 2.4) - 0.055
    # sRGB = torch.where(sRGB <= 0.0031308,
    #                 12.92 * sRGB,
    #                 1.055 * torch.pow(sRGB, 1 / 2.4) - 0.055)

    # sRGB = torch.clip(sRGB,0,1).view(H, W, 3).permute(2,0,1)
    sRGB = sRGB.view(H, W, 3).permute(2,0,1)
    sRGB = torch.clip(sRGB,0,1)

    if torch.isnan(sRGB).any():
        temp = sRGB.clone().detach()
        mask = torch.nonzero(torch.isnan(temp))
        # print(R_mix.shape)
        print(mask)
        print(temp[mask[0]])
        assert False, "sRGB has nan"


    return sRGB





def ink_to_albedo(mix, H, W):
    '''
    mix: (H*W,6) array of ink mixtures
    output: (3,H,W) array of RGB values
    
    '''
    # mix: (H*W,6) array of ink mixtures
    # C, H, W = mix.shape
    # mix = mix.permute(1,2,0).view(-1,C)

    # no transparent and white ink
    # mix = mix[:,:4]
    mix =  mix[:,:5]


    if (mix < 0.0).any():
        mask = torch.nonzero(mix < 0.0)
        print(mask)
        print(mix[mask[0]])
        # print(temp[mask[0]])
        assert False, "Negative ink concentration inside ink_to_RGB"
    
    # mix: (H*W,6) array of ink mixtures
    # K
    mix_K = mix @ INK.absorption_RGB[:5]
    # S
    mix_S = mix @ INK.scattering_RGB[:5]

    if torch.isnan( mix_S / (mix_K + mix_S + 1e-8)).any():
        temp =  mix_S / (mix_K + mix_S + 1e-8)
        mask = torch.nonzero(torch.isnan(temp))
        # print(R_mix.shape)
        print(mask)
        print(temp[mask[0]])
        assert False, "albedo has nan"

    albedo =  mix_S / (mix_K + mix_S + 1e-8)
    sRGB = torch.clip(albedo,0,1).view(H, W, 3).permute(2,0,1)
    return sRGB




from kornia.color import rgb_to_lab
import math
from torchmetrics.image import StructuralSimilarityIndexMeasure

def loss_ink_mix(mix, surface_mix, out_alpha, viewpoint_cam, gt_images_folder_path):
    
    C, H, W = mix.shape
    # mix = F.relu(mix)
    # print(mix.shape)

    #TODO: deal with error in the future
    if (mix.sum(dim=0) > 1.0 + 1e-1).any():
        print(mix.sum(dim=0).max()) 
        raise RuntimeError("TODO: normalize the ink")


    # NOTE: The training image has transparent background!
    # NOTE: Therefore we add white ink only to the place where there is a blob!
    # mix[4,:,:] = mix[4,:,:] + transparent_mask  # This is used in the original ink_to_rgb.py file
    # mix[4,:,:] = mix[4,:,:] + transparent_blob_mask

    C, H, W = mix.shape
    mix_alpha =  mix.sum(dim=0)

    # mix = mix.permute(1,2,0).view(-1,C)

    # current_render = ink_to_RGB(mix.permute(1,2,0).view(-1,C), (1.0 - sum_Ta_RGB).permute(2,0,1),  H, W)
    # surface_render = ink_to_RGB(mix.permute(1,2,0).view(-1,C), H, W)
    # current_render_ = ink_to_albedo(mix.permute(1,2,0).view(-1,C), H, W)
    # out_img = effective_transmission * effective_albedo
    # out_img = effective_transmission
    # surface_render = ink_to_RGB(mix.permute(1,2,0).view(-1,C), H, W)

    # current_render = ink_to_spectral_albedo(mix.permute(1,2,0).view(-1,C), mul_transmittance.permute(2,0,1),  H, W)

    # albedo_spectral =  ink_to_spectral_albedo(mix.permute(1,2,0).view(-1,C), H, W)
    # current_render = ink_to_albedo(mix.permute(1,2,0).view(-1,C), H, W)
    # surface_render = ink_to_albedo(surface_mix.permute(1,2,0).view(-1,C), H, W)
    current_render =  ink_to_spectral_albedo(mix.permute(1,2,0).view(-1,C), H, W)
    surface_render = ink_to_spectral_albedo(surface_mix.permute(1,2,0).view(-1,C), H, W)
    # surface_render = ink_to_RGB(surface_mix.permute(1,2,0).view(-1,C), H, W)


    # NOTE: we need to take special care to the backgroud color
    # For the rasterized result, if there is no ink mixture, e.g., np.zeros(6), the resulting RGB is near white (1,1,1)
    # However, the GT image has a transparent background, and depends on whether we use '-w' or not, the background is filled with white or black
    # For the loss computation, we care two things:
    # 1. For the palces where GT has RGB color, we want the render to be close to that color
    # 2. For the places where GT has no color (alpha is 0), we want the render's out alpha to be close to 0, and other places to be close to 1

    # NOTE: Therefore
    # 1. We compute the loss between the render * GT alpha and GT, so that image only where GT has RGB color
    # 2. We compute the loss between the render alpha and GT alpha, so that the render's alpha is close to GT alpha, especially when GT has no color


    # TODO: Sometimes the viewpoint_cam.image_name is larger than 99. Need to fix this
    gt_image = viewpoint_cam.original_image.cuda() # This is the GT with post processed bkg depends on -w flag
    gt_original_path = os.path.join(gt_images_folder_path, viewpoint_cam.image_name+".png")
    gt_orginal = Image.open(gt_original_path)
    if gt_orginal.size != (H, W): # in case we are not training on the original size
        gt_orginal = gt_orginal.resize((H, W))
    im_data = np.array(gt_orginal.convert("RGBA"))/ 255.0

    gt_original_rgba = torch.tensor(im_data, dtype=torch.float32, device="cuda").view(H, W, 4).permute(2, 0, 1)
    out_alpha = out_alpha.view(H, W, 1).permute(2, 0, 1)


    # render * GT alpha
    render_filtered = current_render * gt_original_rgba[3:4, :, :]
    assert render_filtered.shape == current_render.shape, (render_filtered.shape, current_render.shape)
    render_filtered_rgba = torch.cat([render_filtered, out_alpha], dim=0)
    # gt_rgba = torch.cat([gt_image, gt_alpha], dim=0)

    # 1. l1 between render and GT, RGB loss
    # Ll1 = l1_loss(current_render, gt_image)

    # 2. l1 between render * GT alpha and GT, RGBA loss
    # assert render_filtered_rgba.shape == gt_original_rgba.shape, (render_filtered_rgba.shape, gt_original_rgba.shape)
    # Ll1 = l1_loss(render_filtered_rgba, gt_original_rgba)

    # 3. l1 between render and GT, RGBA loss

    mix_alpha = mix_alpha.view(H, W, 1).permute(2, 0, 1)
    # for the places where mix_alpha is 0, we want the corresponding render pixel to be white (1,1,1)
    mask = mix_alpha == 0.0
    mask = mask.expand_as(current_render)

    current_render[mask] = 1.0
    surface_render[mask] = 1.0
    # albedo_rgb[mask] = 1.0
    # albedo_spectral[mask] = 1.0


    mask_gt = gt_original_rgba[3:4, :, :] == 0.0
    alpha_redundent = out_alpha * mask_gt

    mask_gt = mask_gt.expand_as(mix)
    mix_redundent = mix * mask_gt
    mix_zeros =  torch.zeros_like(mix, device="cuda")
    alpha_zeros =  torch.zeros_like(out_alpha, device="cuda")

    # current_render_rgba = torch.cat([current_render, mix_alpha], dim=0)
    # current_render_rgba = torch.cat([current_render,  surface_render, out_alpha, mix_redundent], dim=0)

    # gt_image_rgba = torch.cat([gt_image, gt_image, gt_original_rgba[3:4,:,:],mix_zeros], dim=0)

    current_render_rgba = torch.cat([current_render,surface_render, out_alpha, alpha_redundent, mix_redundent], dim=0)

    gt_image_rgba = torch.cat([gt_image, gt_image, gt_original_rgba[3:4, :, :], alpha_zeros, mix_zeros], dim=0)
    assert current_render_rgba.shape == gt_image_rgba.shape, (current_render_rgba.shape, gt_image_rgba.shape)
    Ll1 = l1_loss(current_render_rgba, gt_image_rgba)


    def total_variation_loss(image):
        tv_h = torch.pow(image[:, :, 1:, :] - image[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(image[:, :, :, 1:] - image[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / image.numel()
    
    TV_loss = total_variation_loss(current_render[None,:])

    # alpha_constraint = -torch.log(out_alpha + 1e-8).mean()

    # albedo_loss = l1_loss(albedo_rgb, albedo_spectral)

    
    # original_GT= Image.open('lego/train/r_25.png')
    # origin_GT_RGBA = np.array(original_GT.convert("RGBA"))/ 255.0
    # alpha = torch.tensor(origin_GT_RGBA[:, :, 3:4], dtype=torch.float32, device="cuda")
    # # print("I got GT alpha!", out_alpha.shape, alpha.shape)
    # out_alpha = out_alpha.view(H, W, 1).permute(2, 0, 1)
    # alpha = alpha.view(H, W, 1).permute(2, 0, 1)
    # assert out_alpha.shape == alpha.shape, (out_alpha.shape, alpha.shape)

    # # Here we are compute loss between RGBA not just RGB
    # render_rgba = torch.cat([current_render, out_alpha], dim=0)
    # gt_rgba = torch.cat([gt_image, alpha], dim=0)
    # Ll1 = l1_loss(render_rgba, gt_rgba)
    # Ll1 = l1_loss(current_render, gt_image)

    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(current_render, gt_image))
    # loss = (1.0 - 0.25) * Ll1 + 0.2 * (1.0 - ssim(current_render, gt_image)) + 0.05 * alpha_constraint

    # lab_GT = rgb_to_lab(gt_rgb.reshape(1, 3, H , W)).reshape(H, W,3)
    # lab_image = rgb_to_lab(current_render.reshape(1, 3, H,W)).reshape(H,W,3)
    # delta_e76 = torch.sqrt(torch.sum((lab_image - lab_GT)**2, dim=(0, 1))/(H*W))
    # loss_delta_e76 = torch.mean(delta_e76)

    # ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    # loss_ssim = ssim(current_render.reshape(1,3, H, W), gt_rgb.reshape(1,3, H, W))
    # return 0.2* loss_ssim + 0.7 * loss_mse + 0.1 * loss_delta_e76, current_render


    return loss, current_render, surface_render

     


def training(dataset : ModelParams, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, save_imgs=False):
    # # enable anomaly detection
    # torch.autograd.set_detect_anomaly(True)

    tb_writer = prepare_output_and_logger(dataset)
    lr = 0.01
    gaussians = GaussianModel(0)
    scene = Scene(dataset, gaussians, shuffle = False) # TODO: should change to True during training
    # gaussians.training_setup(opt)

    print("!!!!!!!Loaded scene with {} gaussians".format(gaussians.get_xyz.shape[0]))


    bg_color = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # CMYKWT the backgroud is fully black ink
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


     # set up rasterization configuration
    # views = scene.getTrainCameras()
    # view_idx = 2
    # viewpoint_camera = views[view_idx]
    # image_height=int(viewpoint_camera.image_height)
    # image_width=int(viewpoint_camera.image_width)
    # fy = fov2focal(viewpoint_camera.FoVy, image_height)
    # fx = fov2focal(viewpoint_camera.FoVx, image_width)

    # gaussians._xyz.requires_grad = False
    gaussians._opacity.requires_grad = False
    # gaussians._scaling.requires_grad = False
    # gaussians._rotation.requires_grad = False

    l = [
        {'params': [gaussians._xyz], 'lr': 0.00016, "name": "xyz"},
        {'params': [gaussians._ink_mix], 'lr': lr, "name": "ink"},
        # {'params': [gaussians._opacity], 'lr': lr, "name": "opacity"},
        {'params': [gaussians._scaling], 'lr': lr, "name": "scaling"},
        {'params': [gaussians._rotation], 'lr': lr, "name": "rotation"}
    ]

    optimizer = torch.optim.Adam(l, lr=0.01, eps=1e-15) 

    # l = [
    #     {'params': [gaussians._xyz], 'lr': 0.00016, "name": "xyz"},
    #     {'params': [gaussians._ink_mix], 'lr': 0.0025, "name": "ink"},
    #     {'params': [gaussians._opacity], 'lr': 0.05, "name": "opacity"},
    #     {'params': [gaussians._scaling], 'lr': 0.005, "name": "scaling"},
    #     {'params': [gaussians._rotation], 'lr': 0.001, "name": "rotation"}
    # ]

    # optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15) 
    frames = []
    B_SIZE = 16

    #  TODO
    
    viewpoint_stack = None
    gt_images_folder_path = os.path.join(dataset.source_path, "train")
    # torch.autograd.set_detect_anomaly(True)


    blob_factor_model = ExtinctionModel(4, 64, 2).to('cuda')
    blob_factor_model.load_state_dict(torch.load('blob_factor_albedo_spec_new/best_model.pth'))
    blob_factor_model.eval()




    for i in range(opt.iterations):
        #TODO
        if not (gaussians.get_rotation.norm(dim=-1) - 1 < 1e-6).all():
            nan_mask = torch.isnan(gaussians.get_rotation.norm(dim=-1)-1)
            nan_idex = torch.nonzero(nan_mask)
            print("nan_idx: ", nan_idex)
            print(gaussians.get_rotation[nan_idex])
            print(gaussians._rotation[nan_idex])
            # # print(quats.shape)
            # print(nan_idex[0], nan_idex.shape)
            # print(nan_idex)
            # print(gaussians.get_rotation[nan_idex[0]])
            # print(gaussians._rotation[nan_idex[0]])
            # return 0
            # print("Here i have nan")
            assert False, "Here I have nan"


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        
        image_height=int(viewpoint_camera.image_height)
        image_width=int(viewpoint_camera.image_width)
        fy = fov2focal(viewpoint_camera.FoVy, image_height)
        fx = fov2focal(viewpoint_camera.FoVx, image_width)

        (
            xys,
            depths,
            radii,
            conics_,
            cov1d,
            compensation,
            num_tiles_hit,
            cov3d,
        ) = project_gaussians(
            means3d = gaussians.get_xyz,
            scales = gaussians.get_scaling,
            glob_scale = 1,
            quats = gaussians.get_rotation,
            viewmat = viewpoint_camera.world_view_transform.T,
            fx = fx,
            fy = fy,
            cx = image_height/2,
            cy = image_width/2,
            img_height = image_height,
            img_width = image_width,
            block_width = B_SIZE,
        )
        # preprocess given ink mixtures given the transmittance
        # K
        mix_absorption_RGB = gaussians.get_ink_mix[:,:5] @ INK.absorption_RGB[:5]
        # S
        mix_scattering_RGB = gaussians.get_ink_mix[:,:5] @ INK.scattering_RGB[:5]

        mix_extinction_RGB = mix_absorption_RGB + mix_scattering_RGB


        if i < 1500:
            transmittance = torch.exp(-mix_extinction_RGB * torch.sqrt(cov1d -  5e-3 + 1e-8)[:,None] * 6).mean(dim=1)
            conics = conics_.clone()

        else:
            blob_model_input = torch.cat([torch.sqrt(cov1d -  5e-3 + 1e-8)[:,None] / 0.05 , mix_extinction_RGB / 255.0], dim=1)
            blob_factors = blob_factor_model(blob_model_input) * torch.tensor([20.0, 60.0], dtype=torch.float32, device='cuda')
            transmittance = torch.exp(-mix_extinction_RGB * torch.sqrt(cov1d -  5e-3 + 1e-8)[:,None] * blob_factors[:,1][:,None] *6).mean(dim=1)

            conics = conics_.clone()
            conics = conics.to(conics_.device)

            conics_x = conics[:, 0] * blob_factors[:,0]
            conics_y = conics[:, 2] * blob_factors[:,0]
            conics_xy = conics[:, 1] * blob_factors[:,0]

            conics = torch.stack([conics_x, conics_xy, conics_y], dim=1)
        
        # transmittance = torch.exp(-mix_extinction_RGB * torch.sqrt(cov1d -  5e-3 + 1e-8)[:,None] * 6)
        
        factor =  1.0 - transmittance

        # NOTE: in the most ideal case, for the place that GT has no color, the rasterize result alpha should be 0
        ink_mix, out_alpha = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                gaussians.get_ink_mix,
                factor[:,None],
                # torch.ones_like(factor[:,None]),
                image_height,
                image_width,
                B_SIZE,
                background,
                return_alpha=True
            )
            
        # Here it is the opaque mix of inks
        final_ink_mix = ink_mix.permute(2, 0, 1)


        surface_mix = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                gaussians.get_ink_mix,
                torch.ones_like(factor[:,None]),
                image_height,
                image_width,
                B_SIZE,
                background,
            )
            
        # Here it is the opaque mix of inks
        final_surface_mix = surface_mix.permute(2, 0, 1)


        # ink_mix_R, alpha_R = rasterize_gaussians(
        #         xys,
        #         depths,
        #         radii,
        #         conics,
        #         num_tiles_hit,
        #         torch.ones_like(gaussians.get_ink_mix),
        #         factor[:,0][:,None],
        #         image_height,
        #         image_width,
        #         B_SIZE,
        #         background,
        #         return_alpha=True
        # )

        # ink_mix_G, alpha_G = rasterize_gaussians(
        #         xys,
        #         depths,
        #         radii,
        #         conics,
        #         num_tiles_hit,
        #         torch.ones_like(gaussians.get_ink_mix),
        #         factor[:,1][:,None],
        #         image_height,
        #         image_width,
        #         B_SIZE,
        #         background,return_alpha=True
        # )

        # ink_mix_B, alpha_B = rasterize_gaussians(
        #         xys,
        #         depths,
        #         radii,
        #         conics,
        #         num_tiles_hit,
        #         torch.ones_like(gaussians.get_ink_mix),
        #         factor[:,2][:,None],
        #         image_height,
        #         image_width,
        #         B_SIZE,
        #         background,
        #         return_alpha=True
        # )


        # sum_Ta_RGB = torch.stack((ink_mix_R[:,:,0], ink_mix_G[:,:,0], ink_mix_B[:,:,0]), dim=2)
        # mul_transmittance =  1.0 - torch.stack((alpha_R,alpha_G, alpha_B), dim=2)

        torch.cuda.synchronize()
        loss1, out_img, surface_render = loss_ink_mix(final_ink_mix, final_surface_mix, out_alpha, viewpoint_camera, gt_images_folder_path)



        
        # mean_scales = torch.mean(gaussians.get_scaling, dim=0)
        # std_scales = torch.std(gaussians.get_scaling, dim=0)
        # threshold1 = mean_scales + std_scales
        # # # threshold1 = mean_scales

        # # # threshold2 = mean_scales - 0.5 * std_scales

        # # # # assert False, (std_scales, threshold1, threshold2)
        # # # # ReLU() is used to implement a soft thresholding mechanism where any scale value below the threshold results in zero penalty. 
        # # # # The penalty only kicks in for values above the threshold and increases linearly beyond that point.
        # import torch.nn as nn
        # relu = nn.ReLU()
        # scale_loss = relu(gaussians.get_scaling - threshold1).sum()
        # # # scale_loss += relu(threshold2 - gaussians.get_scaling).sum()
        # # # scale_loss = relu(mean_scales - gaussians.get_scaling).sum()
        # # # # scale_loss = relu(gaussians.get_scaling - mean_scales).sum()
        # # # # scale_loss += relu(mean_scales - gaussians.get_scaling).sum()


        if i% 500 == 0:
            scale_, idx_ = gaussians.get_scaling.max(dim=0)
            print("=============================================================================")
            print(" the largest blob is: ", scale_, idx_)
            for i_ in idx_:
                print("----------------------------------")
                print("mix: ", gaussians.get_ink_mix[i_])
                print("opacity: ", factor[i_])
            print("=============================================================================")
            torchvision.utils.save_image(out_img, "{}_render.png".format(i))
            torchvision.utils.save_image(surface_render, "{}_surf.png".format(i))





        # # mean_scale =  gaussians.get_scaling.mean()
        # # scale_variance = ((gaussians.get_scaling[:,0] - mean_scale).pow(2) + (gaussians.get_scaling[:,1] - mean_scale).pow(2) + (gaussians.get_scaling[:,2] - mean_scale).pow(2)) / 3.0
        # if i < 1500:
        #     loss =  loss1
        # else:
        #     loss =  loss1 + 0.001 * scale_loss

        import torch.nn as nn
        relu = nn.ReLU()
        z_loss= relu(torch.sqrt(cov1d -  5e-3 + 1e-8) - 0.01).sum() + relu(0.0025 - torch.sqrt(cov1d -  5e-3 + 1e-8)).sum()
        # z_loss= relu(torch.sqrt(cov1d -  5e-3 + 1e-8) - 0.01).sum()




        # if i  < 1500:
        #     loss =  loss1
        # else:
        #     loss =  loss1 + 0.001 * z_loss

        loss =  loss1 + 0.01 * z_loss




        # max_ext,_ = torch.max(mix_extinction_RGB, dim=1)
        # loss =  loss1 + 1e-4 * (max_ext * cov1d* 6).mean()

        # #  add regularization term to avoid super sharp blob
        # mean_scale =  gaussians.get_scaling.mean(dim = 1)
        # # scale_variance = ((gaussians.get_scaling[:,0] - mean_scale).pow(2) + (gaussians.get_scaling[:,1] - mean_scale).pow(2) + (gaussians.get_scaling[:,2] - mean_scale).pow(2)) / 3.0
        # scale_variance = ((gaussians.get_scaling[:,0] - mean_scale).abs() + (gaussians.get_scaling[:,1] - mean_scale).abs() + (gaussians.get_scaling[:,2] - mean_scale).abs()) / 3.0
        
        # loss  = loss1 +  0.001 * scale_variance.sum()
        # # loss += 0.1 * scale_variance.mean()


        # print(scale_variance.sum())

        # # add regularization term to avoid large dense blob
        # scale_norm = gaussians.get_scaling.norm(dim=1)
        # norm_times_exinction = scale_norm * mix_extinction_RGB.mean(dim=1)
        # loss = loss1 + 0.1 * norm_times_exinction.mean()

        
        # cov1d.register_hook(print)

        # if loss is nan
        #TODO
        if torch.isnan(loss):
            mask = torch.nonzero(torch.isnan(out_img))
            print("loss is nan, output image nan: ", mask)
            c, h, w = mask[0]
            bug_ink = final_ink_mix[:, h, w]
            print("before ", bug_ink)

            bug_ink = F.relu(bug_ink)
            epsilon = 1e-8
            bug_ink = bug_ink / (bug_ink.sum() + epsilon)
            print("after ", bug_ink)

        optimizer.zero_grad()
        loss.backward()
        
        # Check if there is nan in the gradient
        if torch.isnan(gaussians._rotation.grad).any():
            mask = torch.nonzero(torch.isnan(gaussians._rotation.grad))
            print("rotation grad has nan: ", mask)
            print(gaussians._rotation.grad[mask[0]])
            assert False, "rotation grad has nan"

        torch.cuda.synchronize()
        optimizer.step()
        # print(f"Iteration {i + 1}/{opt.iterations}, Loss: {loss.item()}")
        print(f"Iteration {i + 1}/{opt.iterations}, Loss: {loss.item()}, loss1: {loss1.item()}, z_loss:{0.001 * z_loss}")
        # print(f"Iteration {i + 1}/{opt.iterations}, Loss: {loss.item()}, loss1: {loss1.item()},ne_mean:{0.1 * norm_times_exinction.mean().item()}")
        # print(f"Iteration {i + 1}/{opt.iterations}, Loss: {loss.item()}, loss1: {loss1.item()},abs_sum:{ 0.001 * scale_variance.sum().item()}")

        # print(f"Iteration {i + 1}/{opt.iterations}, Loss: {loss.item()}, loss1: {loss1.item()}, z_ext:{1e-4 * (max_ext * cov1d* 6).mean().item()}")




        if save_imgs and i % 5 == 0:
            frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))


        if i == opt.iterations - 1:
            imgs_path = os.path.join(dataset.model_path, "result_imgs")
            if not os.path.exists(imgs_path):
                os.makedirs(imgs_path)
            torchvision.utils.save_image(out_img, os.path.join(imgs_path, "3000_optimize.png"))
            # torchvision.utils.save_image(surface_render, os.path.join(imgs_path, "3000_surface.png"))


            print("\n[ITER {}] Saving Gaussians".format(i))
            # NOTE: to compare with reading from ply
            print("{}'s ink mix: {}".format(0, gaussians.get_ink_mix[0]))
            print("{}'s ink mix: {}".format(5000, gaussians.get_ink_mix[5000]))



            #  plot scales xyz in a histogram
            plt.figure()
            scale_xyz = gaussians.get_scaling.detach().cpu().numpy()
            plt.hist(scale_xyz[:,0], bins=100)
            plt.hist(scale_xyz[:,1], bins=100)
            plt.hist(scale_xyz[:,2], bins=100)
            plt.savefig(os.path.join(imgs_path, f"histogram_scales.png"))

            pos = gaussians.get_xyz.detach().cpu().numpy()
            debug_color = gaussians.convert_center_ink_2_rgb_4_debug()
            assert debug_color.shape == pos.shape, (debug_color.shape, pos.shape)
            # debug_alpha = gaussians.get_opacity.detach().cpu().numpy()

            # with torch.no_grad():
            #     blob_model_input = torch.cat([gaussians.get_scaling.detach().mean(axis = 1)[:,None] / 0.05 , mix_extinction_RGB/ 255.0], dim=1)
            #     blob_factors = blob_factor_model(blob_model_input)* torch.tensor([2.0, 200.0], dtype=torch.float32, device='cuda')
            # transmittance_ = np.exp(-mix_extinction_RGB.detach().cpu().numpy() * scale_xyz.mean(axis = 1)[:,None]* blob_factors[:,1][:,None].detach().cpu().numpy() * 6).mean(axis=1)
            transmittance_ = np.exp(-mix_extinction_RGB.detach().cpu().numpy() * scale_xyz.mean(axis = 1)[:,None]* 6).mean(axis=1)
            debug_alpha = 1.0 - transmittance_
            # plot factor in a histogram
            plt.figure()
            plt.hist(debug_alpha, bins=100)
            plt.savefig(os.path.join(imgs_path, f"histogram_opacity_factor.png"))

            alpha_ones = np.ones(pos.shape[0])
            debug_plot(pos, debug_color, debug_alpha, os.path.join(imgs_path,"gaussian_init_data_alpha.png"))
            debug_plot(pos, debug_color, alpha_ones, os.path.join(imgs_path,"gaussian_init_data.png"))

            debug_alpha = np.squeeze(out_alpha.detach().cpu().numpy())
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.imshow(debug_alpha, cmap='viridis') 
            fig.colorbar(cax)
            fig.savefig(os.path.join(imgs_path,'alpha_channel_image.png'))
            scene.save(i+1)

            #  check if the ink mix contains no white ink and transparent ink
            debug_ink = gaussians.get_ink_mix.detach().cpu().numpy()
            # assert (debug_ink[:, 4] == 0.0).all(), "There should not be white ink in the ink mix"
            assert (debug_ink[:, 5] == 0.0).all(), "There should not transparent ink in the ink mix"



            # preprocess given ink mixtures given the transmittance
            # plot factor in a histogram
            mix_absorption_RGB = gaussians.get_ink_mix[:,:5] @ INK.absorption_RGB[:5]
            # S
            mix_scattering_RGB = gaussians.get_ink_mix[:,:5] @ INK.scattering_RGB[:5]

            mix_extinction_RGB = mix_absorption_RGB + mix_scattering_RGB
            
            # with torch.no_grad():
            #     blob_model_input = torch.cat([torch.sqrt(cov1d -  5e-3 + 1e-8)[:,None] / 0.05 , mix_extinction_RGB/255.0], dim=1)
            #     blob_factors = blob_factor_model(blob_model_input) * torch.tensor([2.0, 200.0], dtype=torch.float32, device='cuda')
            # transmittance = torch.exp(-mix_extinction_RGB * torch.sqrt(cov1d -  5e-3 + 1e-8)[:,None] * blob_factors[:,1][:,None] * 6).mean(dim=1)
            transmittance = torch.exp(-mix_extinction_RGB * torch.sqrt(cov1d - 5e-3 + 1e-8)[:,None] * 6).mean(dim=1)
            factor =  1.0 - transmittance

            # plt.figure()
            # plt.hist(blob_factors[:,1][:,None].detach().cpu().numpy(), bins=100)
            # plt.savefig(os.path.join(imgs_path,'blob_factor.png'))


            depth_map = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                depths[:, None].repeat(1, 6),
                factor[:, None],
                image_height,
                image_width,
                B_SIZE,
                background,
            )
            depth_map = depth_map.permute(2, 0, 1)[0:1, :, :]

            debug_depth_map = np.squeeze(depth_map.detach().cpu().numpy())
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.imshow(debug_depth_map, cmap='viridis') 
            fig.colorbar(cax)
            fig.savefig(os.path.join(imgs_path,'depth_map.png'))


            ink_sum = np.squeeze(ink_mix.sum(dim=2).detach().cpu().numpy())
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.imshow(ink_sum, cmap='viridis') 
            fig.colorbar(cax)
            fig.savefig(os.path.join(imgs_path,'ink_sum.png'))




            cams = scene.getTrainCameras().copy()
            for idx in [69,73,64]:
                viewpoint_camera = cams[idx]

                
                image_height=int(viewpoint_camera.image_height)
                image_width=int(viewpoint_camera.image_width)
                fy = fov2focal(viewpoint_camera.FoVy, image_height)
                fx = fov2focal(viewpoint_camera.FoVx, image_width)

                (
                    xys,
                    depths,
                    radii,
                    conics_,
                    cov1d,
                    compensation,
                    num_tiles_hit,
                    cov3d,
                ) = project_gaussians(
                    means3d = gaussians.get_xyz,
                    scales = gaussians.get_scaling,
                    glob_scale = 1,
                    quats = gaussians.get_rotation,
                    viewmat = viewpoint_camera.world_view_transform.T,
                    fx = fx,
                    fy = fy,
                    cx = image_height/2,
                    cy = image_width/2,
                    img_height = image_height,
                    img_width = image_width,
                    block_width = B_SIZE,
                )


                # plot factor in a histogram
                mix_absorption_RGB = gaussians.get_ink_mix[:,:5] @ INK.absorption_RGB[:5]
                # S
                mix_scattering_RGB = gaussians.get_ink_mix[:,:5] @ INK.scattering_RGB[:5]

                mix_extinction_RGB = mix_absorption_RGB + mix_scattering_RGB
                with torch.no_grad():
                    blob_model_input = torch.cat([torch.sqrt(cov1d -  5e-3 + 1e-8)[:,None] / 0.05 , mix_extinction_RGB/255.0], dim=1)
                    blob_factors = blob_factor_model(blob_model_input) * torch.tensor([20.0, 60.0], dtype=torch.float32, device='cuda')
                # transmittance = torch.exp(-mix_extinction_RGB * torch.sqrt(cov1d -  5e-3 + 1e-8)[:,None] * blob_factors * 6).mean(dim=1)
            

                
                transmittance = torch.exp(-mix_extinction_RGB * torch.sqrt(cov1d - 5e-3 + 1e-8)[:,None] * blob_factors[:,1][:,None] * 6).mean(dim=1)
                factor =  1.0 - transmittance
                conics = conics_.clone()
                conics = conics.to(conics_.device)

                conics_x = conics[:, 0] * blob_factors[:,0]
                conics_y = conics[:, 2] * blob_factors[:,0]
                conics_xy = conics[:, 1] * blob_factors[:,0]

                conics = torch.stack([conics_x, conics_xy, conics_y], dim=1)



                ink_mix, out_alpha = rasterize_gaussians(
                        xys,
                        depths,
                        radii,
                        conics,
                        num_tiles_hit,
                        gaussians.get_ink_mix,
                        factor[:,None],
                        # opaque_opacity,
                        image_height,
                        image_width,
                        B_SIZE,
                        background,
                        return_alpha=True
                    )
                
                final_ink_mix = ink_mix.permute(2, 0, 1)
                _, out_img, _ = loss_ink_mix(final_ink_mix, final_ink_mix, out_alpha, viewpoint_camera, gt_images_folder_path)
                torchvision.utils.save_image(out_img, os.path.join(imgs_path, f"3000_optimize_{idx}.png"))
                print("Saved image for camera ", idx)
                
    if save_imgs:
            
            print(frames[0].shape)
            # img = Image.fromarray(frames[0])
            # save them as a gif with PIL
            # frames = [Image.fromarray(frame.transpose(1, 2, 0)).resize((100,100), Image.LANCZOS) for frame in frames]
            frames = [Image.fromarray(frame.transpose(1, 2, 0)) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                os.path.join(dataset.model_path, "result_imgs","training.gif"),
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
            # frames[0].save(
            #     f"training.gif",
            #     save_all=True,
            #     append_images=frames[1:],
            #     optimize=False,
            #     duration=5,
            #     loop=0,
            # )



if __name__ == "__main__":
    # NOTE: Here we add eval just to using the 100 training cameras. It is easier to find the original gt image this way
    '''
    python train_torch.py -m 3dgs_lego_train -s lego --iterations 3000 --sh_degree 0  --eval -w -r 8 
    python train_torch.py -m 3dgs_pattern_train -s pattern --iterations 3000 --sh_degree 0  --eval -w -r 8 

    '''
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")