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
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
import numpy as np
from PIL import Image   
import torch.nn.functional as F
import torchvision


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# Load ink intrinsic data
INK = Ink()


def ink_to_RGB(mix):
    back_ground_mask =torch.all(mix == 0.0, axis=0).int()
    blob_mask = 1 - back_ground_mask
    transparent_mask = 1.0 - mix.sum(axis=0)
    transparent_blob_mask = blob_mask - mix.sum(axis=0)
    # NOTE: The training image has transparent background!
    # NOTE: Therefore we add white ink only to the place where has blob!
    # mix[4,:,:] = mix[4,:,:] + transparent_mask  # This is used in the original ink_to_rgb.py file
    mix[4,:,:] = mix[4,:,:] + transparent_blob_mask

    C, H, W = mix.shape
    mix = mix.permute(1,2,0).view(-1,C)
    
    # mix: (H*W,6) array of ink mixtures
    # K
    mix_K = mix @ INK.absorption_matrix + 1e-8
    # S
    mix_S = mix @ INK.scattering_matrix + 1e-8
    #equation 2
    R_mix = 1 + mix_K / mix_S - torch.sqrt( (mix_K / mix_S)**2 + 2 * mix_K / mix_S)
    # Saunderson’s correction reflectance coefficient, commonly used values
    k1 = 0.04
    k2 = 0.6
    # equation 6
    R_mix = (1-k1) * (1 - k2) * R_mix / (1 - k2 * R_mix)


    # equation 3 - 5
    x_D56 = INK.x_observer * INK.sampled_illuminant_D65
    # x_D56 /= np.sum(x_D56)
    y_D56 = INK.y_observer * INK.sampled_illuminant_D65
    # y_D56 /= np.sum(y_D56)
    z_D56 = INK.z_observer * INK.sampled_illuminant_D65
    # z_D56 /= np.sum(z_D56)
    X = R_mix @ x_D56
    Y = R_mix @ y_D56
    Z = R_mix @ z_D56

    XYZ = torch.stack([X,Y,Z],axis=1).T
    Y_D56 = torch.sum(y_D56)

    # Convert XYZ to sRGB, Equation 7
    sRGB_matrix = torch.tensor([[3.2406, -1.5372, -0.4986],
                            [-0.9689, 1.8758, 0.0415],
                            [0.0557, -0.2040, 1.0570]], device="cuda")
    sRGB = ((sRGB_matrix @ XYZ) / Y_D56).T
    sRGB = torch.clip(sRGB,0,1).view(H, W, 3).permute(2,0,1)
    return sRGB



def image_path_to_tensor(image_path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3] # Original gsplat: Here it is in shape H, W, 3
    img_tensor = img_tensor.permute(2, 0, 1) # For our optimization implementation, we need 3, H, W
    return img_tensor.to("cuda")


from kornia.color import rgb_to_lab
import math
from torchmetrics.image import StructuralSimilarityIndexMeasure

def loss_ink_mix(mix, gt_rgb):
    
    C, H, W = mix.shape
    mix = F.relu(mix)
    epsilon = 1e-8
    mix = mix / (mix.sum(dim=1, keepdim=True) + epsilon)
    current_render = ink_to_RGB(mix)

    loss_mse = F.mse_loss(current_render, gt_rgb)

    lab_GT = rgb_to_lab(gt_rgb.reshape(1, 3, H , W)).reshape(H, W,3)
    lab_image = rgb_to_lab(current_render.reshape(1, 3, H,W)).reshape(H,W,3)
    delta_e76 = torch.sqrt(torch.sum((lab_image - lab_GT)**2, dim=(0, 1))/(H*W))
    loss_delta_e76 = torch.mean(delta_e76)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    loss_ssim = ssim(current_render.reshape(1,3, H, W), gt_rgb.reshape(1,3, H, W))
    return 0.2* loss_ssim + 0.7 * loss_mse + 0.1 * loss_delta_e76, current_render

     


def training(dataset : ModelParams, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, save_imgs=True):
    lr = 0.01
    gaussians = GaussianModel(0)
    scene = Scene(dataset, gaussians)
    # gaussians.training_setup(opt)

    bg_color = [0, 0, 0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


     # set up rasterization configuration
    views = scene.getTrainCameras()
    view_idx = 26
    viewpoint_camera = views[view_idx]
    image_height=int(viewpoint_camera.image_height)
    image_width=int(viewpoint_camera.image_width)
    fy = fov2focal(viewpoint_camera.FoVy, image_height)
    fx = fov2focal(viewpoint_camera.FoVx, image_width)

    l = [
        {'params': [gaussians._xyz], 'lr': lr, "name": "xyz"},
        {'params': [gaussians._ink_mix], 'lr': lr, "name": "f_dc"},
        {'params': [gaussians._opacity], 'lr': lr, "name": "opacity"},
        {'params': [gaussians._scaling], 'lr': lr, "name": "scaling"},
        {'params': [gaussians._rotation], 'lr': lr, "name": "rotation"}
    ]

    optimizer = torch.optim.Adam(l, lr=0.01, eps=1e-15) 
    frames = []
    B_SIZE = 16

    #  TODO
    gt_img = image_path_to_tensor('lego/train/r_25.png')


    for i in range(opt.iterations):
        (
            xys,
            depths,
            radii,
            conics,
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


        ink_mix = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                gaussians.get_ink_mix,
                gaussians.get_opacity,
                image_height,
                image_width,
                B_SIZE,
                background,
            )
        
        final_ink_mix = ink_mix .permute(2, 0, 1)

        # print("final_ink_mix: ", final_ink_mix[:, int(final_ink_mix.shape[1]/2), int(final_ink_mix.shape[2]/2)])

        # out_img = ink_to_RGB(final_ink_mix)
        # torchvision.utils.save_image(torch.tensor(out_img), "ink_torch.png")



        # return 0
        torch.cuda.synchronize()
        loss, out_img = loss_ink_mix(final_ink_mix, gt_img)
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        print(f"Iteration {i + 1}/{opt.iterations}, Loss: {loss.item()}")
        if save_imgs and i  % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))

    if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )



if __name__ == "__main__":
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
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")