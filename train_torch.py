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
from utils.loss_utils import l1_loss, ssim
from train import prepare_output_and_logger

from mitsuba_conversion import debug_plot_g


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
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=rgba, s=1.0)
    plt.savefig(path)


def ink_to_RGB(mix, H, W):
    # back_ground_mask = torch.all(mix == 0.0, axis=0).int()
    # blob_mask = 1 - back_ground_mask
    # transparent_mask = 1.0 - mix.sum(axis=0)
    # transparent_blob_mask = blob_mask - mix.sum(axis=0)
    # # NOTE: The training image has transparent background!
    # # NOTE: Therefore we add white ink only to the place where there is a blob!
    # # mix[4,:,:] = mix[4,:,:] + transparent_mask  # This is used in the original ink_to_rgb.py file
    # mix[4,:,:] = mix[4,:,:] + transparent_blob_mask

    # C, H, W = mix.shape
    # mix = mix.permute(1,2,0).view(-1,C)

    if (mix < 0.0).any():
        mask = torch.nonzero(mix < 0.0)
        print(mask)
        print(mix[mask[0]])
        # print(temp[mask[0]])
        assert False, "Negative ink concentration inside ink_to_RGB"
    
    # mix: (H*W,6) array of ink mixtures
    # K
    mix_K = mix @ INK.absorption_matrix
    # S
    mix_S = mix @ INK.scattering_matrix + 1e-8

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

    # Saunderson’s correction reflectance coefficient, commonly used values
    k1 = 0.04
    k2 = 0.6
    # equation 6
    R_mix = (1 - k1) * (1 - k2) * R_mix / (1 - k2 * R_mix)

    with torch.no_grad():
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
    with torch.no_grad():
        sRGB_matrix = torch.tensor([[3.2406, -1.5372, -0.4986],
                                [-0.9689, 1.8758, 0.0415],
                                [0.0557, -0.2040, 1.0570]], device="cuda")
    sRGB = ((sRGB_matrix @ XYZ) / Y_D56).T
    sRGB = torch.clip(sRGB,0,1).view(H, W, 3).permute(2,0,1)
    return sRGB



def image_path_to_tensor(image_path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    img = img.resize((100, 100))
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3] # Original gsplat: Here it is in shape H, W, 3
    img_tensor = img_tensor.permute(2, 0, 1) # For our optimization implementation, we need 3, H, W
    return img_tensor.to("cuda")


from kornia.color import rgb_to_lab
import math
from torchmetrics.image import StructuralSimilarityIndexMeasure

def loss_ink_mix(mix, out_alpha, viewpoint_cam):
    
    C, H, W = mix.shape
    # mix = F.relu(mix)
    # print(mix.shape)

    #TODO: deal with error in the future
    if (mix.sum(dim=0) > 1.0 + 1e-1).any():
        print(mix.sum(dim=0).max())
        raise RuntimeError("TODO: normalize the ink")


    # epsilon = 1e-8
    # mix = mix / (mix.sum(dim=0, keepdim=True) + epsilon)

    back_ground_mask = mix.sum(dim=0) < 1e-6
    blob_mask = torch.logical_not(back_ground_mask)
    transparent_mask = 1.0 - mix.sum(axis=0)
    transparent_blob_mask = blob_mask.float() - mix.sum(axis=0)

    # NOTE: The training image has transparent background!
    # NOTE: Therefore we add white ink only to the place where there is a blob!
    # mix[4,:,:] = mix[4,:,:] + transparent_mask  # This is used in the original ink_to_rgb.py file
    # mix[4,:,:] = mix[4,:,:] + transparent_blob_mask

    C, H, W = mix.shape
    mix = mix.permute(1,2,0).view(-1,C)

    current_render = ink_to_RGB(mix, H, W)

    # loss_mse = F.mse_loss(current_render, gt_rgb)

    gt_image = viewpoint_cam.original_image.cuda()
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
    Ll1 = l1_loss(current_render, gt_image)

    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(current_render, gt_image))

    # lab_GT = rgb_to_lab(gt_rgb.reshape(1, 3, H , W)).reshape(H, W,3)
    # lab_image = rgb_to_lab(current_render.reshape(1, 3, H,W)).reshape(H,W,3)
    # delta_e76 = torch.sqrt(torch.sum((lab_image - lab_GT)**2, dim=(0, 1))/(H*W))
    # loss_delta_e76 = torch.mean(delta_e76)

    # ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    # loss_ssim = ssim(current_render.reshape(1,3, H, W), gt_rgb.reshape(1,3, H, W))
    # return 0.2* loss_ssim + 0.7 * loss_mse + 0.1 * loss_delta_e76, current_render

    return loss, current_render

     


def training(dataset : ModelParams, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, save_imgs=False):
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
    # view_idx = 26
    # viewpoint_camera = views[view_idx]
    # image_height=int(viewpoint_camera.image_height)
    # image_width=int(viewpoint_camera.image_width)
    # fy = fov2focal(viewpoint_camera.FoVy, image_height)
    # fx = fov2focal(viewpoint_camera.FoVx, image_width)

    # print("GT 000 :", viewpoint_camera.original_image.cuda()[:, 0, 0])

    l = [
        {'params': [gaussians._xyz], 'lr': lr, "name": "xyz"},
        {'params': [gaussians._ink_mix], 'lr': lr, "name": "ink"},
        {'params': [gaussians._opacity], 'lr': lr, "name": "opacity"},
        {'params': [gaussians._scaling], 'lr': lr, "name": "scaling"},
        {'params': [gaussians._rotation], 'lr': lr, "name": "rotation"}
    ]

    optimizer = torch.optim.Adam(l, lr=0.01, eps=1e-15) 
    frames = []
    B_SIZE = 16

    #  TODO
    # gt_img = image_path_to_tensor('lego/train/r_25.png')
    
    viewpoint_stack = None
    # torch.autograd.set_detect_anomaly(True)

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


        ink_mix, out_alpha = rasterize_gaussians(
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
                return_alpha=True
            )
        
        final_ink_mix = ink_mix.permute(2, 0, 1)
        # print("after: ", (final_ink_mix.sum(dim=0) > 1.0 + 1e-2).any())

        # print("final_ink_mix: ", final_ink_mix[:, int(final_ink_mix.shape[1]/2), int(final_ink_mix.shape[2]/2)])

        # out_img = ink_to_RGB(final_ink_mix)
        # torchvision.utils.save_image(torch.tensor(out_img), "ink_torch.png")
        torch.cuda.synchronize()
        loss, out_img = loss_ink_mix(final_ink_mix, out_alpha, viewpoint_camera)

        
        

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
        torch.cuda.synchronize()
        optimizer.step()
        print(f"Iteration {i + 1}/{opt.iterations}, Loss: {loss.item()}")


        # if i % 50 == 0:
        # #     debug_plot_g(gaussians, "gaussian_init_data.png")
        # #     debug_plot_g(gaussians, "gaussian_init_data_alpha.png", use_alpha=True)
        #     plot_pos = gaussians.get_xyz.detach().cpu().numpy()
        #     plot_alpha = gaussians.get_opacity.detach().cpu().numpy()
        #     debug_color = np.zeros((plot_pos.shape[0], 3))
        #     debug_color[:, 0] = 1.0
        #     debug_alpha = np.ones(plot_pos.shape[0])

        #     torchvision.utils.save_image(out_img, "3000_optimize.png")
        #     debug_plot(plot_pos, 
        #                debug_color, 
        #                plot_alpha,
        #                "gaussian_init_data_alpha.png"
        #                )
        #     debug_plot(plot_pos, 
        #                debug_color, 
        #                debug_alpha,
        #                 "gaussian_init_data.png"
        #                )


        if save_imgs and i % 5 == 0:
            frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))


        if i == opt.iterations - 1:
            torchvision.utils.save_image(out_img, "3000_optimize.png")
            print("\n[ITER {}] Saving Gaussians".format(i))
            # NOTE: to compare with reading from ply
            print("{}'s ink mix: {}".format(0, gaussians.get_ink_mix[0]))
            print("{}'s ink mix: {}".format(5000, gaussians.get_ink_mix[5000]))
            debug_plot_g(gaussians, "gaussian_init_data.png")
            debug_plot_g(gaussians, "gaussian_init_data_alpha.png", use_alpha=True)
            scene.save(i+1)



    if save_imgs:
            
            print(frames[0].shape)
            # img = Image.fromarray(frames[0])
            # save them as a gif with PIL
            # frames = [Image.fromarray(frame.transpose(1, 2, 0)).resize((100,100), Image.LANCZOS) for frame in frames]
            frames = [Image.fromarray(frame.transpose(1, 2, 0)) for frame in frames]

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
    '''
    python train_torch.py -m 3dgs_lego_train -s lego --iterations 3000 --sh_degree 0 -r 8
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