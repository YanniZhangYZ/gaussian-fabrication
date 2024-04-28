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
import matplotlib.pyplot as plt

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


def ink_to_RGB(mix, H, W):
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
    mix_K = mix @ INK.absorption_matrix[:5]
    # S
    mix_S = mix @ INK.scattering_matrix[:5] + 1e-8

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
    sRGB = torch.where(sRGB <= 0.0031308,
                    12.92 * sRGB,
                    1.055 * torch.pow(sRGB, 1 / 2.4) - 0.055)
        

    sRGB = torch.clip(sRGB,0,1).view(H, W, 3).permute(2,0,1)
    return sRGB




from kornia.color import rgb_to_lab
import math
from torchmetrics.image import StructuralSimilarityIndexMeasure

def loss_ink_mix(mix, out_alpha, viewpoint_cam, gt_images_folder_path):
    
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
    mix = mix.permute(1,2,0).view(-1,C)

    current_render = ink_to_RGB(mix, H, W)


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
    current_render_rgba = torch.cat([current_render, out_alpha], dim=0)
    gt_image_rgba = torch.cat([gt_image, gt_original_rgba[3:4,:,:]], dim=0)
    assert current_render_rgba.shape == gt_image_rgba.shape, (current_render_rgba.shape, gt_image_rgba.shape)
    Ll1 = l1_loss(current_render_rgba, gt_image_rgba)

    
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
    # view_idx = 2
    # viewpoint_camera = views[view_idx]
    # image_height=int(viewpoint_camera.image_height)
    # image_width=int(viewpoint_camera.image_width)
    # fy = fov2focal(viewpoint_camera.FoVy, image_height)
    # fx = fov2focal(viewpoint_camera.FoVx, image_width)

    gaussians._xyz.requires_grad = False
    gaussians._scaling.requires_grad = False
    gaussians._rotation.requires_grad = False

    l = [
        # {'params': [gaussians._xyz], 'lr': lr, "name": "xyz"},
        {'params': [gaussians._ink_mix], 'lr': lr, "name": "ink"},
        {'params': [gaussians._opacity], 'lr': lr, "name": "opacity"},
        # {'params': [gaussians._scaling], 'lr': lr, "name": "scaling"},
        # {'params': [gaussians._rotation], 'lr': lr, "name": "rotation"}
    ]

    optimizer = torch.optim.Adam(l, lr=0.01, eps=1e-15) 
    frames = []
    B_SIZE = 16

    #  TODO
    
    viewpoint_stack = None
    gt_images_folder_path = os.path.join(dataset.source_path, "train")
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


        # NOTE: in the most ideal case, for the place that GT has no color, the rasterize result alpha should be 0
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
        loss, out_img = loss_ink_mix(final_ink_mix, out_alpha, viewpoint_camera, gt_images_folder_path)
        

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

        # return 0


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
            imgs_path = os.path.join(dataset.model_path, "result_imgs")
            if not os.path.exists(imgs_path):
                os.makedirs(imgs_path)
            torchvision.utils.save_image(out_img, os.path.join(imgs_path, "3000_optimize.png"))
            print("\n[ITER {}] Saving Gaussians".format(i))
            # NOTE: to compare with reading from ply
            print("{}'s ink mix: {}".format(0, gaussians.get_ink_mix[0]))
            print("{}'s ink mix: {}".format(5000, gaussians.get_ink_mix[5000]))
            pos = gaussians.get_xyz.detach().cpu().numpy()
            debug_color = gaussians.convert_center_ink_2_rgb_4_debug()
            assert debug_color.shape == pos.shape, (debug_color.shape, pos.shape)
            debug_alpha = gaussians.get_opacity.detach().cpu().numpy()
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