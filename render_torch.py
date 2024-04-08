import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from utils.graphics_utils import fov2focal, getWorld2View, getWorld2View2



def render_torch_two_blobs(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
     
     with torch.no_grad():
        gaussians = GaussianModel(0)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        bg_color = [0, 0, 0, 0, 0, 0]

        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print("using background color: ", bg_color)

        model_path = dataset.model_path
        iteration = scene.loaded_iter
        views = scene.getTrainCameras()


        # set up rasterization configuration
        view_idx = 26
        viewpoint_camera = views[view_idx]
        image_height=int(viewpoint_camera.image_height)
        image_width=int(viewpoint_camera.image_width)
        fy = fov2focal(viewpoint_camera.FoVy, image_height)
        fx = fov2focal(viewpoint_camera.FoVx, image_width)
        # viewmatrix=viewpoint_camera.world_view_transform
        projmatrix=viewpoint_camera.full_proj_transform

        viewmatrix =getWorld2View(viewpoint_camera.R, viewpoint_camera.T)
        viewmatrix = torch.tensor(viewmatrix, dtype=torch.float32, device="cuda")



        temp = torch.tensor(getWorld2View2(viewpoint_camera.R, viewpoint_camera.T), dtype=torch.float32, device="cuda")
        print(fx, fy)
        # return 0


        B_SIZE = 16
        # test = (viewmatrix.unsqueeze(0).bmm(projmatrix.unsqueeze(0))).squeeze(0)

  
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

        print("finish projection, start rasterization")
        print("num hit tiles: ", num_tiles_hit)

        out_img = rasterize_gaussians(
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
        
        final_ink_mix = out_img.permute(2, 0, 1)
        print(final_ink_mix[:,400, 400])
        print("count non zeros: ", torch.count_nonzero(final_ink_mix))
        np.save('Two_blobs_torch/renders/final_ink_mix.npy', final_ink_mix.cpu().detach().numpy())
        # if not all zeros, save to text file
        if torch.count_nonzero(final_ink_mix) > 0:
            np.savetxt('Two_blobs_torch/renders/final_ink_mix.txt', final_ink_mix.cpu().detach().numpy().sum(axis=0), fmt='%.4f')


if __name__ == "__main__":

    '''
        python render_torch.py -m Two_blobs_torch/ -w --iteration 1000 --sh_degree 0
        python render_torch.py -m Two_blobs_torch/ --iteration 1000 --sh_degree 0
    '''


    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_torch_two_blobs(model.extract(args), args.iteration, pipeline.extract(args))