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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

def render_two_blobs(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(0)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print("using background color: ", bg_color)

        model_path = dataset.model_path
        iteration = scene.loaded_iter
        views = scene.getTrainCameras()

        render_path = os.path.join(model_path, "renders")
        makedirs(render_path, exist_ok=True)


        # view_idx = 32
        view_idx = 26
        print("Rendering progress")
        result = render(views[view_idx], gaussians, pipeline, background)
        rendering = result["render"]
        final_ink_mix = result["final_ink_mix"]


        print("color: ", rendering.shape)
        print("final_ink_mix: ", final_ink_mix.shape)
        print(torch.count_nonzero(rendering,dim=(1,2)))
        print(torch.count_nonzero(final_ink_mix,dim=(1,2)))


        np.save('Two_blobs/renders/final_ink_mix.npy', final_ink_mix.cpu().detach().numpy())
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(view_idx) + ".png"))

        # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        #     rendering = render(view, gaussians, pipeline, background)["render"]
        #     torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = ArgumentParser(description="Testing script parameters")
#     model = ModelParams(parser, sentinel=True)
#     pipeline = PipelineParams(parser)
#     parser.add_argument("--iteration", default=-1, type=int)
#     parser.add_argument("--skip_train", action="store_true")
#     parser.add_argument("--skip_test", action="store_true")
#     parser.add_argument("--quiet", action="store_true")
#     args = get_combined_args(parser)
#     print("Rendering " + args.model_path)

#     # Initialize system state (RNG)
#     safe_state(args.quiet)

#     render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
            

if __name__ == "__main__":

    '''
        python render.py -m Two_blobs/ -w --iteration 1000 --sh_degree 0
        python render.py -m Two_blobs/ --iteration 1000 --sh_degree 0
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

    render_two_blobs(model.extract(args), args.iteration, pipeline.extract(args))

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)