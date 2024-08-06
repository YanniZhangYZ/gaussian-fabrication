from ink_intrinsics import Ink
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal, getWorld2View, getWorld2View2
import open3d as o3d
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from mitsuba_utils import get_mixing_mitsuba_scene_dict, render_mitsuba_scene, write_to_vol_file, convert_data_to_C_indexing_style, get_camera_dict
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')


def voxel_splatting_analysis( gaussians: GaussianModel, dimensions: tuple, viewcameras: list, model_path:str):
    assert len(dimensions) == 3, "Dimensions must be a 3-tuple"

    # ============= Create a gaussians centers as a point cloud and get its bounding box ============= 
    g_pos = gaussians.get_xyz.detach().cpu().numpy()
    g_pcd = o3d.geometry.PointCloud()
    g_pcd.points = o3d.utility.Vector3dVector(g_pos)
    g_aabb = g_pcd.get_axis_aligned_bounding_box()
    g_aabb_len = g_aabb.get_extent() 

    # ============= Choose the voxel size as the minimum of the bounding box dimensions ============= 
    voxel_size = (np.round(g_aabb_len / np.array(dimensions), 3)).min()
    H, W, D = np.ceil(g_aabb_len / voxel_size).astype(int)
    g_min = g_aabb.get_min_bound() # NOTE: this is the min bound we use throughout the code
    print("The dimensions of the voxel grid are: ", H, W, D)
    print("The voxel size is: ", voxel_size)

    # =============  Create a voxel grid ============= 
    # Generate grid indices
    i = np.arange(H).reshape(H, 1, 1)
    j = np.arange(W).reshape(1, W, 1)
    k = np.arange(D).reshape(1, 1, D)

    # Broadcast i, j, k to form a complete grid of coordinates
    indices = np.stack(np.meshgrid(i, j, k, indexing='ij'), axis=-1)

    # Compute the center of each voxel
    grid_center = (indices + 0.5) * voxel_size + g_min

    # Compute the center of the grid
    grid_bbox_center = (grid_center[0,0,0] + grid_center[-1,-1,-1]) * 0.5
    assert grid_bbox_center.shape == (3,), "Grid center shape should be (3,)"
    assert grid_center.shape == (H, W, D, 3), "Grid center shape should be (H, W, D, 3)"

    print("The grid center is: ", grid_bbox_center)


    # ============= Voxel splatting =============
    voxel_ink = np.zeros((H, W, D, 6))
    voxel_opacity = np.zeros((H, W, D))
    debug_opacity = np.zeros((H, W, D))
    debug_ink = np.zeros((H, W, D, 6))


    # this is (N,6) containing the lower diagonal elements of the covariance matrix
    # The index correspondce are:
        # [:,0] <- [:,0,0]
        # [:,1] <- [:,0,1]
        # [:,2] <- [:,0,2]
        # [:,3] <- [:,1,1]
        # [:,4] <- [:,1,2]
        # [:,5] <- [:,2,2]
    g_cov_6 = gaussians.get_covariance().detach().cpu().numpy()
    g_cov_diag = g_cov_6[:,[0,3,5]]
    g_cov = g_cov_6[:,[0,1,2,1,3,4,2,4,5]].reshape(-1,3,3)
    g_inv_cov = np.linalg.inv(g_cov)
    assert g_cov_diag.shape == (g_pos.shape[0], 3), "Covariance diagonal shape should be (N,3)"
    assert g_cov.shape == (g_pos.shape[0], 3, 3), "Covariance shape should be (N,3,3)"

    # compute the min and max coordinates of aabb for each gaussian blob
    g_aabb_min = g_pos - 3 * np.sqrt(g_cov_diag)
    g_aabb_max = g_pos + 3 * np.sqrt(g_cov_diag)

    # prepare related data for voxel splatting
    # g_opacity = gaussians.get_opacity.detach().cpu().numpy()
    # g_opacity = np.ones_like(gaussians.get_opacity.detach().cpu().numpy())
    # # TODO:
    # g_opacity = np.ones_like(g_opacity)

    g_ink = gaussians.get_ink_mix.detach().cpu().numpy()



    # preprocess given ink mixtures given the transmittance
    INK = Ink()
    # K
    mix_absorption_RGB = gaussians.get_ink_mix[:,:5] @ INK.absorption_RGB[:5]
    # S
    mix_scattering_RGB = gaussians.get_ink_mix[:,:5] @ INK.scattering_RGB[:5]

    mix_extinction_RGB = mix_absorption_RGB + mix_scattering_RGB
    scale_xyz = gaussians.get_scaling.detach().cpu().numpy()
    transmittance_ = np.exp(-mix_extinction_RGB.detach().cpu().numpy() * scale_xyz.mean(axis = 1)[:,None] * 6).mean(axis=1)
    g_opacity = 1.0 - transmittance_


    print("max opacity: ", g_opacity.max())
    print("min opacity: ", g_opacity.min())

    # print the ink mixture of min and max opacity
    max_opacity_idx = np.argmax(g_opacity)
    min_opacity_idx = np.argmin(g_opacity)
    print("max opacity ink: ", g_ink[max_opacity_idx])
    print("min opacity ink: ", g_ink[min_opacity_idx])


    # get the idx of the gaussians that have opacity larger than 0.0367
    mask = g_opacity.reshape(-1) > 0.3
    test_idx = np.argwhere(mask).reshape(-1)

    # for g_idx in tqdm(test_idx):
    for g_idx in tqdm(range(g_pos.shape[0])):
    

        #  Method 1: inside aabb
        min_voxel_idx = np.floor((g_aabb_min[g_idx] - g_min) / voxel_size + 0.5).astype(int)
        max_voxel_idx = np.ceil((g_aabb_max[g_idx] - g_min) / voxel_size - 0.5).astype(int)

        # # Method 2:  3*3 around blob center
        # min_voxel_idx = np.floor((g_pos[g_idx] - g_min) / voxel_size - 0.5).astype(int)
        # max_voxel_idx = np.ceil((g_pos[g_idx] - g_min) / voxel_size + 0.5).astype(int)


        related_voxels_idx = indices[min_voxel_idx[0]:max_voxel_idx[0], min_voxel_idx[1]:max_voxel_idx[1], min_voxel_idx[2]:max_voxel_idx[2]].reshape(-1,3)
        related_voxels_center = grid_center[min_voxel_idx[0]:max_voxel_idx[0], min_voxel_idx[1]:max_voxel_idx[1], min_voxel_idx[2]:max_voxel_idx[2]].reshape(-1,3)
       

        # Transform Grid Points, Check Inclusion Using the Ellipsoid Equation
        eigen_val, eigen_vec = np.linalg.eigh(g_cov[g_idx]) # NOTE: eigen_vec's columns are the eigenvectors
        
        if (eigen_val <= 0).any() :
            # here it is not a 3d gaussian. From the probability aspect, it could not splat to any voxel (prob = 0)
            continue

        # # Work together with method 2
        # inside_mahalanobis_dist = (related_voxels_center - g_pos[g_idx])@ g_inv_cov[g_idx] * (related_voxels_center - g_pos[g_idx])
        # inside_mahalanobis_dist = np.sum(inside_mahalanobis_dist, axis=1)
        # inside_idx =  related_voxels_idx.copy()


        # Translate points by the negative of the mean
        translated_points = related_voxels_center - g_pos[g_idx]
        
        # Rotate points into the coordinate system of the blob
        transformed_points = translated_points @ eigen_vec
        
        # Scale factors for the ellipsoid axes

        scales = 3 * np.sqrt(eigen_val) 

        
        # Check if each point is within the ellipsoid
        is_inside_mask = ((transformed_points[:,0] / scales[0]) ** 2 + 
                (transformed_points[:,1] / scales[1]) ** 2 + 
                (transformed_points[:,2] / scales[2]) ** 2 <= 1.0)
        
        
        # for the voxels that are inside the 0-level set, compute the ink mixture and opacity
        temp_idx = np.argwhere(is_inside_mask).reshape(-1)
        
        inside_idx = related_voxels_idx[temp_idx]
        inside_voxels_centers = related_voxels_center[temp_idx]

        inside_mahalanobis_dist = (inside_voxels_centers - g_pos[g_idx])@ g_inv_cov[g_idx] * (inside_voxels_centers - g_pos[g_idx])
        inside_mahalanobis_dist = np.sum(inside_mahalanobis_dist, axis=1)
        assert inside_mahalanobis_dist.shape == (inside_voxels_centers.shape[0],), "Mahalanobis distance shape should be (#,), actually it is {}".format(inside_mahalanobis_dist.shape)
        #  NOTE: This assertion may be need further consideration
        # assert (inside_mahalanobis_dist - 9.0 <= 1e-4 ).all(), "Mahalanobis distance should be smaller than 9.0, eigen val{}, max {},  min {}".format(eigen_val, inside_mahalanobis_dist.max(), inside_mahalanobis_dist.min())

        inside_probs = np.exp(-0.5 * inside_mahalanobis_dist)
        i,j,k = inside_idx[:,0], inside_idx[:,1], inside_idx[:,2]



        voxel_opacity[i,j,k] += inside_probs * g_opacity[g_idx]

        voxel_ink[i,j,k] += g_ink[g_idx] * (inside_probs * g_opacity[g_idx])[:,None]


    #  ============= Post process the ink mixture =============
    # store the original ink mixture for debugging
    # np.save(os.path.join(model_path,"voxel_ink.npy"), voxel_ink)

    print("Post processing the ink mixture")

    # number of voxels that is not all transparent

    mask = voxel_opacity > 0.0
    print("Number of voxels that is not all transparent: ", mask.sum())
    

    assert False, "stop here"




    # if opacity sum is larger than 1, normalize the ink mixture
    mask = voxel_opacity > 1.0
    voxel_ink[mask] /= voxel_opacity[mask][:,None]
    voxel_opacity[mask] = 1.0
    assert np.isnan(voxel_ink[mask]).any() == False, "Ink mixture should not have nan"
    # prepare debug_opacity for visualization
    debug_opacity = voxel_opacity.copy()
    debug_ink = voxel_ink.copy()



    # HELPER.debug_histogram(g_opacity, "opacity_histogram_g_blob.png")
    # HELPER.debug_histogram(g_ink[:,4], "white_histogram_g_blob.png")
    # debug_mask =  voxel_opacity > 0.0
    # HELPER.debug_histogram(voxel_opacity[debug_mask], "opacity_histogram_has_val_voxel.png")
    # assert False, "stop here"



    # if opacity is smaller than 1, add transparent ink
    mask = voxel_opacity < 1.0
    trans_ink = 1.0 - voxel_ink[mask].sum(axis=1)
    trans_ink[trans_ink < 0] = 0
    
    assert voxel_ink[mask].sum(axis=1).shape == voxel_opacity[mask].shape, "Ink mixture sum shape should be the same as opacity shape"
    assert (trans_ink >= 0.0).all(), "The added transparent ink should be positive"
    voxel_ink[mask,5] += trans_ink

    # now the ink mixture of each voxel sum should be 1
    assert np.abs(voxel_ink.sum(axis=3) - 1.0).max() < 1e-6, "Ink mixture should sum to 1.0"

    # np.save(os.path.join(model_path,"voxel_ink_post.npy"), voxel_ink)

    print("mean voxel ink in the 6 channels: ", voxel_ink.mean(axis=(0, 1, 2)))






def mitsuba_gaussians(dataset : ModelParams, iteration : int, pipeline : PipelineParams, readvol=False):
    with torch.no_grad():
        gaussians = GaussianModel(0)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        cameras = scene.getTrainCameras()

        g_opacity = gaussians.get_opacity.detach().cpu().numpy()
        voxel_splatting_analysis(gaussians,[400,400,400], cameras, dataset.model_path)







if __name__ == "__main__":

    '''
        python pc_analysis.py -m 3dgs_lego_train -w --iteration 1 --sh_degree 0

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

    mitsuba_gaussians(model.extract(args), args.iteration, pipeline.extract(args), readvol=True)
