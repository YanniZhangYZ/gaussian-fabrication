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
from scene.cameras import Camera


from utils.general_utils import strip_symmetric, build_scaling_rotation

from scipy.spatial.transform import Rotation as Rot
from mitsuba_utils import get_mixing_mitsuba_scene_dict, render_mitsuba_scene, write_to_vol_file, convert_data_to_C_indexing_style, get_camera_dict

from voxel_splatting import Helper
import open3d as o3d


INK = Ink()

def ink_to_RGB(mix, H, W, scale_z):
    '''
    mix: (H*W,6) array of ink mixtures
    output: (3,H,W) array of RGB values
    
    '''
    # mix: (H*W,6) array of ink mixtures
    # C, H, W = mix.shape
    mix =  mix[:,:5]


    # preprocess given ink mixtures given the transmittance
    # K
    # mix_absorption_RGB = mix[:,:4] @ INK.absorption_RGB[:4]
    # # S
    # mix_scattering_RGB = mix[:,:4] @ INK.scattering_RGB[:4]
    # mix_extinction_RGB = mix_absorption_RGB + mix_scattering_RGB
    # transmittance = torch.exp(-mix_extinction_RGB * scale_z*6).min(dim=1).values
    # mix = mix * (1.0 - transmittance[:,None])

    print( mix[256*127 + 127])


    residual = 1.0 - mix.sum(dim=1)

    mix[:,4] += residual

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

    sRGB = torch.clip(sRGB,0,1).view(H, W, 3).permute(2,0,1)

    if torch.isnan(sRGB).any():
        temp = sRGB.clone().detach()
        mask = torch.nonzero(torch.isnan(temp))
        # print(R_mix.shape)
        print(mask)
        print(temp[mask[0]])
        assert False, "sRGB has nan"

    return sRGB




def get_one_blob(scale_z,idx):

    scale_ = np.ones((1, 3)) * 0.5
    scale_[0,2] = scale_z
    # scale_[:, 2] = np.linspace(0.1, 10, 100)
    scale_ = torch.tensor(scale_, dtype=torch.float32, device="cuda")
    quat_ = torch.tensor(np.array([[1.0, 0.0, 0.0, 0.0]] * 1), dtype=torch.float32, device="cuda")

    mean3d_ = torch.tensor(np.array([[0,0,0]]), dtype=torch.float32, device="cuda")

    opacities =  torch.tensor(np.ones(1), dtype=torch.float32, device="cuda")


    mixtures =  np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    new_x =  torch.tensor(mixtures, dtype=torch.float32, device="cuda")


    bg_color = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # CMYKWT the backgroud is fully black ink
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    #  prepare camera that looks to the z-axis
    R_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    T_ = np.array([0, 0, 10])

    image_ = torch.ones((3, 800, 800))
    viewpoint_camera = Camera(colmap_id = 0, 
                      R = R_, 
                      T = T_, 
                      FoVx = 0.6911112070083618, 
                      FoVy = 0.6911112070083618, 
                      image = image_, 
                      gt_alpha_mask = None,
                      image_name = "r_0", 
                      uid = 0)

    B_SIZE = 16


    image_height= 256
    image_width= 256
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
        means3d = mean3d_,
        scales = scale_,
        glob_scale = 1,
        quats = quat_,
        viewmat = viewpoint_camera.world_view_transform.T,
        fx = fx,
        fy = fy,
        cx = image_height/2,
        cy = image_width/2,
        img_height = image_height,
        img_width = image_width,
        block_width = B_SIZE,
    )
    assert num_tiles_hit.sum() > 0, "No tiles hit"



    # preprocess given ink mixtures given the transmittance
    # K
    mix_absorption_RGB = new_x[:,:4] @ INK.absorption_RGB[:4]
    # S
    mix_scattering_RGB = new_x[:,:4] @ INK.scattering_RGB[:4]
    mix_extinction_RGB = mix_absorption_RGB + mix_scattering_RGB
    transmittance = torch.exp(-mix_extinction_RGB * scale_z * 6).mean(dim=1)
    # new_x = new_x * (1.0 - transmittance[:,None])
    opacities = opacities * (1.0 - transmittance)

    ink_mix, out_alpha = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            new_x,
            opacities,
            image_height,
            image_width,
            B_SIZE,
            background,
            return_alpha=True
    )



    


    final_ink_mix = ink_mix.permute(2, 0, 1)
    C, H, W = final_ink_mix.shape
    final_ink_mix = final_ink_mix.permute(1,2,0).view(-1,C)

    out_img = ink_to_RGB(final_ink_mix, H, W,scale_z)


    parent_path = "blob_thickness_test"
    rasterize_path = os.path.join(parent_path, "rasterize")
    alpha_path = os.path.join(parent_path, "alpha")
    slice_path = os.path.join(parent_path, "slice")
    render_path = os.path.join(parent_path, "render")
    sum_path = os.path.join(parent_path, "sum")
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
        os.makedirs(rasterize_path)
        os.makedirs(alpha_path)
        os.makedirs(slice_path)
        os.makedirs(render_path)
        os.makedirs(sum_path)
    

    # only include two decimal points in file name( format: idx_scale_z.png)
    torchvision.utils.save_image(out_img, rasterize_path+f"/{idx}_{scale_z:.2f}.png")

    debug_sum = np.squeeze(ink_mix[:,:,2].detach().cpu().numpy())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(debug_sum, cmap='viridis') 
    fig.colorbar(cax)
    fig.savefig(os.path.join(sum_path, f"sum_{idx}.png"))

    debug_alpha = np.squeeze(out_alpha.detach().cpu().numpy())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(debug_alpha, cmap='viridis') 
    fig.colorbar(cax)
    fig.savefig(os.path.join(alpha_path, f"alpha_{idx}.png"))

    return viewpoint_camera





def get_blobs():
    
    # prepare 10 * 10 blobs, these blobs should have the same z value
    # the x and y values should be in the range of -1 to 1
    x = np.linspace(-3,3, 10)
    y = np.linspace(-3,3, 10)
    z = np.zeros(100)

    xy = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)
    xyz = np.concatenate([xy, z.reshape(-1, 1)], axis=1)
    
    
    #  prepare the blobs
    
    scale_ = np.ones((100, 3)) * 0.05
    scale_[0,0] = 0.5
    # scale_[:, 2] = np.linspace(0.1, 10, 100)
    scale_ = torch.tensor(scale_, dtype=torch.float32, device="cuda")
    quat_ = torch.tensor(np.array([[1.0, 0.0, 0.0, 0.0]] * 100), dtype=torch.float32, device="cuda")

    mean3d_ = torch.tensor(xyz, dtype=torch.float32, device="cuda")
    # mean3d_[:,2] = (scale_[:, 2]**2) / 2.0



    L = build_scaling_rotation(scale_, quat_)
    actual_covariance = L @ L.transpose(1, 2)

    print("actual_covariance", actual_covariance[0])
    print("actual_covariance", actual_covariance[1])
    print("actual_covariance", actual_covariance[60])

    # assert False

    opacities =  torch.tensor(np.ones(100), dtype=torch.float32, device="cuda")


    mixtures =  np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 100)
    new_x =  torch.tensor(mixtures, dtype=torch.float32, device="cuda")
	
    # new_x = torch.zeros((mixtures.shape[0], 6), device=mixtures.device)
    # new_x[:, :5] = F.softmax(mixtures[:,:5], dim=-1).clamp(0.0, 1.0)

    # print("new_x", new_x[0])

	



    bg_color = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # CMYKWT the backgroud is fully black ink
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    #  prepare camera that looks to the z-axis
    R_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    T_ = np.array([0, 0, 10])

    image_ = torch.ones((3, 800, 800))
    viewpoint_camera = Camera(colmap_id = 0, 
                      R = R_, 
                      T = T_, 
                      FoVx = 0.6911112070083618, 
                      FoVy = 0.6911112070083618, 
                      image = image_, 
                      gt_alpha_mask = None,
                      image_name = "r_0", 
                      uid = 0)

    B_SIZE = 16


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
        means3d = mean3d_,
        scales = scale_,
        glob_scale = 1,
        quats = quat_,
        viewmat = viewpoint_camera.world_view_transform.T,
        fx = fx,
        fy = fy,
        cx = image_height/2,
        cy = image_width/2,
        img_height = image_height,
        img_width = image_width,
        block_width = B_SIZE,
    )

    print("num_tiles_hit", num_tiles_hit.sum())
    assert num_tiles_hit.sum() > 0, "No tiles hit"

    ink_mix, out_alpha = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            new_x,
            opacities,
            image_height,
            image_width,
            B_SIZE,
            background,
            return_alpha=True
    )

    final_ink_mix = ink_mix.permute(2, 0, 1)
    _, out_img = loss_ink_mix(final_ink_mix, out_alpha, viewpoint_camera, 'lego/train')


    torchvision.utils.save_image(out_img, "blob_thickness_test.png")





def voxel_splatting(dimensions, viewcamera, model_path, scale_z, name_idx):
    
    
    HELPER = Helper()
    assert len(dimensions) == 3, "Dimensions must be a 3-tuple"

    g_pos = np.array([[0.0, 0.0, 0.0]])
    z_len = max((scale_z ** 2) * 6, 2)
    # g_aabb_len = np.array([2,2, z_len])
    g_aabb_len = np.array([6,6,6])

    temp_vertex =  np.array([[3,3,3],
                             [3,3,-3],
                             [3,-3,3],
                             [3,-3,-3],
                             [-3,3,3],
                             [-3,3,-3],
                             [-3,-3,3],
                             [-3,-3,-3]])
    
    g_pcd = o3d.geometry.PointCloud()
    g_pcd.points = o3d.utility.Vector3dVector(temp_vertex)
    g_aabb = g_pcd.get_axis_aligned_bounding_box()
    g_min = g_aabb.get_min_bound() # NOTE: this is the min bound we use throughout the code
    print(g_min)




    # ============= Create a gaussians centers as a point cloud and get its bounding box ============= 
    # g_pos = gaussians.get_xyz.detach().cpu().numpy()
    # g_pcd = o3d.geometry.PointCloud()
    # g_pcd.points = o3d.utility.Vector3dVector(g_pos)
    # g_aabb = g_pcd.get_axis_aligned_bounding_box()
    # g_aabb_len = g_aabb.get_extent() 

    # ============= Choose the voxel size as the minimum of the bounding box dimensions ============= 
    voxel_size = (np.round(g_aabb_len / np.array(dimensions), 3)).min()
    H, W, D = np.ceil(g_aabb_len / voxel_size).astype(int)
    # g_min = g_aabb.get_min_bound() # NOTE: this is the min bound we use throughout the code
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
    scale_ = np.ones((1, 3)) * 0.5
    scale_[0,2] = scale_z
    scale_ = torch.tensor(scale_, dtype=torch.float32, device="cuda")
    quat_ = torch.tensor(np.array([[1.0, 0.0, 0.0, 0.0]] * 1), dtype=torch.float32, device="cuda")
    L = build_scaling_rotation(scale_, quat_)
    actual_covariance = L @ L.transpose(1, 2)
    g_cov_6 = strip_symmetric(actual_covariance).detach().cpu().numpy()
    g_cov_diag = g_cov_6[:,[0,3,5]]
    g_cov = g_cov_6[:,[0,1,2,1,3,4,2,4,5]].reshape(-1,3,3)
    g_inv_cov = np.linalg.inv(g_cov)
    assert g_cov_diag.shape == (g_pos.shape[0], 3), "Covariance diagonal shape should be (N,3)"
    assert g_cov.shape == (g_pos.shape[0], 3, 3), "Covariance shape should be (N,3,3)"


    print(g_cov)

    # compute the min and max coordinates of aabb for each gaussian blob
    g_aabb_min = g_pos - 3 * np.sqrt(g_cov_diag)
    g_aabb_max = g_pos + 3 * np.sqrt(g_cov_diag)

    # prepare related data for voxel splatting
    g_opacity = np.ones(1)
    # # TODO:
    # g_opacity = np.ones_like(g_opacity)

    mixtures =  np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    g_ink = mixtures


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

    for g_idx in tqdm(test_idx):
    # for g_idx in tqdm(range(g_pos.shape[0])):
    

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
    assert voxel_ink[mask].sum(axis=1).shape == voxel_opacity[mask].shape, "Ink mixture sum shape should be the same as opacity shape"
    assert (trans_ink >= 0.0).all(), "The added transparent ink should be positive"
    voxel_ink[mask,5] += trans_ink

    # now the ink mixture of each voxel sum should be 1
    assert np.abs(voxel_ink.sum(axis=3) - 1.0).max() < 1e-6, "Ink mixture should sum to 1.0"



    # visualize the center x y z slice of voxel
    center_x, center_y, center_z = H // 2, W // 2, D // 2
    slice_x = voxel_ink[center_x,:,:,:]
    slice_y = voxel_ink[:,center_y,:,:]
    slice_z = voxel_ink[:,:,center_z,:]

    # we set the place where has 100% opacity to white, other places to black
    slice_x_ = np.where(slice_x[:,:,-1] == 1.0, 1.0, 0.0)
    slice_y_ = np.where(slice_y[:,:,-1] == 1.0, 1.0, 0.0)
    slice_z_ = np.where(slice_z[:,:,-1] == 1.0, 1.0, 0.0)

    # save the slices as png
    plt.imsave(os.path.join(model_path,"slice",f"{name_idx}_x.png"), slice_x_, cmap='gray')
    plt.imsave(os.path.join(model_path,"slice",f"{name_idx}_z.png"), slice_z_, cmap='gray')

    # np.save(os.path.join(model_path,"voxel_ink_post.npy"), voxel_ink)

    # assert False, "stop here"

    # visualize the voxel splatting result for debugging
    # debug_color = HELPER.mix_2_RGB_wavelength(debug_ink)
    # HELPER.debug_plot(grid_center.reshape(-1,3), debug_color, debug_opacity.reshape(-1), "voxel_splatting/voxel_splatting_alpha.png")
    # debug_alpha_one = np.ones_like(debug_opacity) - (debug_opacity == 0).astype(int)
    # HELPER.debug_plot(grid_center.reshape(-1,3), debug_color, debug_alpha_one.reshape(-1), "voxel_splatting/voxel_splatting.png")

    
    # ============= Get the albedo and sigma =============

    
    print("Computing albedo and sigma")
    # voxel_ink = np.load(os.path.join(model_path,"voxel_ink_post.npy"))

    absorption, scattering = HELPER.RGB_ink_param_2_RGB(voxel_ink)
    # t =  a + s
    sigma = scattering + absorption
    # albedo = s / t
    albedo = scattering / (sigma + 1e-8) # avoid division by zero


    # # Debug use the color as albedo and opacity as sigma
    
    # albedo = HELPER.mix_2_RGB_wavelength(debug_ink, keep_dim = True)
    # sigma = debug_opacity.reshape(H,W,D,1)
    # print("<<<<<<<<<<<< DEBUG albedo and sigma >>>>>>>>>>>>")

    # # Debug set sigma and albedo to pure yellow ink.
    # debug_mask = voxel_opacity > 0.0
    # sigma[debug_mask] = np.array([2.25, 3.75, 19.0])
    # albedo[debug_mask] = np.array([0.997, 0.995, 0.15])

    assert sigma.shape == (H, W, D,3), sigma.shape
    assert albedo.shape == (H, W, D,3), albedo.shape
    #  check if there are any nan values in sigma and albedo
    assert not np.isnan(sigma).any()
    assert not np.isnan(albedo).any()


    # ============= Save the data to vol =============
    path = os.path.join(model_path,"meta")
    if not os.path.exists(path):
        os.makedirs(path)

    print("Converting data to vol format")
    c_sigma = convert_data_to_C_indexing_style(sigma, 3, (H, W, D))
    c_albedo = convert_data_to_C_indexing_style(albedo, 3, (H, W, D))
    print("Done converting data to C indexing style")


    print("Writing to vol files")
    write_to_vol_file(os.path.join(path, f"{name_idx}_albedo.vol"), c_albedo, 3, g_min, np.array([H,W,D]), voxel_size=voxel_size)
    write_to_vol_file(os.path.join(path, f"{name_idx}_sigma.vol"), c_sigma, 3, g_min, np.array([H,W,D]), voxel_size=voxel_size)

    print("Done converting gaussians to volume representation")


    # ============= Create a mitsuba scene =============

    # We render from the vol path
    # NOTE: VERY IMPORTANT!!!!!!! the sigma_t scale factor should be 20!!!!!

    # # For debug
    # voxel_size = 0.009
    # H,W,D = (148, 258, 160)
    # grid_bbox_center = np.array([ 0.00852721, -0.00794841, 0.29117552])

    # voxel_size = 0.007
    # H,W,D = (191, 332, 205)
    # grid_bbox_center = np.array([ 0.01102721,-0.00694841, 0.28867552])

    # here we assume the scene is in 5 cm, scale = 5cm / 1mm = 50

    # mi_albedo = mi.VolumeGrid(mi.TensorXf(c_albedo.reshape(D,W,H,3)))
    # mi_sigma = mi.VolumeGrid(mi.TensorXf(c_sigma.reshape(D,W,H,3)))


    # voxel_size = 0.007
    # H,W,D = (191, 332, 205)
    # grid_bbox_center = np.array([ 0.01102721,-0.00694841, 0.28867552])

    scene_dict = get_mixing_mitsuba_scene_dict(50, 
                                            grid_bbox_center,
                                            np.array([H,W,D])*voxel_size,
                                            os.path.join(model_path,"meta",f"{name_idx}_albedo.vol"), 
                                            os.path.join(model_path,"meta",f"{name_idx}_sigma.vol"))
    # scene_dict = get_mixing_mitsuba_scene_dict(50, 
    #                                        aabb.get_center(),
    #                                        aabb.get_max_bound() - aabb.get_min_bound(),
    #                                        '3dgs_lego_train/try/color.vol', 
    #                                         '3dgs_lego_train/try/density.vol')
    
    camera_dict = get_camera_dict(viewcamera, H = 256, W = 256)
    print("Rendering scene")

    render_mitsuba_scene(scene_dict,camera_dict, 
                         np.array([H,W,D])*voxel_size, 
                         filepath =  os.path.join(model_path,"render"),
                         set_spp = 64, 
                         view_idx=name_idx,
                         save_exr = False)
    
    return 0



def merge_images(image_folder, output_image, image_size, grid_size):

    from PIL import Image
    import re

    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else 0

    image_files = sorted([img for img in os.listdir(image_folder) if img.endswith(("png"))], key = extract_number)
    image_files[0] = image_files[40]
    image_files = image_files[:40]
    print(image_files)
    images = [Image.open(os.path.join(image_folder, img)).convert('RGB') for img in image_files]

    print(images[0].size)

    # Ensure the number of images is correct
    if len(images) != grid_size[0] * grid_size[1]:
        raise ValueError("Number of images does not match the grid size")

    # Create a new image with the appropriate height and width
    merged_image = Image.new('RGB', (image_size[0] * grid_size[1], image_size[1] * grid_size[0]))

    # Paste each image into the appropriate position in the grid
    for index, image in enumerate(images):
        row = index // grid_size[1]
        col = index % grid_size[1]
        merged_image.paste(image, (col * image_size[0], row * image_size[1]))

    # Save the merged image
    merged_image.save(output_image)


if __name__ == "__main__":
    scales_z = np.linspace(0.001, 0.1, 20)
    scales_z = np.concatenate([scales_z, np.linspace(0.1, 1, 20)])
    for idx, scale_z in enumerate(scales_z):
        viewpoint_camera = get_one_blob(scale_z,idx)

        voxel_splatting([200,200,200], viewpoint_camera, "blob_thickness_test", scale_z, idx)

    mean_scale = 0.0050642
    viewpoint_camera = get_one_blob(mean_scale, len(scales_z))
    voxel_splatting([200,200,200], viewpoint_camera, "blob_thickness_test", mean_scale, len(scales_z))

    image_folder = "blob_thickness_test/rasterize" # Folder containing images
    output_image = 'blob_thickness_test/rasterize_40.jpg'     # Output image file name
    image_size = (256, 256)  # Width and height of each image
    # image_size = (640, 480)  # Width and height of each image
    # # image_size = (200, 200)  # Width and height of each image

    grid_size = (4, 10)  # Rows, Columns

    merge_images(image_folder, output_image, image_size, grid_size)


