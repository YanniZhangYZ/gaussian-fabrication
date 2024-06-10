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


class Helper:
    def __init__(self):
        super(Helper, self).__init__()

    def debug_plot(self, pos, debug_color, debug_alpha, path):
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

    def debug_histogram(self, opacity, path):
        print("Ploting histogram")
        opacity = opacity.reshape(-1)
        import matplotlib.pyplot as plt
        plt.hist(opacity, bins=100)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(opacity, bins=100,)
        bins = np.linspace(np.min(opacity), np.max(opacity), 31)  # 100 bins
        counts, bin_edges, patches = ax.hist(opacity, bins=bins)
        ax.set_xticks(bin_edges)
        ax.set_yticks(range(0, int(max(counts))+1, max(1, int(max(counts)/10))))
        plt.xticks(rotation=45)
        ax.set_title('Histogram of Opacity')
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        plt.savefig(path)

    def mix_2_RGB_wavelength(self, mix, keep_dim = False):
        INK = Ink(use_torch = False)
        assert mix.shape[-1] == 6, "Ink mixture should have 6 channels"
        assert (mix >= 0.0).all(), "Ink mixture should be positive"
        assert (mix <= 1.0 + 1e-1).all(), "Ink mixture should be less than 1.0. {} > 1.0".format(mix.max())

        H,W,D,C = mix.shape
        mix = mix.reshape(-1,6)
        N, C = mix.shape
        # K
        mix_K = mix @ INK.absorption_matrix
        # S
        mix_S = mix @ INK.scattering_matrix + 1e-8

        #equation 2
        R_mix = 1 + mix_K / mix_S - np.sqrt( (mix_K / mix_S)**2 + 2 * mix_K / mix_S)

        # the new R_mix contains 3 parts: the original R_mix, the mean of the original R_mix, and zeros
        R_mix = np.concatenate([R_mix, (R_mix[:,1:] + R_mix[:,:-1]) / 2], axis=1)
        R_mix = np.concatenate([R_mix, np.zeros((R_mix.shape[0], INK.w_num - R_mix.shape[1]))], axis=1)
        assert (R_mix[:,71] == 0.0).all(), "The 71th element should be 0.0"

        
        if np.isnan(np.sqrt( (mix_K / mix_S)**2 + 2 * mix_K / mix_S)).any():
                temp = (mix_K / mix_S)**2 + 2 * mix_K / mix_S
                mask = np.nonzero(np.isnan(np.sqrt( temp)))
                # print(R_mix.shape)
                print(mask)
                print(temp[mask[0][0]])
                assert False, "sqrt negative value has nan"

       
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

        XYZ = np.stack([X,Y,Z],axis=1).T
    

        # Convert XYZ to sRGB, Equation 7
        sRGB_matrix = np.array([[3.2406, -1.5372, -0.4986],
                                [-0.9689, 1.8758, 0.0415],
                                [0.0557, -0.2040, 1.0570]])
        sRGB = (sRGB_matrix @ XYZ).T

        # Apply gamma correction to convert linear RGB to sRGB
        mask = sRGB > 0.0031308
        sRGB[~mask] = sRGB[~mask] * 12.92
        sRGB[mask] = 1.055 * np.power(sRGB[mask], 1 / 2.4) - 0.055
        # sRGB = np.where(sRGB <= 0.0031308,
        #               12.92 * sRGB,
        #               1.055 * np.power(sRGB, 1 / 2.4) - 0.055)
        sRGB = np.clip(sRGB,0.0,1.0)
        assert sRGB.shape == (N, 3), "sRGB shape should be (N,3)"
        if keep_dim:
            return sRGB.reshape(H,W,D,3)
        return sRGB
        # sRGB = torch.clip(sRGB,0,1).view(H, W, 3).permute(2,0,1)

    def add_transparency(self, mix):
        '''
        input shape: (H,W,D,6)
        output shape: (H,W,D,6)
        '''


        H, W, D, C = mix.shape
        # check the voxel whose ink mixture sum is smaller than 1
        mix_sum =  np.sum(mix, axis=3)
        assert mix_sum.shape == (H, W, D), "mix_sum shape should be (H, W, D)"
        add_trans_idx = np.argwhere(mix_sum < 1)
        if add_trans_idx.shape[0] > 0:
            trans_concentration =  1.0 - mix_sum[add_trans_idx[:,0], add_trans_idx[:,1], add_trans_idx[:,2]]
            assert (trans_concentration > 0.0).all() , " should be bigger than 0.0"
            mix[add_trans_idx[:,0], add_trans_idx[:,1], add_trans_idx[:,2],5] += trans_concentration

        # normalize the ink mixture so that for every voxel the mixture sum is 1
        mix = mix / mix.sum(axis=3, keepdims=True)
        assert (np.abs(mix.sum(axis=3) - 1.0) < 1e-6).all(), "Ink mixture should sum to 1.0"
        # print(INK.absorption_RGB.shape, INK.scattering_RGB.shape) # (6, 3)
        return mix
    

    # def mix_2_RGB_RGB(self, mix):
    #     absoprtion, scattering = self.RGB_ink_param_2_RGB(mix)

    #     I0 = np.array([1.0, 1.0, 1.0]) # assuming initial light is white
    #     d = 5.0 # Distance light travels through the medium



    def RGB_ink_param_2_RGB(self,mix):
        '''
        input shape: (H,W,D,6)
        output shape: (H,W,D,3)
        
        '''
        INK = Ink(use_torch = False)

        assert mix.shape[-1] == 6, "Ink mixture should have 6 channels"

        mix_K = mix @ INK.absorption_RGB
        mix_S = mix @ INK.scattering_RGB

        assert (mix_K >= 0.0).all() and (mix_S >= 0.0).all(), "albedo and scattering should be positive"
       
        return mix_K, mix_S


    
HELPER = Helper()


def voxel_splatting( gaussians: GaussianModel, dimensions: tuple, viewcameras: list, model_path:str):
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
    g_opacity = gaussians.get_opacity.detach().cpu().numpy()
    # # TODO:
    # g_opacity = np.ones_like(g_opacity)

    g_ink = gaussians.get_ink_mix.detach().cpu().numpy()


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

    # np.save(os.path.join(model_path,"voxel_ink_post.npy"), voxel_ink)


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
    plt.imsave(os.path.join(model_path,"result_imgs/slice_x.png"), slice_x_, cmap='gray')
    plt.imsave(os.path.join(model_path,"result_imgs/slice_y.png"), slice_y_, cmap='gray')
    plt.imsave(os.path.join(model_path,"result_imgs/slice_z.png"), slice_z_, cmap='gray')



    # we set the value to the percentatge of transparent ink (the last channel of ink mixture)
    #  it should be computed as the the transparent ink divided by the sum of the ink mixture
    slice_x_ = slice_x[:,:,-1] / slice_x.sum(axis=2)
    slice_y_ = slice_y[:,:,-1] / slice_y.sum(axis=2)
    slice_z_ = slice_z[:,:,-1] / slice_z.sum(axis=2)

    # save the slices as png
    plt.imsave(os.path.join(model_path,"result_imgs/percent_slice_x.png"), slice_x_, cmap='viridis')
    plt.imsave(os.path.join(model_path,"result_imgs/percent_slice_y.png"), slice_y_, cmap='viridis')
    plt.imsave(os.path.join(model_path,"result_imgs/percent_slice_z.png"), slice_z_, cmap='viridis')
    
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
    path = os.path.join(model_path,"mitsuba")
    if not os.path.exists(path):
        os.makedirs(path)

    print("Converting data to vol format")
    c_sigma = convert_data_to_C_indexing_style(sigma, 3, (H, W, D))
    c_albedo = convert_data_to_C_indexing_style(albedo, 3, (H, W, D))
    print("Done converting data to C indexing style")


    print("Writing to vol files")
    write_to_vol_file(os.path.join(path, "albedo.vol"), c_albedo, 3, g_min, np.array([H,W,D]), voxel_size=voxel_size)
    write_to_vol_file(os.path.join(path, "sigma.vol"), c_sigma, 3, g_min, np.array([H,W,D]), voxel_size=voxel_size)

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
                                            os.path.join(model_path,"mitsuba","albedo.vol"), 
                                            os.path.join(model_path,"mitsuba","sigma.vol"))
    # scene_dict = get_mixing_mitsuba_scene_dict(50, 
    #                                        aabb.get_center(),
    #                                        aabb.get_max_bound() - aabb.get_min_bound(),
    #                                        '3dgs_lego_train/try/color.vol', 
    #                                         '3dgs_lego_train/try/density.vol')
    
    camera_dict = get_camera_dict(viewcameras[69])
    # camera_dict = get_camera_dict(viewcameras[0])

    # ================Rendering scene================
    print("Rendering scene")

    render_mitsuba_scene(scene_dict,camera_dict, np.array([H,W,D])*voxel_size, filepath =  os.path.join(model_path,"mitsuba","render"),set_spp = 64, view_idx=0)

    
    camera_dict = get_camera_dict(viewcameras[73])
    print("Rendering scene")

    render_mitsuba_scene(scene_dict,camera_dict, np.array([H,W,D])*voxel_size, filepath =  os.path.join(model_path,"mitsuba","render"),set_spp = 64, view_idx=1)

    camera_dict = get_camera_dict(viewcameras[64])
    print("Rendering scene")

    render_mitsuba_scene(scene_dict,camera_dict, np.array([H,W,D])*voxel_size, filepath =  os.path.join(model_path,"mitsuba","render"),set_spp = 64, view_idx=2)

    
    
    return 0


def mitsuba_gaussians(dataset : ModelParams, iteration : int, pipeline : PipelineParams, readvol=False):
    with torch.no_grad():
        gaussians = GaussianModel(0)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        cameras = scene.getTrainCameras()

        g_opacity = gaussians.get_opacity.detach().cpu().numpy()
        HELPER.debug_histogram(g_opacity, os.path.join(dataset.model_path,"mitsuba","render","opacity_histogram_g_blob.png"))
        voxel_splatting(gaussians,[200,200,200], cameras, dataset.model_path)







if __name__ == "__main__":

    '''
        python voxel_splatting.py -m 3dgs_lego_train -w --iteration 3000 --sh_degree 0
        python voxel_splatting.py -m 3dgs_lego_train --iteration 3000 --sh_degree 0
        python voxel_splatting.py -m 3dgs_pattern_train --iteration 3000 --sh_degree 0
        python voxel_splatting.py -m 3dgs_lego_train_albedo --iteration 3000 --sh_degree 0

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
