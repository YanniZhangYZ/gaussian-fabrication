# This file takes as input a .ply file, converting it to a volume representation, and rendering it using Mitsuba3.
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

from mitsuba_utils import get_mixing_mitsuba_scene_dict, render_mitsuba_scene, write_to_vol_file, convert_data_to_C_indexing_style, get_camera_dict

INK = Ink()

def mitsuba_gaussians(dataset : ModelParams, iteration : int, pipeline : PipelineParams, readvol=False):
    voxel_size = 0.05
    with torch.no_grad():
        gaussians = GaussianModel(0)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        debug_plot_g( gaussians)
        g_center_color = g_center_RGB(gaussians)
        

        # camera parameters
        views = scene.getTrainCameras()
        view_idx = 26
        viewpoint_camera = views[view_idx]
        image_height=int(viewpoint_camera.image_height)
        image_width=int(viewpoint_camera.image_width)
        
        fy = fov2focal(viewpoint_camera.FoVy, image_height)
        fx = fov2focal(viewpoint_camera.FoVx, image_width)
        viewmat = viewpoint_camera.world_view_transform.T
        cx = image_height/2
        cy = image_width/2

        # just to check if the ink mixture are correct
        # print(gaussians.get_ink_mix[0])
        # print(gaussians.get_ink_mix[5000])
        centers = gaussians.get_xyz.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers) # The idx is the same as the gaussian_blobs
        # pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        # pcd =  o3d.io.read_point_cloud(os.path.join(dataset.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(iteration),
        #                                                    "point_cloud.ply"))
        pcd.colors = o3d.utility.Vector3dVector(g_center_color)
        aabb = pcd.get_axis_aligned_bounding_box()
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=0.01)
        # o3d.visualization.draw_geometries([voxel_grid])

        voxels = voxel_grid.get_voxels()
        colors = np.asarray([v.color for v in voxels])
        idx = np.asarray([v.grid_index for v in voxels])
        pos = np.asarray([voxel_grid.get_voxel_center_coordinate(i) for i in idx])

        

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Voxel Data')
        print("Ploting voxel representation")
        # debug_alpha = ~np.all(colors < 0.1, axis=1)
        debug_alpha = np.ones(colors.shape[0])
        # debug_alpha = gaussians.get_opacity.detach().cpu().numpy()
        assert debug_alpha.shape == (pos.shape[0],), (debug_alpha.shape, pos.shape, gaussians.get_xyz.shape)
        rgba = np.concatenate((colors.reshape(-1, 3), debug_alpha.reshape(-1, 1)), axis=1)
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=rgba, s=1.0)
        ax.view_init(elev=0, azim=45)
        plt.savefig("open3d_voxel_data.png")


        print("I put colors inside!: ", g_center_color.shape)
        debug_plot_g( gaussians, filename = "gaussian_data_RGB.png",center_RGB = g_center_color, use_alpha=True)



        # return 0
        
        if not readvol:
            gaussians2vol(pcd, gaussians, voxel_size, dataset.model_path)
        
        # We render from the vol path
        scene_dict = get_mixing_mitsuba_scene_dict(10, 
                                               aabb.get_center(),
                                               aabb.get_max_bound() - aabb.get_min_bound(),
                                               os.path.join(dataset.model_path,"mitsuba","albedo.vol"), 
                                                os.path.join(dataset.model_path,"mitsuba","sigma.vol"))
        # scene_dict = get_mixing_mitsuba_scene_dict(50, 
        #                                        aabb.get_center(),
        #                                        aabb.get_max_bound() - aabb.get_min_bound(),
        #                                        '3dgs_lego_train/try/color.vol', 
        #                                         '3dgs_lego_train/try/density.vol')
        camera_dict = get_camera_dict(viewpoint_camera)
        
        print()
        print("================Rendering scene================")

        render_mitsuba_scene(scene_dict,camera_dict, aabb.get_max_bound() - aabb.get_min_bound(), filepath =  os.path.join(dataset.model_path,"mitsuba","render"),set_spp = 16, view_idx=0)


def g_center_RGB(gaussians):
    # c0 = gaussians._features_dc.detach().cpu().numpy()
    # SH_C0 = 0.28209479177387814
    # color = SH_C0 * c0
    # color += 0.5
    # color =  np.clip(color, 0.0, 1.0).reshape(-1, 3)
    # assert color.shape == (gaussians.get_xyz.shape[0], 3), color.shape
    # return color
    mix = gaussians.get_ink_mix * gaussians.get_opacity
    N, C = mix.shape
     # K
    mix_K = mix @ INK.absorption_matrix
    # S
    mix_S = mix @ INK.scattering_matrix + 1e-8

    #equation 2
    R_mix = 1 + mix_K / mix_S - torch.sqrt( (mix_K / mix_S)**2 + 2 * mix_K / mix_S)

    if torch.isnan(torch.sqrt( (mix_K / mix_S)**2 + 2 * mix_K / mix_S)).any():
            temp = (mix_K / mix_S)**2 + 2 * mix_K / mix_S
            mask = torch.nonzero(torch.isnan(torch.sqrt( temp)))
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
    return sRGB.detach().cpu().numpy()
    # sRGB = torch.clip(sRGB,0,1).view(H, W, 3).permute(2,0,1)



        
def gaussians2vol(pcd: o3d.geometry.PointCloud, gaussians: GaussianModel, voxel_size: float, model_path: str):
    print("================Converting gaussians to volume representation================")
    aabb = pcd.get_axis_aligned_bounding_box()
    dimensions = np.ceil((aabb.get_max_bound() - aabb.get_min_bound()) / voxel_size).astype(int) + 1    
    bbox_center = aabb.get_center()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    print("Voxel grid dimensions: ", dimensions)
    print("Voxel grid center: ", bbox_center)

    # # ink mixtures for each grid
    # closest_points_idx, prob_matrix, min_dist_matrix, mix_3d, grid = get_mix_3d(dimensions,voxel_size,min_bound,gaussians, pcd)
    # # albedo and sigma o each grid
    # # absorption, scattering = wavelength2RGB(mix_3d)
    # absorption, scattering = RGB_ink_param_2_RGB(mix_3d)
    # # t =  a + s
    # sigma = scattering + absorption
    # # albedo = s / t
    # albedo = scattering / (sigma + 1e-8) # avoid division by zero

    # sigma = sigma.detach().cpu().numpy()
    # albedo = albedo.detach().cpu().numpy()

    closest_points_idx, prob_matrix, min_dist_matrix, albedo, sigma, grid = get_color_3d(dimensions,voxel_size,min_bound,gaussians, pcd)




    assert grid.shape == (dimensions[0], dimensions[1], dimensions[2], 3)
    assert sigma.shape == (dimensions[0], dimensions[1], dimensions[2],3), sigma.shape
    assert albedo.shape == (dimensions[0], dimensions[1], dimensions[2],3), albedo.shape
    #  check if there are any nan values in sigma and albedo
    assert not np.isnan(sigma).any()
    assert not np.isnan(albedo).any()

    path = os.path.join(model_path,"mitsuba")
    if not os.path.exists(path):
        os.makedirs(path)
    
    np.save(os.path.join(path, "grid_pos.npy"), grid)
    np.save(os.path.join(path, "sigma.npy"), sigma)
    np.save(os.path.join(path, "albedo.npy"), albedo)


    c_sigma = convert_data_to_C_indexing_style(sigma, 3, dimensions)
    c_albedo = convert_data_to_C_indexing_style(albedo, 3, dimensions)
    
    print("Writing to vol files")
    write_to_vol_file(os.path.join(path, "albedo.vol"), c_albedo, 3, min_bound, dimensions, voxel_size=voxel_size)
    write_to_vol_file(os.path.join(path, "sigma.vol"), c_sigma, 3, min_bound, dimensions, voxel_size=voxel_size)

    print("Done converting gaussians to volume representation")


def get_color_3d(dimensions, voxel_size, min_bounds, gaussians, pcd):
    print("hello")
    H, W, D= dimensions[0], dimensions[1], dimensions[2]
    grid = np.empty((H, W, D, 3), dtype=float)

    g_mean = gaussians.get_xyz.detach().cpu().numpy() # (num_points, 3)
    g_cov = gaussians.get_actual_covariance().detach().cpu().numpy() # (num_points, 3, 3)
    g_color = np.array(pcd.colors).reshape(-1, 3)
    g_opacity = gaussians.get_opacity.detach().cpu().numpy() # (num_points, 1)
    # inv_cov = np.linalg.inv(g_cov)

    closest_points_idx = np.empty((H, W, D), dtype=int)
    prob_matrix = np.empty((H, W, D), dtype=float)
    min_dist_matrix = np.empty((H, W, D), dtype=float)
    mix_albedo= np.empty((H, W, D,3), dtype=float)
    mix_sigma= np.empty((H, W, D,3), dtype=float)

    # mix_3d = np.empty((6, H, W, D), dtype=float)
    # debug_color = np.empty((H, W, D, 3), dtype=float)
    # debug_alpha = np.empty((H, W, D), dtype=float)

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    from scipy.spatial.distance import mahalanobis
    with tqdm(total=H*W*D) as pbar:
        for i in range(H):
            for j in range(W):
                for k in range(D):
                    point = np.array([i, j, k]) * voxel_size + min_bounds
                    grid[i, j, k] = point
                    _, idx, _ = pcd_tree.search_knn_vector_3d(np.array([i, j, k])* voxel_size + min_bounds, 20)
                    # mahalanobis distance for the 10 closest points
                    distances = np.empty(len(idx))
                    for k,m in enumerate(idx):
                        inv_cov = np.linalg.inv(g_cov[m])
                        dist = mahalanobis(point, g_mean[m], inv_cov)
                        distances[k] = dist
                    # distances = np.array([np.sqrt((point - g_mean[m]).T @ np.linalg.inv(g_cov[m]) @ (point - g_mean[m])) for m in idx])
                    # print(distances)
                    min_idx = idx[np.argmin(distances)]
                    min_dist = np.min(distances)
                    # print(distances.shape, min_idx, min_dist)
                    if min_dist > 5: # larger than 3-sigma distance
                        closest_points_idx[i, j, k] = -1
                        prob_matrix[i, j, k] = 0.0
                        mix_albedo[i, j, k] = np.array([1.0,1.0,1.0]) # set as transparent
                        mix_sigma[i, j, k] = np.array([1e-4, 1e-4, 1e-4]) # set as transparent
                        min_dist_matrix[i, j, k] = -1
                        # debug_color[i, j, k] = np.array([1.0, 1.0, 1.0])
                        # debug_alpha[i, j, k] = 0.0
                    else:
                        closest_points_idx[i, j, k] = min_idx
                        prob_matrix[i, j, k] = np.exp(-0.5 * min_dist)
                        mix_albedo[i, j, k] = g_color[min_idx]
                        mix_sigma[i, j, k] = g_opacity[min_idx]

                        min_dist_matrix[i, j, k] = min_dist
                        # debug_color[i, j, k] = np.array([1.0, 0.0, 0.0])
                        # debug_alpha[i, j, k] = 1.0
                    pbar.update(1)

    # print("writing debug plot...")
    # np.save("debug_color.npy", debug_color)
    # np.save("debug_alpha.npy", debug_alpha)
    # debug_plot(grid, debug_color, debug_alpha, "voxel_data.png")

    # print("Done getting ink mixtures")

    return closest_points_idx, prob_matrix, min_dist_matrix, mix_albedo, mix_sigma, grid




def get_mix_3d(dimensions, voxel_size, min_bounds, gaussians, pcd):
    print("hello")
    H, W, D= dimensions[0], dimensions[1], dimensions[2]
    grid = np.empty((H, W, D, 3), dtype=float)

    g_mean = gaussians.get_xyz.detach().cpu().numpy() # (num_points, 3)
    g_cov = gaussians.get_actual_covariance().detach().cpu().numpy() # (num_points, 3, 3)
    g_ink_mix = gaussians.get_ink_mix.detach().cpu().numpy() # (num_points, 6)
    g_opacity = gaussians.get_opacity.detach().cpu().numpy() # (num_points, 1)
    # inv_cov = np.linalg.inv(g_cov)

    closest_points_idx = np.empty((H, W, D), dtype=int)
    prob_matrix = np.empty((H, W, D), dtype=float)
    min_dist_matrix = np.empty((H, W, D), dtype=float)
    mix_3d = np.empty((6, H, W, D), dtype=float)
    debug_color = np.empty((H, W, D, 3), dtype=float)
    debug_alpha = np.empty((H, W, D), dtype=float)

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    with tqdm(total=H*W*D) as pbar:
        for i in range(H):
            for j in range(W):
                for k in range(D):
                    point = np.array([i, j, k])* voxel_size + min_bounds
                    grid[i, j, k] = point
                    _, idx, _ = pcd_tree.search_knn_vector_3d(point, 20)
                    # mahalanobis distance for the 10 closest points
                    distances = np.array([np.sqrt((point - g_mean[m]).T @ np.linalg.inv(g_cov[m]) @ (point - g_mean[m])) for m in idx])
                    min_idx = idx[np.argmin(distances)]
                    min_dist = np.min(distances)
                    # print(distances.shape, min_idx, min_dist)
                    if min_dist > 2: # larger than 3-sigma distance
                        closest_points_idx[i, j, k] = -1
                        prob_matrix[i, j, k] = 0.0
                        mix_3d[:, i, j, k] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # set as transparent
                        min_dist_matrix[i, j, k] = -1
                        debug_color[i, j, k] = np.array([1.0, 1.0, 1.0])
                        debug_alpha[i, j, k] = 0.0
                    else:
                        closest_points_idx[i, j, k] = min_idx
                        prob_matrix[i, j, k] = np.exp(-0.5 * min_dist)
                        # mix_3d[:, i, j, k] = g_ink_mix[min_idx] * g_opacity[min_idx] * prob_matrix[i, j, k]
                        # NOTE:NOTE:NOTE:NOTE:NOTE:NOTE:NOTE: CHANGE!
                        mix_3d[:, i, j, k] = g_ink_mix[min_idx]

                        min_dist_matrix[i, j, k] = min_dist
                        debug_color[i, j, k] = np.array([1.0, 0.0, 0.0])
                        debug_alpha[i, j, k] = 1.0
                    pbar.update(1)

    print("writing debug plot...")
    np.save("debug_color.npy", debug_color)
    np.save("debug_alpha.npy", debug_alpha)
    debug_plot(grid, debug_color, debug_alpha, "voxel_data.png")

    print("Done getting ink mixtures")

    return closest_points_idx, prob_matrix, min_dist_matrix, mix_3d, grid


def debug_plot(grid, debug_color, debug_alpha, path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Voxel Data')
    print("Ploting voxel representation")
    rgba = np.concatenate((debug_color.reshape(-1, 3), debug_alpha.reshape(-1, 1)), axis=1)
    ax.scatter(grid[:, :, :,0], grid[:,:,:,1], grid[:,:,:,2], c=rgba, s=1.0)
    plt.savefig(path)

def debug_plot_g( gaussians, filename = "gaussian_data.png",center_RGB = None, use_alpha=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Voxel Data')
    print("Ploting voxel representation")
    grid = gaussians.get_xyz.detach().cpu().numpy()
    if center_RGB is not None:
        debug_color = center_RGB
    else:
        debug_color = np.zeros_like(grid) 
        debug_color[:] = np.array([1.0, 0.0, 0.0])
    if not use_alpha:
        debug_alpha = np.ones(grid.shape[0])
    else:
        debug_alpha = gaussians.get_opacity.detach().cpu().numpy().reshape(-1)

    rgba = np.concatenate((debug_color.reshape(-1, 3), debug_alpha.reshape(-1, 1)), axis=1)
    ax.scatter(grid[:,0], grid[:,1], grid[:,2], c=rgba, s=1.0)
    plt.savefig(filename)


def RGB_ink_param_2_RGB(mix_3d):
    '''
    input shape: (6, H, W, D)
    output shape: (H, W, D,3)
    
    '''
    C, H, W, D = mix_3d.shape
    print(INK.absorption_RGB.shape, INK.scattering_RGB.shape) # (6, 3)
    mix_K = mix_3d.transpose(1, 2, 3, 0) @ INK.absorption_RGB
    mix_S = mix_3d.transpose(1, 2, 3, 0) @ INK.scattering_RGB
    
    return mix_K, mix_S


# TODO: seems not correct
def wavelength2RGB(mix_3d):
    # NOTE: wavelength is in nm! sigma and albedo should be in mm-1
    C, H, W, D = mix_3d.shape
    mix_3d = torch.tensor(mix_3d, dtype = torch.float, device = 'cuda').permute(1, 2, 3, 0).reshape(-1, C)

    # mix: (H*W*D,6) array of ink mixtures

    # NOTE: unlike ink mixture to RGB color, there are k/s involved and we need add 1e-8 to avoid division by zero
    # But here we only care the conversion to RGB absorption and scattering
    # K
    mix_K = mix_3d @ INK.absorption_matrix
    # S
    mix_S = mix_3d @ INK.scattering_matrix 

    # equation 3 - 5
    x_D56 = INK.x_observer * INK.sampled_illuminant_D65
    # x_D56 /= np.sum(x_D56)
    y_D56 = INK.y_observer * INK.sampled_illuminant_D65
    # y_D56 /= np.sum(y_D56)
    z_D56 = INK.z_observer * INK.sampled_illuminant_D65
    
    Y_D56 = torch.sum(y_D56)


    # ----- Compute K in RGB space -----
   
    # z_D56 /= np.sum(z_D56)
    k_X = mix_K @ x_D56
    k_Y = mix_K @ y_D56
    k_Z = mix_K @ z_D56
    k_XYZ = torch.stack([k_X,k_Y,k_Z],axis=1).T

    # ----- Compute S in RGB space -----
    s_X = mix_S @ x_D56
    s_Y = mix_S @ y_D56
    s_Z = mix_S @ z_D56
    s_XYZ = torch.stack([s_X,s_Y,s_Z],axis=1).T


    sRGB_matrix = torch.tensor([[3.2406, -1.5372, -0.4986],
                                [-0.9689, 1.8758, 0.0415],
                                [0.0557, -0.2040, 1.0570]], device="cuda")
    
    k_sRGB = ((sRGB_matrix @ k_XYZ) / Y_D56).T
    # k_sRGB = torch.clip(k_sRGB,0,1).view(H, W, D, 3)
    k_sRGB = k_sRGB.view(H, W, D, 3)


    s_sRGB = ((sRGB_matrix @ s_XYZ) / Y_D56).T
    # s_sRGB = torch.clip(s_sRGB,0,1).view(H, W, D, 3)
    s_sRGB = s_sRGB.view(H, W, D, 3)


    return k_sRGB, s_sRGB



def test_wavelength2RGB():
    mix_3d = np.zeros((6, 100, 100, 100))
    mix_3d[4,:,:,:] = 1.0
    print(mix_3d[:, 0, 0, 0])
    # # albedo and sigma o each grid
    # absorption, scattering = wavelength2RGB(mix_3d)
    # # t =  a + s
    # sigma = scattering + absorption
    # # albedo = s / t
    # albedo = scattering / (sigma + 1e-8) # avoid division by zero

    # print("albedo", albedo[0, 0, 0])
    # print("sigma", sigma[0, 0, 0])

    
    absorption, scattering = RGB_ink_param_2_RGB(mix_3d)
    # t =  a + s
    sigma = scattering + absorption
    # albedo = s / t
    albedo = scattering / (sigma + 1e-8) # avoid division by zero
    print(sigma.shape, albedo.shape)

    print("albedo", albedo[0, 0, 0])
    print("sigma", sigma[0, 0, 0])


    return 0


if __name__ == "__main__":

    '''
        python mitsuba_conversion.py -m 3dgs_lego_train -w --iteration 2999 --sh_degree 0
        python mitsuba_conversion.py -m 3dgs_lego_train --iteration 2999 --sh_degree 0
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
    # mitsuba_gaussians(model.extract(args), args.iteration, pipeline.extract(args))
