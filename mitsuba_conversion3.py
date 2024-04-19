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

    def g_center_RGB(self, mix):
        INK = Ink()
        assert mix.shape[1] == 6, "Ink mixture should have 6 channels"
        assert (mix >= 0.0).all(), "Ink mixture should be positive"
        assert (mix <= 1.0 + 1e-1).all(), "Ink mixture should be less than 1.0. {} > 1.0".format(mix.max())
        N, C = mix.shape
        mix =  torch.tensor(mix, dtype = torch.float , device="cuda")
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
                print(temp[mask[0][0]])
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




    
HELPER = Helper()


def one_voxel_one_gaussian(gaussians:GaussianModel,dimensions: tuple):
    '''
        The steps are as follows:
        1. Build a bounding box around the gaussian blobs
        2. Create a voxel grid that covers the bounding box
        3. Create a KDTree for the voxel centers
        4. For each gaussian blob, find the 3-sigma voxel centers
        5. add corresponding ink mixture * gaussian opacity to the voxel centers

    '''
    assert len(dimensions) == 3, "Dimensions must be a 3-tuple"


    # ====================== Build bounding box and voxel grid =======================
    g_pos = gaussians.get_xyz.detach().cpu().numpy()
    g_pcd = o3d.geometry.PointCloud()
    g_pcd.points = o3d.utility.Vector3dVector(g_pos)
    g_aabb = g_pcd.get_axis_aligned_bounding_box()
    g_aabb_len = g_aabb.get_extent() 
    # NOTE: TODO: here we specify the precision of the voxel grid is 0.0x. Should be more elegant.
    voxel_size = (np.round(g_aabb_len / np.array(dimensions), 2)).min()
    # we add one layer of voxels to make sure the bounding box is covered
    v_min = g_aabb.get_min_bound() - voxel_size
    v_max = g_aabb.get_max_bound() + voxel_size
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(g_pcd, voxel_size, v_min, v_max)


    aabb = voxel_grid.get_axis_aligned_bounding_box()
    assert (g_aabb.get_min_bound() - aabb.get_min_bound() >= 0).all(), " guassian bbox min should bigger than voxel grid min"
    assert (aabb.get_max_bound() - g_aabb.get_max_bound() >= 0).all(), " guassian bbox max should smaller than voxel grid max"

    voxels_idx = [v.grid_index for v in voxel_grid.get_voxels()]
    vc_pos = np.array([voxel_grid.get_voxel_center_coordinate(i) for i in voxels_idx])
    assert len(vc_pos) == len(voxels_idx), "voxel centers and indices should have the same length"

    print(aabb.get_extent())
    

    # ====================== one voxel one gaussian =======================

    ink_mix_voxel = np.zeros((vc_pos.shape[0], 6)) # the index of ink_mix_voxel is the same as vc_pos
    voxel_opacity = np.zeros(vc_pos.shape[0]) # all False

    g_pos = gaussians.get_xyz.detach().cpu().numpy()
    g_cov = gaussians.get_actual_covariance().detach().cpu().numpy()
    g_opacities = gaussians.get_opacity.detach().cpu().numpy()
    g_ink_mix = gaussians.get_ink_mix.detach().cpu().numpy()

    # post process: if the opacity is 0, then the ink mixture should be substituted with 100% transparent ink
    transparent_gaussians = np.where(g_opacities == 0)[0]
    g_ink_mix[transparent_gaussians] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # 100% transparent ink

    assert (g_ink_mix >= 0.0).all(), "gaussian ink mixture should be positive, there might be bug in optimization"

    g_pcd_tree = o3d.geometry.KDTreeFlann(g_pcd)

    print("total number of iterations: ", vc_pos.shape[0])

    with tqdm(total=vc_pos.shape[0]) as pbar:

        for vc_idx, vc in enumerate(vc_pos):
            [k, g_neighbor_idx, squared_distances] = g_pcd_tree.search_knn_vector_3d(vc, 20)

            g_neighbor_pos = g_pos[g_neighbor_idx]
            # get the 3 sigma index of the query_idx
            g_result_idx, maha_dists = get_3_sgima_idx(vc, g_cov[g_neighbor_idx], g_neighbor_idx, g_neighbor_pos)
            if g_result_idx is None:
                # No gaussian is close, assign 100% transparent ink
                ink_mix_voxel[vc_idx] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

                # RuntimeError("Should decrease voxel size. No voxel center is within 3 sigma of the gaussian blob")
            else:
                # g_voxel_counter[vc_result_idx] += 1
                probs = np.exp(-0.5 * maha_dists)
                max_prob = np.max(probs)
                max_prob_idx = np.argmax(probs)
                max_prob_g_idx = g_result_idx[max_prob_idx]
                ink_mix_voxel[vc_idx] =  (g_opacities[max_prob_g_idx] * max_prob) * g_ink_mix[max_prob_g_idx]
                voxel_opacity[vc_idx] = g_opacities[max_prob_g_idx]

                # assert probs.shape == vc_result_idx.shape, "probs and vc_result_idx should have the same shape {} != {}".format(probs.shape, vc_result_idx.shape)
                # assert (probs >= 0.0).all() and np.isnan(probs).any() == False, "probs should be positive and not nan"
                # ink_mix_voxel[vc_result_idx] +=  (g_opacities[g_idx] * probs)[:, np.newaxis] * g_ink_mix[g_idx]
            pbar.update(1)
        
    # print(g_voxel_counter.max())
    # return
    # transparent_voxels = np.where(g_voxel_counter == 0)[0]
    # ink_mix_voxel[transparent_voxels] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # 100% transparent ink
    # g_voxel_counter[transparent_voxels] = 1

    # ====================== Debug visualize =======================
    debug_color = HELPER.g_center_RGB(ink_mix_voxel) # TODO: not sure if this is reasonable
    debug_alpha = np.ones(debug_color.shape[0])
    HELPER.debug_plot(vc_pos, debug_color, voxel_opacity, "voxel_splatting.png")



    return 0





def get_3_sgima_idx(reference_pos, g_cov, neighbor_idxs, queries_pos):
    
    # get the 3 sigma index of the query_idx
    inv_cov = np.linalg.inv(g_cov)
    diff = queries_pos - reference_pos
    # print(diff.shape)
    # print(inv_cov.shape)
    # print(np.isnan(inv_cov).any(), "Inverse covariance should not have nan")
    # print(np.dot(diff, inv_cov).shape)
    # print((np.array([diff[i, :].T @ inv_cov[i, :, :] @ diff[i, :] for i in range(diff.shape[0])]).shape))
    # assert False, "stop here"

    maha_distance = np.sqrt(np.array([diff[i, :].T @ inv_cov[i, :, :] @ diff[i, :] for i in range(diff.shape[0])]))
    assert np.isnan(maha_distance).any() == False, "Mahalanobis distance should not have nan"
    in_3_sigma = np.where(maha_distance <= 3)[0]
    if len(in_3_sigma) == 0:
        return None, None
    else:
        neighbor_idxs = np.array(neighbor_idxs)
        return neighbor_idxs[in_3_sigma], maha_distance[in_3_sigma]



def mitsuba_gaussians(dataset : ModelParams, iteration : int, pipeline : PipelineParams, readvol=False):
    voxel_size = 0.05
    with torch.no_grad():
        gaussians = GaussianModel(0)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        one_voxel_one_gaussian(gaussians, [600,600,600])


if __name__ == "__main__":

    '''
        python mitsuba_conversion2.py -m 3dgs_lego_train -w --iteration 3000 --sh_degree 0
        python mitsuba_conversion2.py -m 3dgs_lego_train --iteration 3000 --sh_degree 0
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
