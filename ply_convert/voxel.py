import numpy as np
from tqdm import tqdm
import math
import open3d as o3d
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from guassian_utils import GaussianData, Gaussian
from scipy.spatial.distance import mahalanobis
from plot_utils import plot_gaussian_ellipsoid_bbx, plot_gaussian_scene_bbox, plot_voxel_data


def load_ply(path):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # pass activate function
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    opacities = 1/(1 + np.exp(- opacities))  # sigmoid
    opacities = opacities.astype(np.float32)
    # shs = np.concatenate([features_dc.reshape(-1, 3),
    #                     features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    # shs = shs.astype(np.float32)
    shs = features_dc.reshape(-1, 3).astype(np.float32)
    return GaussianData(xyz, rots, scales, opacities, shs)



def build_kd_tree_g_center(gaussian_blobs):
    centers = np.array([blob.pos for blob in gaussian_blobs])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers) # The idx is the same as the gaussian_blobs
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    print("There are {d} gaussian blobs in the point cloud".format(d=len(centers)))
    return pcd_tree, pcd.get_axis_aligned_bounding_box()




def generate_voxel_grid(pcd_tree : o3d.geometry.KDTreeFlann, gaussian_blobs,bbox, voxel_size=0.05):
    bbox_min = np.asarray(bbox.get_min_bound())
    bbox_max = np.asarray(bbox.get_max_bound())
    dimensions = np.ceil((bbox_max - bbox_min) / voxel_size).astype(int) + 1
    bbox_center = np.asarray(bbox.get_center())
    print("Voxel grid dimensions: ", dimensions)
    print("Voxel grid center: ", bbox_center)

    pos = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))
    colors = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))
    probes = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))
    opacities = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))
    color_blob_idx = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))
    maha_dists = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))

    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                pos[x, y, z] = np.array([x, y, z]) * voxel_size + bbox_min


            # for idx in tqdm(range(dimensions[0]*dimensions[1]*dimensions[2])):
            #     # idx = x*dimensions[1]*dimensions[2] + y*dimensions[2] + z
            #     x = idx // (dimensions[1]*dimensions[2])
            #     y = (idx % (dimensions[1]*dimensions[2])) // dimensions[2]
            #     z = (idx % (dimensions[1]*dimensions[2])) % dimensions[2]
            #     pos[idx] = np.array([x, y, z]) * voxel_size + bbox_min

            #     if idx == dimensions[0]*dimensions[1]*dimensions[2] - 1:
            #         print("Last voxel position: ", pos[idx])
            #         print("bbox", np.asarray(bbox.get_box_points() ))

                # [k, neighbor_idxs, squared_distances] = pcd_tree.search_hybrid_vector_3d(query = np.array([x, y, z]) * voxel_size, radius = 0.05, max_nn=10)
                [k, neighbor_idxs, squared_distances] = pcd_tree.search_knn_vector_3d(np.array([x, y, z]) * voxel_size + bbox_min, 20)

                # most_likely_idx, prob = get_most_likely_blob_prob(np.array([x, y, z]) * voxel_size, neighbor_idxs, gaussian_blobs)
                most_likely_idx, prob, m_distance = get_most_likely_blob_maha_distance(np.array([x, y, z]) * voxel_size + bbox_min, neighbor_idxs, gaussian_blobs)

                # if most_likely_idx == -1:
                color_blob_idx[x,y,z] = most_likely_idx
                maha_dists[x,y,z] = m_distance

                if most_likely_idx==-1 and prob == 0:
                    colors[x,y,z] = (1.0,1.0,1.0)
                    opacities[x,y,z] = 0.0
                    probes[x,y,z] = 0.0
                else:
                    colors[x,y,z] = gaussian_blobs[int(most_likely_idx)].iso_color
                    opacities[x,y,z] = gaussian_blobs[int(most_likely_idx)].opacity
                    # opacities[x,y,z] = 1.0
                    probes[x,y,z] = prob

    print(np.count_nonzero(probes))
    print(np.count_nonzero(opacities))
    # pos = pos - np.asarray(bbox.get_center())
    np.save('ply_convert/pos.npy', pos)
    np.save('ply_convert/color.npy', colors)
    np.save('ply_convert/opacity.npy', opacities)
    np.save('ply_convert/probes.npy', probes)
    np.save('ply_convert/idx.npy',  color_blob_idx)
    np.save('ply_convert/maha_dists.npy',  maha_dists)
                
    return pos, colors, opacities, dimensions
    

def compute_mahalanobis_distance(point, blob):
    # Invert the covariance matrix for the Mahalanobis distance calculation
    inv_cov_matrix = np.linalg.inv(blob.cov3D)
    
    # Calculate the Mahalanobis distance
    distance = mahalanobis(point, blob.pos, inv_cov_matrix)
    
    # Check if the distance is within 3 standard deviations
    return distance

def get_most_likely_blob_maha_distance(query_point, neighbor_idxs, gaussian_blobs, distance_threshold=5):
    min_distance = float('inf')
    min_index = -1
    for list_id, n_idx in enumerate(neighbor_idxs):
        maha_distance = compute_mahalanobis_distance(query_point, gaussian_blobs[n_idx])
        if maha_distance < min_distance:
            min_distance = maha_distance
            min_index = list_id
    if min_distance > distance_threshold:
        return -1, 0, min_distance
    return neighbor_idxs[min_index], gaussian_blobs[neighbor_idxs[min_index]].compute_3d_gaussian_prob(query_point), min_distance


def get_most_likely_blob_prob(query_point, neighbor_idxs, gaussian_blobs):
    max_index = 0
    max_prob = 0
    for list_id, n_idx in enumerate(neighbor_idxs):
        # if not is_point_within_3sigma(query_point,gaussian_blobs[i]):
        #     continue
        if gaussian_blobs[n_idx].compute_3d_gaussian_prob(query_point) >= gaussian_blobs[neighbor_idxs[max_index]].compute_3d_gaussian_prob(query_point):
            max_index = list_id
            max_prob = gaussian_blobs[n_idx].compute_3d_gaussian_prob(query_point)
    # if not is_point_within_3sigma(query_point,gaussian_blobs[neighbor_idxs[max_index]]):
    #     return -1, 0
    if max_prob < 0.03:
        return neighbor_idxs[max_index], -1
    return neighbor_idxs[max_index], max_prob


# def get_most_likely_blob_prob(k, query_point, neighbor_idxs, gaussian_blobs):
#     max_index = 0
#     max_prob = 0
#     for k, i in enumerate(neighbor_idxs):
#         if gaussian_blobs[i].compute_3d_gaussian_prob(query_point) >= gaussian_blobs[neighbor_idxs[max_index]].compute_3d_gaussian_prob(query_point):
#             max_index = k
#             max_prob = gaussian_blobs[i].compute_3d_gaussian_prob(query_point)
#     if not is_point_within_3sigma(query_point,gaussian_blobs[neighbor_idxs[max_index]]):
#         return -1, 0
#     return neighbor_idxs[max_index], max_prob

def convert_data_to_C_indexing_style(old_data,channels,dimensions):
    xres, yres, zres = dimensions

    # An empty array to hold the reorganized data
    data = np.zeros(xres * yres * zres * channels)

    # Iterate over each position in 3D space and each channel
    for xpos in range(xres):
        for ypos in range(yres):
            for zpos in range(zres):
                for chan in range(channels):
                    # idx = (xpos * yres * zres + ypos * zres + zpos)
                    new_idx = ((zpos * yres + ypos) * xres + xpos) * channels + chan
                    # Store the RGB value at the new index
                    data[new_idx] = old_data[xpos,ypos,zpos][chan]
    return data



def write_to_vol_file(filename, values, channels, bbox, dimensions, voxel_size=0.05):
    xmin, ymin, zmin = np.asarray(bbox.get_min_bound())
    xmax, ymax, zmax = (dimensions - 1) * voxel_size + np.asarray(bbox.get_min_bound())
    
    # scale to [0, 1]


    # xmin = 0.0
    # ymin = 0.0
    # zmin = 0.0

    # xmax = dimensions[0] * voxel_size
    # ymax = dimensions[1] * voxel_size
    # zmax = dimensions[2] * voxel_size
    with open(filename, 'wb') as f:
            f.write(b'VOL')
            version = 3
            type_ = 1
            f.write(version.to_bytes(1, byteorder='little'))
            f.write(np.int32(type_).newbyteorder('<').tobytes())

            f.write(np.int32(dimensions[0]).newbyteorder('<').tobytes())
            f.write(np.int32(dimensions[1]).newbyteorder('<').tobytes())
            f.write(np.int32(dimensions[2]).newbyteorder('<').tobytes())

            f.write(np.int32(channels).newbyteorder('<').tobytes())

            f.write(np.float32(xmin).newbyteorder('<').tobytes())
            f.write(np.float32(ymin).newbyteorder('<').tobytes())
            f.write(np.float32(zmin).newbyteorder('<').tobytes())
            f.write(np.float32(xmax).newbyteorder('<').tobytes())
            f.write(np.float32(ymax).newbyteorder('<').tobytes())
            f.write(np.float32(zmax).newbyteorder('<').tobytes())

            for val in values:
                f.write(np.float32(val).newbyteorder('<').tobytes())
                # if val.shape[0] == 3:
                #     # f.write(struct.pack('<f', val[0]))
                #     # f.write(struct.pack('<f', val[1]))
                #     # f.write(struct.pack('<f', val[2]))
                #     f.write(np.float32(val[0]).newbyteorder('<').tobytes())
                #     f.write(np.float32(val[1]).newbyteorder('<').tobytes())
                #     f.write(np.float32(val[2]).newbyteorder('<').tobytes())
                # else:
                #     f.write(np.float32(val[0]).newbyteorder('<').tobytes())

            f.close()
    print('Done writing to vol file')  





if __name__ == '__main__':
    ply_path = "3dgs_lego_train/point_cloud/iteration_3000/point_cloud.ply"
    # ply_path = "ply_convert/handcrafted/hand_scene.ply"

    do_intermediate_plot = False
    model = load_ply(ply_path)
    print('Loading gaussians ...')
    gaussian_blobs = []
    for (pos, scale, rot, opacity, sh) in tqdm(zip(model.xyz, model.scale, model.rot, model.opacity, model.sh)):
        gaussian_blobs.append(Gaussian(pos, scale, rot, opacity, sh))
    print('Done loading gaussians')
    
    # pcd_tree, aabb = build_kd_tree_g_center(gaussian_blobs)
    centers = np.array([blob.pos for blob in gaussian_blobs])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers) # The idx is the same as the gaussian_blobs
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    aabb = pcd.get_axis_aligned_bounding_box()

    print("There are {d} gaussian blobs in the point cloud".format(d=len(centers)))

    if do_intermediate_plot:
        # plot example gaussian blob and bbx
        mean = gaussian_blobs[152].pos
        cov = gaussian_blobs[152].cov3D
        vertices = gaussian_blobs[152].bbox
        color = tuple(gaussian_blobs[152].iso_color)
        plot_gaussian_ellipsoid_bbx(mean, cov, vertices,color)
        plot_gaussian_scene_bbox(gaussian_blobs, np.asarray(aabb.get_box_points()))

    voxel_size = 0.02
    pos, colors, opacities, dimensions = generate_voxel_grid(pcd_tree, gaussian_blobs, aabb, voxel_size=voxel_size)

    plot_voxel_data(pos, colors, opacities)

    opacities_c_indexing = convert_data_to_C_indexing_style(opacities, 1, dimensions)
    write_to_vol_file('ply_convert/density.vol', opacities_c_indexing, 1, aabb, dimensions, voxel_size=voxel_size)
    color_c_indexing = convert_data_to_C_indexing_style(colors, 3, dimensions)
    write_to_vol_file('ply_convert/color.vol', color_c_indexing, 3, aabb, dimensions, voxel_size=voxel_size)


  