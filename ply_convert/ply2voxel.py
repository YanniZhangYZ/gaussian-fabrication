import numpy as np
from tqdm import tqdm
import math
import open3d as o3d
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from guassian_utils import GaussianData, Gaussian
from scipy.spatial.distance import mahalanobis

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


def plot_gaussian_ellipsoid_bbx(mean, cov, vertices,color):
    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Get the radii of the ellipsoid
    radii = np.sqrt(eigenvalues)*3
    
    # Generate data for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    
    # Rotate and translate the points to the correct position
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j], y[i,j], z[i,j]] = np.dot([x[i,j], y[i,j], z[i,j]], eigenvectors) + mean
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha=0.5)


    # plot bounding box
    sides = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7]   # Sides
    ]

    # Plot each side
    for side in sides:
        xs = vertices[side, 0]
        ys = vertices[side, 1]
        zs = vertices[side, 2]
        ax.plot(xs, ys, zs, color='r')

    # Plot settings
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title('3D Gaussian Distribution Ellipsoid')
    plt.savefig('ply_convert/gaussuan_ellipsoid_bbox.png')


def plot_gaussian_scene_bbox(gaussian_blobs, scene_bbox):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Gaussian Blob and Scene Bounding Box')

    
    # reorder the scene_bbox vertices for printing
    correct_order = [0, 1, 7, 2, 3, 6, 4, 5]
    scene_bbox = scene_bbox[correct_order]
    # Plot the scene bounding box
    for i in range(4):
        ax.plot3D(
            *scene_bbox[[i, (i+1) % 4]].T,
            color='k'
        )
        ax.plot3D(
            *scene_bbox[[i+4, (i+1) % 4 + 4]].T,
            color='k'
        )
        ax.plot3D(
            scene_bbox[[i, i+4], 0],
            scene_bbox[[i, i+4], 1],
            scene_bbox[[i, i+4], 2],
            color='k'
        )

    # # Plot Gaussian blobs center
    visualize_idx =1000
    pos_x = np.array([blob.pos[0] for blob in gaussian_blobs[:visualize_idx]])
    pos_y = np.array([blob.pos[1] for blob in gaussian_blobs[:visualize_idx]])
    pos_z = np.array([blob.pos[2] for blob in gaussian_blobs[:visualize_idx]])
    colors = np.array([blob.iso_color for blob in gaussian_blobs[:visualize_idx]])
    opacities = np.array([blob.opacity for blob in gaussian_blobs[:visualize_idx]])
    ax.scatter(pos_x, pos_y, pos_z, c=colors,alpha=opacities)
    
    # Plot Gaussian blobs as 3-sigma ellipsoids
    # for blob in gaussian_blobs[:visualize_idx]:
    #     x,y,z = blob.compute_blob_ellipsoid()
    #     ax.plot_surface(x, y, z, color=blob.iso_color, alpha=blob.opacity)



    plt.savefig('ply_convert/g_blobs_scene_bbox.png')




def compute_scene_bbox(gaussian_blobs):
    bboxs = [blob.bbox for blob in gaussian_blobs]
    # Concatenate all vertices of all bounding boxes into a single array
    all_vertices = np.vstack(bboxs)

    # Find the minimum and maximum coordinates along each axis
    min_x, min_y, min_z = np.min(all_vertices, axis=0)
    max_x, max_y, max_z = np.max(all_vertices, axis=0)

    # Define the vertices of the overall bounding box
    overall_bbox = np.array([
        [min_x, min_y, min_z], [max_x, min_y, min_z],
        [max_x, max_y, min_z], [min_x, max_y, min_z],
        [min_x, min_y, max_z], [max_x, min_y, max_z],
        [max_x, max_y, max_z], [min_x, max_y, max_z]
    ])
    return overall_bbox


def generate_voxel_grid(pcd_tree, gaussian_blobs,bbox, voxel_size=0.05):
    bbox_min = np.asarray(bbox.get_min_bound())
    bbox_max = np.asarray(bbox.get_max_bound())
    dimensions = np.ceil((bbox_max - bbox_min) / voxel_size).astype(int)

    pos = np.zeros((dimensions[0]*dimensions[1]*dimensions[2], 3))
    colors = np.zeros((dimensions[0]*dimensions[1]*dimensions[2], 3))
    opacities = np.zeros((dimensions[0]*dimensions[1]*dimensions[2], 1))
    probes = np.zeros((dimensions[0]*dimensions[1]*dimensions[2], 1))
    color_blob_idx = np.zeros((dimensions[0]*dimensions[1]*dimensions[2], 1))



    for idx in tqdm(range(dimensions[0]*dimensions[1]*dimensions[2])):
        # idx = x*dimensions[1]*dimensions[2] + y*dimensions[2] + z
        x = idx // (dimensions[1]*dimensions[2])
        y = (idx % (dimensions[1]*dimensions[2])) // dimensions[2]
        z = (idx % (dimensions[1]*dimensions[2])) % dimensions[2]
        pos[idx] = np.array([x, y, z]) * voxel_size
        # [k, neighbor_idxs, squared_distances] = pcd_tree.search_hybrid_vector_3d(query = np.array([x, y, z]) * voxel_size, radius = 0.05, max_nn=10)
        [k, neighbor_idxs, squared_distances] = pcd_tree.search_knn_vector_3d(np.array([x, y, z]) * voxel_size, 20)

        most_likely_idx, prob = get_most_likely_blob(k, np.array([x, y, z]) * voxel_size, neighbor_idxs, gaussian_blobs)
        # if most_likely_idx == -1:
        color_blob_idx[idx] = most_likely_idx
        if prob == -1:
            colors[idx] = gaussian_blobs[int(most_likely_idx)].iso_color
            opacities[idx] = 0.0
            probes[idx] = 0.0
        else:
            colors[idx] = gaussian_blobs[int(most_likely_idx)].iso_color
            # opacities[idx] = gaussian_blobs[int(most_likely_idx)].opacity * prob
            opacities[idx] = 1.0
            probes[idx] = prob

    print(np.count_nonzero(probes))
    # pos = pos - np.asarray(bbox.get_center())
    np.savetxt('ply_convert/color.txt', colors, delimiter=',', fmt='%.2f')
    np.savetxt('ply_convert/opacity.txt', opacities, delimiter=',', fmt='%.2f')
    np.savetxt('ply_convert/probes.txt', probes, delimiter=',', fmt='%.2f')
    np.savetxt('ply_convert/idx.txt',  color_blob_idx, delimiter=',', fmt='%.2f')
   



                
    return pos, colors, opacities, dimensions
    

def build_kd_tree(gaussian_blobs):
    centers = np.array([blob.pos for blob in gaussian_blobs])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers) # The idx is the same as the gaussian_blobs
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    return pcd_tree, pcd.get_axis_aligned_bounding_box()

def is_point_within_3sigma(point,blob):
    # Invert the covariance matrix for the Mahalanobis distance calculation
    inv_cov_matrix = np.linalg.inv(blob.cov3D)
    
    # Calculate the Mahalanobis distance
    distance = mahalanobis(point, blob.pos, inv_cov_matrix)
    
    # Check if the distance is within 3 standard deviations
    return distance <= 3

def get_most_likely_blob(k, query_point, neighbor_idxs, gaussian_blobs):
    max_index = 0
    max_prob = 0
    for k, i in enumerate(neighbor_idxs):
        # if not is_point_within_3sigma(query_point,gaussian_blobs[i]):
        #     continue
        if gaussian_blobs[i].compute_3d_gaussian_prob(query_point) >= gaussian_blobs[neighbor_idxs[max_index]].compute_3d_gaussian_prob(query_point):
            max_index = k
            max_prob = gaussian_blobs[i].compute_3d_gaussian_prob(query_point)
    # if not is_point_within_3sigma(query_point,gaussian_blobs[neighbor_idxs[max_index]]):
    #     return -1, 0
    if max_prob < 0.03:
        return max_index, -1
    return neighbor_idxs[max_index], max_prob


# def get_most_likely_blob(k, query_point, neighbor_idxs, gaussian_blobs):
#     max_index = 0
#     max_prob = 0
#     for k, i in enumerate(neighbor_idxs):
#         if gaussian_blobs[i].compute_3d_gaussian_prob(query_point) >= gaussian_blobs[neighbor_idxs[max_index]].compute_3d_gaussian_prob(query_point):
#             max_index = k
#             max_prob = gaussian_blobs[i].compute_3d_gaussian_prob(query_point)
#     if not is_point_within_3sigma(query_point,gaussian_blobs[neighbor_idxs[max_index]]):
#         return -1, 0
#     return neighbor_idxs[max_index], max_prob

def write_to_vol_file(filename, values, dimensions, voxel_size=0.05):
    xmin = 0.0
    ymin = 0.0
    zmin = 0.0
    xmax = dimensions[0] * voxel_size
    ymax = dimensions[1] * voxel_size
    zmax = dimensions[2] * voxel_size
    with open(filename, 'wb') as f:
            f.write(b'VOL')
            version = 3
            type_ = 1
            f.write( version.to_bytes(1, byteorder='little'))
            f.write(np.int32(type_).newbyteorder('<').tobytes())


            f.write(np.int32(dimensions[0] - 1).newbyteorder('<').tobytes())
            f.write(np.int32(dimensions[1] - 1).newbyteorder('<').tobytes())
            f.write(np.int32(dimensions[2] - 1).newbyteorder('<').tobytes())
            f.write(np.int32(values.shape[1]).newbyteorder('<').tobytes())

            f.write(np.float32(xmin).newbyteorder('<').tobytes())
            f.write(np.float32(ymin).newbyteorder('<').tobytes())
            f.write(np.float32(zmin).newbyteorder('<').tobytes())
            f.write(np.float32(xmax).newbyteorder('<').tobytes())
            f.write(np.float32(ymax).newbyteorder('<').tobytes())
            f.write(np.float32(zmax).newbyteorder('<').tobytes())

            # f.write(struct.pack('<i', dimensions[0] - 1))
            # f.write(struct.pack('<i', dimensions[1] - 1))
            # f.write(struct.pack('<i', dimensions[2] - 1))
            # f.write(struct.pack('<i', values.shape[1]))

            # f.write(struct.pack('<f', xmin))
            # f.write(struct.pack('<f', ymin))
            # f.write(struct.pack('<f', zmin))
            # f.write(struct.pack('<f', xmax))
            # f.write(struct.pack('<f', ymax))
            # f.write(struct.pack('<f', zmax))

            for val in values:
                if val.shape[0] == 3:
                    # f.write(struct.pack('<f', val[0]))
                    # f.write(struct.pack('<f', val[1]))
                    # f.write(struct.pack('<f', val[2]))
                    f.write(np.float32(val[0]).newbyteorder('<').tobytes())
                    f.write(np.float32(val[1]).newbyteorder('<').tobytes())
                    f.write(np.float32(val[2]).newbyteorder('<').tobytes())
                else:
                    f.write(np.float32(val[0]).newbyteorder('<').tobytes())

            f.close()
    print('Done writing to vol file')  


def plot_voxel_data(pos, colors, opacities):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Voxel Data')

    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=colors, alpha=opacities)

    plt.savefig('ply_convert/voxel_data2.png')


if __name__ == '__main__':
    ply_path = "ply_convert/point_cloud.ply"
    do_intermediate_plot = True
    model = load_ply(ply_path)
    print('Loading gaussians ...')
    gaussian_blobs = []
    for (pos, scale, rot, opacity, sh) in tqdm(zip(model.xyz, model.scale, model.rot, model.opacity, model.sh)):
        gaussian_blobs.append(Gaussian(pos, scale, rot, opacity, sh))
    print('Done loading gaussians')

    scene_bbox = compute_scene_bbox(gaussian_blobs)
    


    pcd_tree,aabb = build_kd_tree(gaussian_blobs)

    if do_intermediate_plot:
        # plot example gaussian blob and bbx
        mean = gaussian_blobs[152].pos
        cov = gaussian_blobs[152].cov3D
        vertices = gaussian_blobs[152].bbox
        color = tuple(gaussian_blobs[152].iso_color)
        plot_gaussian_ellipsoid_bbx(mean, cov, vertices,color)
        plot_gaussian_scene_bbox(gaussian_blobs, np.asarray(aabb.get_box_points()))

    pos, colors, opacities, dimensions = generate_voxel_grid(pcd_tree, gaussian_blobs,aabb)

    plot_voxel_data(pos, colors, opacities)

    # write_to_vol_file('ply_convert/density.vol', opacities, dimensions)
    # write_to_vol_file('ply_convert/color.vol', colors, dimensions)