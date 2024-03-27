import numpy as np
from tqdm import tqdm
import math
import open3d as o3d
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from guassian_utils import GaussianData, Gaussian
from scipy.spatial.distance import mahalanobis



def plot_gaussian_ellipsoid_bbx(mean, cov, vertices,color,filename='ply_convert/gaussuan_ellipsoid_bbox.png'):
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
    plt.savefig(filename)


def plot_gaussian_scene_bbox(gaussian_blobs, scene_bbox, filename='ply_convert/gaussian_scene_bbox.png'):
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



    plt.savefig(filename)



def plot_voxel_data(pos, colors, opacities, show_bbox = True, filename='ply_convert/voxel_data.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Voxel Data')
    print("Ploting voxel representation")
    if opacities is None:
        ax.scatter(pos[:, :, :,0], pos[:,:,:,1], pos[:,:,:,2], c=colors.reshape(-1, 3), s=1.0)
    else:
        rgba = np.concatenate((colors.reshape(-1, 3), opacities.reshape(-1, 1)), axis=1)
        ax.scatter(pos[:, :, :,0], pos[:,:,:,1], pos[:,:,:,2], c=rgba, s=1.0)


    if show_bbox:
        # reorder the scene_bbox vertices for printing
        scene_bbox = np.array([[-0.65747279, -1.16894841, -0.42882448], 
                            [ 0.67349792, -1.16894841, -0.42882448],
                            [-0.65747279, 1.15254653, -0.42882448],
                            [-0.65747279,-1.16894841, 1.00281084],
                            [ 0.67349792, 1.15254653, 1.00281084],
                            [-0.65747279, 1.15254653, 1.00281084],
                            [ 0.67349792, -1.16894841, 1.00281084],
                            [ 0.67349792, 1.15254653, -0.42882448]])
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
            # set the axis interval to be the same
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-0.5, 1.5)
    # rotate the plot
    # ax.view_init(30, 30)

    plt.savefig(filename)
        

if __name__ == '__main__':
    pos = np.load('ply_convert/pos.npy')
    colors = np.load('ply_convert/color.npy')
    opacities = np.load('ply_convert/opacity.npy')
    probes = np.load('ply_convert/probes.npy')

    # opacities = opacities * probes

    # print(np.min(pos, axis=(0,1,2)))

    plot_voxel_data(pos, colors, opacities)