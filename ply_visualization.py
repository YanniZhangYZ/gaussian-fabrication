

# import open3d as o3d
# import matplotlib.pyplot as plt
# import numpy as np

# # Load point cloud data from .ply file
# # ply_file = "3dgs_lego_train/point_cloud/iteration_10000/point_cloud.ply"
# ply_file = "/home/yanni/Thesis/gaussian-fabrication/lego/points3d.ply"
# point_cloud = o3d.io.read_point_cloud(ply_file)
# points = np.asarray(point_cloud.points)

# # Create a figure and set the background to transparent
# fig = plt.figure(figsize=(10, 10), dpi=100)
# ax = fig.add_subplot(111, projection='3d', facecolor='none')
# fig.patch.set_alpha(0.0)

# # Plot the point cloud
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', marker='o', s=1)

# # Set the axes limits
# ax.set_xlim([points[:, 0].min(), points[:, 0].max()])
# ax.set_ylim([points[:, 1].min(), points[:, 1].max()])
# ax.set_zlim([points[:, 2].min(), points[:, 2].max()])

# # Remove the axes
# ax.axis('off')

# # Save the image with a transparent background
# # plt.savefig("point_cloud.png", transparent=True, bbox_inches='tight', pad_inches=0)
# # plt.show()
# plt.savefig("point_3d.png", transparent=True, bbox_inches='tight', pad_inches=0)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_voxel_grid(grid_size=10, voxel_size=1, edge_color='grey'):
    # Create a figure with a transparent background
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d', facecolor='none')
    fig.patch.set_alpha(0.0)
    
    # Create grid lines
    r = np.arange(0, grid_size + 1) * voxel_size
    
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color=edge_color, alpha=0.5)

    # Set limits
    ax.set_xlim([0, grid_size * voxel_size])
    ax.set_ylim([0, grid_size * voxel_size])
    ax.set_zlim([0, grid_size * voxel_size])
    
    # Remove the axes
    ax.axis('off')
    
    # Save the image with a transparent background
    plt.savefig("voxel_grid.png", transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

# Utility functions to create the grid lines
from itertools import combinations, product

draw_voxel_grid(grid_size=10, voxel_size=1, edge_color='grey')
