import numpy as np
from tqdm import tqdm
import math
import open3d as o3d
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from guassian_utils import GaussianData, Gaussian
from scipy.spatial.distance import mahalanobis
from voxel import get_most_likely_blob_maha_distance , convert_data_to_C_indexing_style
from plot_utils import plot_voxel_data
import drjit as dr
import mitsuba as mi
import json
import random
import os


mi.set_variant('llvm_ad_rgb')

from mitsuba import ScalarTransform4f as T




SH_C0 = 0.28209479177387814


#  s / (a + s)
CMYK_albedo = np.array([
    [0.05, 0.7, 0.98],  # Cyan
    [0.98, 0.1, 0.9],  # Magenta
    [0.997, 0.995, 0.15],  # Yellow
    [0.35, 0.35, 0.35],  # KEY: Black
    [0.9991, 0.9997, 0.999],   # White
    [1.0, 1.0, 1.0] #Transparent
    ])
# ink_albedo = torch.tensor(ink_albedo, dtype=torch.float32)

# a + s
CMYK_sigma_t = np.array([
        [9.0, 4.5, 7.5],  # Cyan
        [2.5, 3.0, 10.0],  # Magenta
        [2.25, 3.75, 19.0],  # Yellow
        [5.0, 5.5, 6.5],  # KEY: Black
        [6.0, 9.0, 24.0],   # White
        [1e-4, 1e-4, 1e-4]] #Transparent
        ) /20

CMYK_scattering = CMYK_albedo * CMYK_sigma_t
CMYK_absorption =  CMYK_sigma_t - CMYK_scattering

red_concentration = np.array([0.0000, 0.5107, 0.4893, 0.0000, 0.0000, 0.0000])
green_concentration = np.array([0.2231, 0.0000, 0.7769, 0.0000, 0.0000, 0.0000])

red_center_absorption= np.dot(red_concentration, CMYK_absorption)
red_center_scattering= np.dot(red_concentration, CMYK_scattering)

green_center_absorption= np.dot(green_concentration, CMYK_absorption)
green_center_scattering= np.dot(green_concentration, CMYK_scattering)


def inverse_iso_color_sh(target_color):
    target_color = np.array(target_color)
    target_color -=0.5
    c0 = target_color / SH_C0
    c0 = list(c0)
    return c0

def compute_CMYKWT_albedo_sigma_t(weights):
    print("compute_CMYKWT_albedo_sigma_t: THIS COMPUTATION IS WRONG")
    # weights element should be either 0 or 1
    albedos = np.array([
    [0.05, 0.7, 0.98],  # Cyan
    [0.98, 0.1, 0.9],  # Magenta
    [0.997, 0.995, 0.15],  # Yellow
    [0.35, 0.35, 0.35],  # KEY: Black
    [0.9991, 0.9997, 0.999],   # White
    [1.0, 1.0, 1.0] #Transparent
    ])

    sgima_t_scale_factor = 20
    sigma_ts = np.array([[9.0, 4.5, 7.5],  # Cyan
        [2.5, 3.0, 10.0],  # Magenta
        [2.25, 3.75, 19.0],  # Yellow
        [5.0, 5.5, 6.5],  # KEY: Black
        [6.0, 9.0, 24.0],   # White
        [1e-4, 1e-4, 1e-4]] #Transparent
        )/sgima_t_scale_factor
    
    
    weights = weights / np.sum(weights)
    albedo = np.dot(weights, albedos)
    sigma_t = np.dot(weights, sigma_ts)
    return albedo, sigma_t, sgima_t_scale_factor

def get_gaussian(pos, opacity, scale, target_color):
    rot = np.array([1,0,0,0])
    opacity = [opacity]
    sh = inverse_iso_color_sh(target_color)
    g = Gaussian(pos, scale, rot, opacity, sh)
    return g

def plot_handcrafted_scene(gaussian_blobs, filename='ply_convert/handcrafted/scene.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Handcrafted Gaussian Blobs and Bounding Box')

    # # Plot Gaussian blobs center
    visualize_idx = len(gaussian_blobs)
     # plot bounding box
    sides = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7]   # Sides
    ]
    
    # Plot Gaussian blobs as 3-sigma ellipsoids
    for blob in gaussian_blobs[:visualize_idx]:
        x,y,z = blob.compute_blob_ellipsoid()
        ax.plot_surface(x, y, z, color=blob.iso_color, alpha=blob.opacity)
            # Plot each side
        for side in sides:
            xs = blob.bbox[side, 0]
            ys = blob.bbox[side, 1]
            zs = blob.bbox[side, 2]
            ax.plot(xs, ys, zs, color=blob.iso_color)
    ax.view_init(0,90)
    plt.savefig(filename)


def get_two_bboxs_min_max(bbox1, bbox2):
    # Combine both bounding boxes into one array
    temp = np.vstack((bbox1, bbox2))
    # Find min and max for each axis
    min_coords = temp.min(axis=0)
    max_coords = temp.max(axis=0)
    return min_coords, max_coords


def compute_bitmap(r_opacity, g_opacity):
    bitmap_red = 0.0
    bitmap_green = 0.0
    is_red_front = r_opacity > g_opacity

    if is_red_front: # then we assume red is in the front
        red_coeff = r_opacity
        green_coeff = g_opacity* (1 - r_opacity)
    else:
        red_coeff = r_opacity* (1 - g_opacity)
        green_coeff = g_opacity

    rand_num = random.random()

    if rand_num < red_coeff:
        bitmap_red = 1.0
        bitmap_green = 0.0
    elif rand_num < red_coeff + green_coeff:
        bitmap_red = 0.0
        bitmap_green = 1.0
    else:
        bitmap_red = 0.0
        bitmap_green = 0.0

    return bitmap_red, bitmap_green

def compute_transparent_mix_absorption_scattering_alpha(r_opacity, r_param, r_color, g_opacity, g_param, g_color):
    r_absorption, r_scattering = r_param
    g_absorption, g_scattering =g_param

    is_red_front = r_opacity > g_opacity

    if is_red_front: # then we assume red is in the front
        red_coeff = r_opacity
        green_coeff = g_opacity* (1 - r_opacity)
    else:
        red_coeff = r_opacity* (1 - g_opacity)
        green_coeff = g_opacity

    g_r_absorption = r_absorption * red_coeff + g_absorption * green_coeff + CMYK_absorption[5] * (1 - r_opacity- g_opacity)
    g_r_scattering = r_scattering * red_coeff + g_scattering * green_coeff + CMYK_scattering[5] * (1 - r_opacity- g_opacity)
    g_r_color = r_color * red_coeff + g_color * green_coeff + np.array([1.0, 1.0, 1.0]) * (1 - r_opacity- g_opacity)

    return g_r_absorption, g_r_scattering, g_r_color


def compute_transparent_mix_absorption_scattering_naive(r_opacity, r_param, r_color, g_opacity, g_param, g_color):
    r_absorption, r_scattering = r_param
    g_absorption,g_scattering =g_param

    r_coeff = r_opacity /2
    g_coeff = g_opacity /2
    tras_coeff = 1 - r_opacity / 2 - g_opacity / 2

    g_r_absorption = r_absorption * r_coeff + g_absorption * g_coeff + CMYK_absorption[5] * tras_coeff
    g_r_scattering = r_scattering * r_coeff + g_scattering * g_coeff + CMYK_scattering[5] * tras_coeff
    g_r_color = r_color * r_coeff + g_color * g_coeff + np.array([1.0, 1.0, 1.0]) * tras_coeff

    return g_r_absorption, g_r_scattering, g_r_color

def compute_transparent_mix_absorption_scattering(front_opacity, front_param, front_color, back_opacity, back_param, back_color, g_r_same_pos = False):
    front_absorption, front_scattering = front_param
    back_absorption, back_scattering = back_param

    if g_r_same_pos:
        g_r_absorption = (front_absorption + back_absorption) /2
        g_r_scattering = (front_scattering + back_scattering) /2
        g_r_color = (front_color + back_color) /2

        g_r_trans_mix_absorption = front_opacity * g_r_absorption + (1 - front_opacity) * CMYK_absorption[5] #mix with transparent
        g_r_trans_mix_scattering = front_opacity * g_r_scattering + (1 - front_opacity) * CMYK_scattering[5]
        return g_r_trans_mix_absorption, g_r_trans_mix_scattering, g_r_color


    else:
        back_trans_mix_absoprtion = back_opacity * back_absorption + (1 - back_opacity) * CMYK_absorption[5] #mix with transparent
        back_trans_mix_scattering = back_opacity * back_scattering + (1 - back_opacity) * CMYK_scattering[5]
        back_white_mix_color = back_opacity * back_color + (1 - back_opacity)* np.array([1.0, 1.0, 1.0])

        front_back_mix_absorption = front_opacity * front_absorption + (1 - front_opacity) * back_trans_mix_absoprtion
        front_back_mix_scattering = front_opacity * front_scattering + (1 - front_opacity) * back_trans_mix_scattering
        front_back_mix_color = front_opacity * front_color  + (1 - front_opacity) * back_white_mix_color
        return front_back_mix_absorption, front_back_mix_scattering, front_back_mix_color

def compute_mahalanobis_distance(point,blob):
    # Invert the covariance matrix for the Mahalanobis distance calculation
    inv_cov_matrix = np.linalg.inv(blob.cov3D)
    
    # Calculate the Mahalanobis distance
    distance = mahalanobis(point, blob.pos, inv_cov_matrix)
    
    # Check if the distance is within 3 standard deviations
    return distance


def generate_3d_bitmap_voxel_grid(min_coords, max_coords, gaussian_blobs,voxel_size=0.05):
    dimensions = np.ceil((max_coords- min_coords) / voxel_size).astype(int) + 1
    bbox_center = (min_coords + max_coords) / 2
    print("Voxel grid dimensions: ", dimensions)
    print("Voxel grid center: ", bbox_center)

    pos = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))
    colors = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))
    prob_r = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))
    prob_g = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))
    albedo = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))
    sigma_t = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))
    bitmap_r = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))
    bitmap_g = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))


    red_albedo = red_center_scattering / (red_center_absorption + red_center_scattering)
    red_sigma_t = red_center_absorption + red_center_scattering
    green_albedo = green_center_scattering / (green_center_absorption + green_center_scattering)
    green_sigma_t = green_center_absorption + green_center_scattering


    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                pos[x, y, z] = np.array([x, y, z]) * voxel_size + min_coords
                prob_r[x, y, z] = gaussian_blobs[0].compute_3d_gaussian_prob(np.array([x, y, z]) * voxel_size + min_coords)
                prob_g[x, y, z] = gaussian_blobs[1].compute_3d_gaussian_prob(np.array([x, y, z]) * voxel_size + min_coords)

                opacity_r = gaussian_blobs[0].opacity * prob_r[x,y,z]
                opacity_g = gaussian_blobs[1].opacity * prob_g[x,y,z]
                
                bitmap_r[x,y,z], bitmap_g[x,y,z] = compute_bitmap(opacity_r, opacity_g)
                
    albedo = bitmap_r * red_albedo + bitmap_g * green_albedo
    sigma_t = bitmap_r * red_sigma_t + bitmap_g * green_sigma_t
    bitmap_tans = 1 - bitmap_r - bitmap_g
    colors = bitmap_r * gaussian_blobs[0].iso_color + bitmap_g * gaussian_blobs[1].iso_color + bitmap_tans * np.array([1.0, 1.0, 1.0])
    return pos, colors, dimensions, albedo, sigma_t



def generate_hetero_voxel_grid(min_coords, max_coords, gaussian_blobs,voxel_size=0.05):
    # dimensions = np.ceil((max_coords- min_coords) / voxel_size).astype(int) + 1
    dimensions = np.ceil((max_coords- min_coords) / voxel_size).astype(int)
    bbox_center = (min_coords + max_coords) / 2
    print("Voxel grid dimensions: ", dimensions)
    print("Voxel grid center: ", bbox_center)

    pos = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))
    colors = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))
    prob_r = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))
    prob_g = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))
    albedo = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))
    sigma_t = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))

    red_param = (red_center_absorption, red_center_scattering)
    green_param = (green_center_absorption, green_center_scattering)

    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                pos[x, y, z] = np.array([x, y, z]) * voxel_size + min_coords
                prob_r[x, y, z] = gaussian_blobs[0].compute_3d_gaussian_prob(np.array([x, y, z]) * voxel_size + min_coords)
                prob_g[x, y, z] = gaussian_blobs[1].compute_3d_gaussian_prob(np.array([x, y, z]) * voxel_size + min_coords)

                opacity_r = gaussian_blobs[0].opacity * prob_r[x,y,z]
                opacity_g = gaussian_blobs[1].opacity * prob_g[x,y,z]
                
                # is_red_front = opacity_r > opacity_g
                # g_r_same_pos = np.array_equal(gaussian_blobs[0].pos, gaussian_blobs[1].pos)
                # if g_r_same_pos:
                #     a, s, c = compute_transparent_mix_absorption_scattering(
                #                                 opacity_r, red_param, gaussian_blobs[0].iso_color,
                #                                 opacity_g, green_param,gaussian_blobs[1].iso_color,
                #                                 g_r_same_pos)
                # else:
                #     if is_red_front:
                #         a, s, c = compute_transparent_mix_absorption_scattering(
                #                                 opacity_r, red_param, gaussian_blobs[0].iso_color,
                #                                 opacity_g, green_param,gaussian_blobs[1].iso_color)
                #     else:
                #         a, s, c = compute_transparent_mix_absorption_scattering(
                #                                     opacity_g, green_param, gaussian_blobs[1].iso_color,
                #                                     opacity_r, red_param,gaussian_blobs[0].iso_color)

                # a, s, c = compute_transparent_mix_absorption_scattering_naive(opacity_r, red_param, gaussian_blobs[0].iso_color,
                #                                                               opacity_g, green_param,gaussian_blobs[1].iso_color)
                
                a,s,c = compute_transparent_mix_absorption_scattering_alpha(opacity_r, red_param, gaussian_blobs[0].iso_color,
                                                                            opacity_g, green_param,gaussian_blobs[1].iso_color) 
                colors[x,y,z] = c
                albedo[x,y,z] = s / (a + s)
                sigma_t[x,y,z] = a + s

    path = 'ply_convert/Czech_color_fitting/meta/'
    os.makedirs(path, exist_ok=True)
    np.save('ply_convert/Czech_color_fitting/meta/pos.npy', pos)
    np.save('ply_convert/Czech_color_fitting/meta/color.npy', colors)
    np.save('ply_convert/Czech_color_fitting/meta/albedo.npy', albedo)
    np.save('ply_convert/Czech_color_fitting/meta/sigma_t.npy', sigma_t)
    return pos, colors, dimensions, albedo, sigma_t




def generate_voxel_grid(min_coords, max_coords, gaussian_blobs, ink_params, voxel_size=0.05):
    ink_red_ablbedo,ink_red_sigma_t, ink_green_ablbedo, ink_green_sigma_t, _, _= ink_params
    dimensions = np.ceil((max_coords- min_coords) / voxel_size).astype(int) + 1
    bbox_center = (min_coords + max_coords) / 2
    print("Voxel grid dimensions: ", dimensions)
    print("Voxel grid center: ", bbox_center)

    pos = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))
    colors = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))
    opacities = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))
    color_blob_idx = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))
    prob_r = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))
    prob_g = np.zeros((dimensions[0], dimensions[1], dimensions[2], 1))
    albedo = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))
    sigma_t = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3))

    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                pos[x, y, z] = np.array([x, y, z]) * voxel_size + min_coords
                prob_r[x, y, z] = gaussian_blobs[0].compute_3d_gaussian_prob(np.array([x, y, z]) * voxel_size + min_coords)
                prob_g[x, y, z] = gaussian_blobs[1].compute_3d_gaussian_prob(np.array([x, y, z]) * voxel_size + min_coords)
                rand_num = random.random()
                if rand_num < prob_r[x, y, z] / (prob_g[x, y, z] + prob_r[x, y, z]):
                    colors[x,y,z] = gaussian_blobs[0].iso_color
                    opacities[x,y,z] = gaussian_blobs[0].opacity
                    albedo[x,y,z] = ink_red_ablbedo
                    sigma_t[x,y,z] = ink_red_sigma_t
                    color_blob_idx[x,y,z] = 0

                else:
                    colors[x,y,z] = gaussian_blobs[1].iso_color
                    opacities[x,y,z] = gaussian_blobs[1].opacity
                    albedo[x,y,z] = ink_green_ablbedo
                    sigma_t[x,y,z] = ink_green_sigma_t
                    color_blob_idx[x,y,z] = 1

    np.save('ply_convert/handcrafted/meta/pos.npy', pos)
    np.save('ply_convert/handcrafted/meta/color.npy', colors)
    np.save('ply_convert/handcrafted/meta/opacity.npy', opacities)
    np.save('ply_convert/handcrafted/meta/albedo.npy', albedo)
    np.save('ply_convert/handcrafted/meta/sigma_t.npy', sigma_t)
    np.save('ply_convert/handcrafted/meta/idx.npy',  color_blob_idx)
    return pos, colors, opacities, dimensions, albedo, sigma_t, color_blob_idx


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

def write_to_vol_file(filename, values, channels, min_coords, dimensions, voxel_size=0.05):
    xmin, ymin, zmin = min_coords
    xmax, ymax, zmax = dimensions * voxel_size + min_coords

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

            f.close()
    print('Done writing to vol file')  



def get_mitsuba_scene_dict(ink_params, pos_r, pos_g, cov3D_scale):
    ink_red_ablbedo,ink_red_sigma_t, ink_green_ablbedo, ink_green_sigma_t, r_sigam_t_scale, g_sigam_t_scale= ink_params
    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'volpathmis'},
        'red_cude': {
            'type': 'cube', #red cude
            'to_world': T.translate(pos_r).rotate([0, 1, 0], 0).scale(list(cov3D_scale*6)),
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'homogeneous',
                'albedo': {
                    'type': 'rgb',
                    'value': list(ink_red_ablbedo)
                },
                'sigma_t': {
                    'type': 'rgb',
                    'value': list(ink_red_sigma_t)
                },
                'scale': r_sigam_t_scale
            }
        },
        'green_cude': {
            'type': 'cube', #red cude
            'to_world': T.translate(pos_g).rotate([0, 1, 0], 0).scale(list(cov3D_scale*6)),
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'homogeneous',
                'albedo': {
                    'type': 'rgb',
                    'value': list(ink_green_ablbedo)
                },
                'sigma_t': {
                    'type': 'rgb',
                    'value': list(ink_green_sigma_t)
                },
                'scale': g_sigam_t_scale
            }
        },
        'emitter': {'type': 'constant'}
    }

    return scene_dict


def get_mixing_mitsuba_scene_dict(sigam_t_scale, bbox_center, bbox_scale, albedo_file_path, sigma_t_file_path, filter_type = 'trilinear'):
    print('bbox_scale', bbox_scale)
    print('bbox_center', bbox_center)
    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'volpathmis'},
        'mix_cude': {
            'type': 'cube',
            'to_world': T.scale(bbox_scale / 2),

            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'heterogeneous',
                'albedo': {
                    'type': 'gridvolume',
                    'filename': albedo_file_path,
                    'use_grid_bbox': True,
                    # 'filter_type': 'nearest',
                    # 'to_world': T.translate(list(bbox_center)).rotate([0, 1, 0], 0).scale(list(bbox_scale)),
                    'to_world': T.translate(-bbox_center),
                    'filter_type': filter_type
                },
                'sigma_t': {
                    'type': 'gridvolume',
                    'filename': sigma_t_file_path,
                    'use_grid_bbox': True,
                    # 'filter_type': 'nearest',
                    # 'to_world':T.translate(list(bbox_center)).rotate([0, 1, 0], 0).scale(list(bbox_scale)),
                    'to_world': T.translate(-bbox_center),
                    'filter_type': filter_type
                },
                'scale': sigam_t_scale
            }
        },
        'emitter': {'type': 'constant'}
    }

    return scene_dict


def render_mitsuba_scene(scene_dict, bbox_scale, filepath = 'ply_convert/handcrafted/render/', set_spp = 256, view_idx = None):
    sensor_count = 12
    sensors = []
    origin_z = np.max(bbox_scale) * 1.5

    for i in range(sensor_count):
        angle = 360.0 / sensor_count * i
        sensor_rotation = T.rotate([0, 1, 0], angle)
        sensor_to_world = T.look_at(target=[0, 0, 0], origin=[0, 0, 4], up=[0, 1, 0])
        sensors.append(mi.load_dict({
            'type': 'perspective',
            'fov': 45,
            'to_world': sensor_rotation @ sensor_to_world,
            'film': {
                'type': 'hdrfilm',
                'width': 512, 'height': 512,
                'filter': {'type': 'gaussian'}
            }
        }))
    
    scene_ref = mi.load_dict(scene_dict)

    fig, axs = plt.subplots(1, sensor_count,figsize=(14, 4))

    # ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]
    if view_idx is None:
        for i in range(sensor_count):
            print('Rendering view ', i)
            img = mi.render(scene_ref, sensor=sensors[i], spp=set_spp)
            name = filepath+'view' + str(i)
            mi.Bitmap(img).write(name+'.exr')
            mi.util.write_bitmap(name+'.png', img)
            axs[i].imshow(mi.util.convert_to_bitmap(img))
            axs[i].axis('off')
        plt.savefig(filepath+"all_view.png")
    else:
        img = mi.render(scene_ref, sensor=sensors[view_idx], spp=set_spp)
        name = filepath+'view' + str(view_idx)
        mi.Bitmap(img).write(name+'.exr')
        mi.util.write_bitmap(name+'.png', img)


def store_gaussian_2_ply(gaussian_blobs,filename='ply_convert/handcrafted/hand_scene.ply'):
    pos = np.array([g.pos for g in gaussian_blobs])
    rot = np.zeros((pos.shape[0],4))
    for i, g in enumerate(gaussian_blobs):
        rot_obj = g.rot.as_quat()
        r= np.array([rot_obj[3], rot_obj[0], rot_obj[1], rot_obj[2]])
        rot[i] = r
    scale = np.array([g.scale for g in gaussian_blobs])
    print(scale)

    opacity = np.array([g.opacity for g in gaussian_blobs]).reshape(-1,1)  
    print(opacity)

    sh = np.array([g.sh for g in gaussian_blobs])

    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    
    dtype_full = [(attribute, 'f4') for attribute in l]

    xyz = pos
    normals = np.zeros_like(xyz)
    f_dc = sh
    print(f_dc)

    opacities = -np.log((1 - opacity) / opacity)
    scale = np.log(scale)
    rotation = rot


    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    print(attributes.shape)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(filename)


def run_halfton():
    use_transparent = 0.0 # should be either 0.0 opaque or 1.0 tranparent
    ink_red_ablbedo, ink_red_sigma_t, ink_red_sigma_t_scale_factor = compute_CMYKWT_albedo_sigma_t(np.array([0.0, 1.0, 1.0, 0.0, 0.0, use_transparent]))
    ink_green_ablbedo, ink_green_sigma_t, ink_green_sigma_t_scale_factor = compute_CMYKWT_albedo_sigma_t(np.array([1.0, 0.0, 1.0, 0.0, 0.0, use_transparent]))
    ink_opaque_opacity =  0.786 if not use_transparent else 0.201 # according to "3D Printing Spatially Translucency" paper
    voxel_size = 0.02

    cov3D_scale = np.array([0.03, 0.15, 0.15])
    # Red gaussian blob:
    # pos_r = [0.2, 0, 0]
    RGB_red = [1.0, 0.0, 0.0]
    # g_r = get_gaussian(pos_r, ink_opaque_opacity,cov3D_scale, RGB_red)
    pos_r = [0.01, 0, 0]
    # cov3D_scale_r = np.array([0.15, 0.15, 0.15])
    g_r = get_gaussian(pos_r, 0.8,cov3D_scale, RGB_red)




    # Green gaussian blob:
    # pos_g = [-0.2, 0, 0]
    RGB_green = [0.0, 1.0, 0.0]
    # g_g = get_gaussian(pos_g, ink_opaque_opacity, cov3D_scale, RGB_green) # RGB green
    pos_g = [-0.01, 0, 0]
    # cov3D_scale_g = np.array([0.15, 0.15, 0.15])
    g_g = get_gaussian(pos_g, 0.8, cov3D_scale, RGB_green) # RGB green

    gaussian_blobs = [g_r,g_g]

    # store_gaussian_2_ply(gaussian_blobs)


    # # plot the handcrafted scene using matplotlib
    # plot_handcrafted_scene(gaussian_blobs)

    # print("=========== Mitsuba Rendering =============")
    ink_params = (ink_red_ablbedo,ink_red_sigma_t, ink_green_ablbedo, ink_green_sigma_t, ink_red_sigma_t_scale_factor, ink_green_sigma_t_scale_factor)
    # scene_dict = get_mitsuba_scene_dict(ink_params, pos_r, pos_g, cov3D_scale)
    # render_mitsuba_scene(scene_dict)
    # print("=========== Finish ==============")

    min_coords, max_coords = get_two_bboxs_min_max(g_r.bbox, g_g.bbox)
    pos, colors, opacities, dimensions, albedo, sigma_t, color_blob_idx = generate_voxel_grid(min_coords, max_coords, gaussian_blobs, ink_params, voxel_size=voxel_size)
   
    plot_voxel_data(pos, colors, opacities, show_bbox = False, filename='ply_convert/handcrafted/meta/voxel_data.png')
   
    converted_albedo = convert_data_to_C_indexing_style(albedo,3,dimensions)
    converted_sigma_t = convert_data_to_C_indexing_style(sigma_t,3,dimensions)

    file_albedo = 'ply_convert/handcrafted/meta/albedo.vol'
    file_sigma_t = 'ply_convert/handcrafted/meta/sigma_t.vol'
    write_to_vol_file(file_albedo, converted_albedo, 3, min_coords, dimensions, voxel_size=voxel_size)
    write_to_vol_file(file_sigma_t, converted_sigma_t, 3, min_coords, dimensions, voxel_size=voxel_size)

    scene_dict = get_mixing_mitsuba_scene_dict(ink_red_sigma_t_scale_factor, 
                                               (min_coords + max_coords)/2,
                                               max_coords - min_coords,
                                               'ply_convert/handcrafted/meta/albedo.vol', 
                                               'ply_convert/handcrafted/meta/sigma_t.vol')
    render_mitsuba_scene(scene_dict, max_coords - min_coords)


def run_hetero():

    use_transparent = 0.0 # should be either 0.0 opaque or 1.0 tranparent
    center_opacity =  0.99
    voxel_size = 0.05

    cov3D_scale = np.array([0.2,0.2,0.2])
    # Red gaussian blob:
    # pos_r = [0.2, 0, 0]
    RGB_red = [1.0, 0.0, 0.0]
    # g_r = get_gaussian(pos_r, ink_opaque_opacity,cov3D_scale, RGB_red)
    pos_r = [0.15, 0, 0]
    # cov3D_scale_r = np.array([0.15, 0.15, 0.15])
    g_r = get_gaussian(pos_r,center_opacity ,cov3D_scale, RGB_red)


    # Green gaussian blob:
    # pos_g = [-0.2, 0, 0]
    RGB_green = [0.0, 1.0, 0.0]
    cov3D_scale = np.array([0.1,0.1,0.1])
    # center_opacity =  0.5

    # g_g = get_gaussian(pos_g, ink_opaque_opacity, cov3D_scale, RGB_green) # RGB green
    pos_g = [-0.4, 0, 0]
    # cov3D_scale_g = np.array([0.15, 0.15, 0.15])
    g_g = get_gaussian(pos_g, center_opacity, cov3D_scale, RGB_green) # RGB green

    gaussian_blobs = [g_r,g_g]

    store_gaussian_2_ply(gaussian_blobs,filename='ply_convert/Czech_color_fitting/meta/hand_scene.ply')

    plot_handcrafted_scene(gaussian_blobs, filename='ply_convert/Czech_color_fitting/scene.png')

    min_coords, max_coords = get_two_bboxs_min_max(g_r.bbox, g_g.bbox)
    pos, colors, dimensions, albedo, sigma_t = generate_hetero_voxel_grid(min_coords, max_coords, gaussian_blobs,voxel_size=voxel_size)
   
    plot_voxel_data(pos, colors, None, show_bbox = False, filename='ply_convert/Czech_color_fitting/voxel_data.png')
   
    converted_albedo = convert_data_to_C_indexing_style(albedo,3,dimensions)
    converted_sigma_t = convert_data_to_C_indexing_style(sigma_t,3,dimensions)

    file_albedo = 'ply_convert/Czech_color_fitting/meta/albedo.vol'
    file_sigma_t = 'ply_convert/Czech_color_fitting/meta/sigma_t.vol'
    write_to_vol_file(file_albedo, converted_albedo, 3, min_coords, dimensions, voxel_size=voxel_size)
    write_to_vol_file(file_sigma_t, converted_sigma_t, 3, min_coords, dimensions, voxel_size=voxel_size)

    scene_dict = get_mixing_mitsuba_scene_dict(20, 
                                               (min_coords + max_coords)/2,
                                               (max_coords - min_coords),
                                               'ply_convert/Czech_color_fitting/meta/albedo.vol',
                                               'ply_convert/Czech_color_fitting/meta/sigma_t.vol')
    render_mitsuba_scene(scene_dict,max_coords - min_coords, filepath = 'ply_convert/Czech_color_fitting/render/',view_idx = 3)


def run_comparison():
    ''''
    Set 1:
    - voxel_size = 0.05
    - Red gaussian blob:
        - conv3D_scale = [0.2,0.2,0.2]
        - center_opacity =  0.99
        - pos_r = [0.15, 0, 0]
    - Green gaussian blob:
        - cov3D_scale = [0.1,0.1,0.1]
        - center_opacity =  0.5
        - pos_g = [-0.4, 0, 0]
    
    
    
    '''

    scene_scale = 1
    voxel_size = 0.2 * scene_scale
    experiment_idx = 1
    folderpath = 'ply_convert/comparison/set{}/'.format(experiment_idx)
    metapath = folderpath + 'meta/'
    renderpath = folderpath + 'render/'


    if not os.path.exists(metapath):
        os.makedirs(metapath)
    if not os.path.exists(renderpath):
        os.makedirs(renderpath)

    # Red gaussian blob:
    cov3D_scale = np.array([0.2,0.2,0.2]) * scene_scale
    center_opacity =  0.99
    RGB_red = [1.0, 0.0, 0.0]
    pos_r = [0.15 * scene_scale, 0, 0] 
    g_r = get_gaussian(pos_r,center_opacity ,cov3D_scale, RGB_red)


    # Green gaussian blob:
    cov3D_scale = np.array([0.1,0.1,0.1]) * scene_scale
    center_opacity =  0.5
    RGB_green = [0.0, 1.0, 0.0]
    pos_g = [-0.4 * scene_scale , 0, 0]
    g_g = get_gaussian(pos_g, center_opacity, cov3D_scale, RGB_green) # RGB green

    gaussian_blobs = [g_r,g_g]

    plot_handcrafted_scene(gaussian_blobs, filename=metapath + 'scene.png')


    store_gaussian_2_ply(gaussian_blobs,filename=metapath + 'hand_scene.ply')


    min_coords, max_coords = get_two_bboxs_min_max(g_r.bbox, g_g.bbox)
    pos, colors, dimensions, albedo, sigma_t = generate_hetero_voxel_grid(min_coords, max_coords, gaussian_blobs,voxel_size=voxel_size)
   
    plot_voxel_data(pos, colors, None, show_bbox = False, filename=metapath + 'voxel_data.png')
   
    converted_albedo = convert_data_to_C_indexing_style(albedo,3,dimensions)
    converted_sigma_t = convert_data_to_C_indexing_style(sigma_t,3,dimensions)

    file_albedo = metapath + 'albedo.vol'
    file_sigma_t = metapath + 'sigma_t.vol'
    write_to_vol_file(file_albedo, converted_albedo, 3, min_coords, dimensions, voxel_size=voxel_size)
    write_to_vol_file(file_sigma_t, converted_sigma_t, 3, min_coords, dimensions, voxel_size=voxel_size)

    scene_dict = get_mixing_mitsuba_scene_dict(20*3, 
                                               (min_coords + max_coords)/2,
                                               (max_coords - min_coords),
                                                metapath + 'albedo.vol',
                                                metapath + 'sigma_t.vol')
    render_mitsuba_scene(scene_dict,max_coords - min_coords, filepath = renderpath, set_spp = 128)


def run_bitmap():
    scene_scale = 1
    # voxel_size = 0.05 * scene_scale
    voxel_size = 0.01

    folderpath = 'ply_convert/comparison/bitmap/'
    metapath = folderpath + 'meta/'
    renderpath = folderpath + 'render/'


    if not os.path.exists(metapath):
        os.makedirs(metapath)
    if not os.path.exists(renderpath):
        os.makedirs(renderpath)

    # Red gaussian blob:
    cov3D_scale = np.array([0.2,0.2,0.2]) * scene_scale
    center_opacity =  0.99
    RGB_red = [1.0, 0.0, 0.0]
    pos_r = [0.15 * scene_scale, 0, 0] 
    g_r = get_gaussian(pos_r,center_opacity ,cov3D_scale, RGB_red)


    # Green gaussian blob:
    cov3D_scale = np.array([0.1,0.1,0.1]) * scene_scale
    center_opacity =  0.5
    RGB_green = [0.0, 1.0, 0.0]
    pos_g = [-0.4 * scene_scale , 0, 0]
    g_g = get_gaussian(pos_g, center_opacity, cov3D_scale, RGB_green) # RGB green

    gaussian_blobs = [g_r,g_g]

    plot_handcrafted_scene(gaussian_blobs, filename=metapath + 'scene.png')


    store_gaussian_2_ply(gaussian_blobs,filename=metapath + 'hand_scene.ply')


    min_coords, max_coords = get_two_bboxs_min_max(g_r.bbox, g_g.bbox)

    pos, colors, dimensions, albedo, sigma_t =  generate_3d_bitmap_voxel_grid(min_coords, max_coords, gaussian_blobs,voxel_size=voxel_size)   
    plot_voxel_data(pos, colors, None, show_bbox = False, filename=metapath + 'voxel_data.png')
   
    converted_albedo = convert_data_to_C_indexing_style(albedo,3,dimensions)
    converted_sigma_t = convert_data_to_C_indexing_style(sigma_t,3,dimensions)

    file_albedo = metapath + 'albedo.vol'
    file_sigma_t = metapath + 'sigma_t.vol'
    write_to_vol_file(file_albedo, converted_albedo, 3, min_coords, dimensions, voxel_size=voxel_size)
    write_to_vol_file(file_sigma_t, converted_sigma_t, 3, min_coords, dimensions, voxel_size=voxel_size)

    # min_coords = np.array([-0.7, -0.6, -0.6])
    # max_coords = np.array([0.75, 0.6, 0.6])
    
    scene_dict = get_mixing_mitsuba_scene_dict(20*5, 
                                               (min_coords + max_coords)/2,
                                               (max_coords - min_coords),
                                                metapath + 'albedo.vol',
                                                metapath + 'sigma_t.vol',
                                                filter_type = 'nearest')
    render_mitsuba_scene(scene_dict,max_coords - min_coords, filepath = renderpath, set_spp = 128)

if __name__ == '__main__':
    
    run_comparison()
    # run_bitmap()
  
