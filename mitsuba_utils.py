import numpy as np
import os
import matplotlib.pyplot as plt
import torch

from utils.graphics_utils import getWorld2View2

import mitsuba as mi

mi.set_variant('cuda_ad_rgb') #AttributeError: jit_init_thread_state(): the LLVM backend is inactive because the LLVM shared library ("libLLVM.so") could not be found! Set the DRJIT_LIBLLVM_PATH environment variable to specify its path.
# mi.set_variant('scalar_rgb')


from mitsuba import ScalarTransform4f as T


def get_mixing_mitsuba_scene_dict(sigam_t_scale, bbox_center, bbox_scale, albedo_file_path, sigma_t_file_path, filter_type = 'nearest'):
    print('bbox_scale', bbox_scale)
    print('bbox_center', bbox_center)
    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'volpathmis'},
        'mix_cude': {
            'type': 'cube',
            'to_world': T.scale(bbox_scale / 2).translate(bbox_center * 2 / bbox_scale),

            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'heterogeneous',
                'albedo': {
                    'type': 'gridvolume',
                    'filename': albedo_file_path,
                    # 'grid': albedo_file_path,
                    'use_grid_bbox': True,
                    # 'filter_type': 'nearest',
                    # 'to_world': T.translate(list(bbox_center)).rotate([0, 1, 0], 0).scale(list(bbox_scale)),
                    # 'to_world': T.translate(-bbox_center),
                    'filter_type': filter_type
                },
                'sigma_t': {
                    'type': 'gridvolume',
                    'filename': sigma_t_file_path,
                    # 'grid': sigma_t_file_path,
                    'use_grid_bbox': True,
                    # 'filter_type': 'nearest',
                    # 'to_world':T.translate(list(bbox_center)).rotate([0, 1, 0], 0).scale(list(bbox_scale)),
                    # 'to_world': T.translate(-bbox_center),
                    'filter_type': filter_type
                },
                'scale': sigam_t_scale
            }
        },
        'emitter': {'type': 'constant',
                    'radiance': {
                        'type': 'rgb',
                        'value': 1.0,
                        }
                    }
    }

    return scene_dict

def get_camera_dict(viewpoint_camera):
    R = viewpoint_camera.R
    T_ = viewpoint_camera.T

    # Mitsuba and 3dgs camera use different coordinate system, one left-handed and the other right-handed

    #flipping the z axis of R
    flip_z_R =  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    R = R @ flip_z_R
    #flipping the z axis of T_
    T_ = T_ * np.array([-1, -1, 1])


    to_world_np = getWorld2View2(R, T_)
    to_world_np = np.linalg.inv(to_world_np)
    to_world = T(to_world_np)
    camera_dict = {
            'type': 'perspective',
            'fov': 40,
            # 'to_world': sensor_rot @ to_world,
            'to_world': to_world,
            'near_clip': 0.01,
            'far_clip': 100.0,
            'film': {
                'type': 'hdrfilm',
                # 'width': int(viewpoint_camera.image_width), 
                # 'height': int(viewpoint_camera.image_height),
                'width': 800, 
                'height': 800,
                'filter': {'type': 'gaussian'}
            }
        }
    return camera_dict



def render_mitsuba_scene(scene_dict, sensor_dict, bbox_scale, filepath = None, set_spp = 256, view_idx = None):
    assert filepath is not None, "need to specify a filepath to save the rendered images"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    sensor_count = 6
    sensors = []
    origin_z = np.max(bbox_scale) * 1.5

    # for i in range(sensor_count):
    #     angle = 360.0 / sensor_count * i
    #     sensor_rotation = T.rotate([0, 1, 0], angle)
    #     sensor_to_world = T.look_at(target=[0, 0, 0], origin=[0, 0, 10], up=[0, 1, 0])
    #     sensors.append(mi.load_dict({
    #         'type': 'perspective',
    #         'fov': 45,
    #         'to_world': sensor_rotation @ sensor_to_world,
    #         'film': {
    #             'type': 'hdrfilm',
    #             'width': 512, 'height': 512,
    #             'filter': {'type': 'gaussian'}
    #         }
    #     }))

    sensor =  mi.load_dict(sensor_dict)
    
    scene_ref = mi.load_dict(scene_dict)

    fig, axs = plt.subplots(1, sensor_count,figsize=(14, 4))

    # ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]
    # if view_idx is None:
    #     for i in range(sensor_count):
    #         print('Rendering view ', i)
    #         img = mi.render(scene_ref, sensor=sensors[i], spp=set_spp)
    #         name = os.path.join(filepath,'view' + str(i))
    #         mi.Bitmap(img).write(name+'.exr')
    #         mi.util.write_bitmap(name+'.png', img)
    #         axs[i].imshow(mi.util.convert_to_bitmap(img))
    #         axs[i].axis('off')
    #     plt.savefig(os.path.join(filepath,"all_view.png"))
    # else:
    img = mi.render(scene_ref, sensor=sensor, spp=set_spp)
    name = os.path.join(filepath,'view' + str(view_idx))
    mi.Bitmap(img).write(name+'.exr')
    mi.util.write_bitmap(name+'.png', img)



def write_to_vol_file(filename, values, channels, min_coords, dimensions, voxel_size=0.05):
    xmin, ymin, zmin = min_coords
    # xmax, ymax, zmax = (dimensions - 1) * voxel_size + min_coords
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



def convert_data_to_C_indexing_style(old_data,channels,dimensions):
    '''
        input: old_data: (xres, yres, zres, channels)
        output: data: (xres * yres * zres * channels)
    '''
    xres, yres, zres = dimensions

    data = np.transpose(old_data, (2, 1, 0, 3)).reshape(xres * yres * zres * channels)

    # # An empty array to hold the reorganized data
    # data = np.zeros(xres * yres * zres * channels)

    # # Iterate over each position in 3D space and each channel
    # for xpos in range(xres):
    #     for ypos in range(yres):
    #         for zpos in range(zres):
    #             for chan in range(channels):
    #                 # idx = (xpos * yres * zres + ypos * zres + zpos)
    #                 new_idx = ((zpos * yres + ypos) * xres + xpos) * channels + chan
    #                 # Store the RGB value at the new index
    #                 data[new_idx] = old_data[xpos,ypos,zpos][chan]
    return data
