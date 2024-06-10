import numpy as np
import os
import matplotlib.pyplot as plt
import torch

from utils.graphics_utils import getWorld2View2
from ink_intrinsics import Ink
from mitsuba_utils import write_to_vol_file, convert_data_to_C_indexing_style


import mitsuba as mi

mi.set_variant('cuda_ad_rgb')


from mitsuba import ScalarTransform4f as T

class MitsubaDict:
    def __init__(self):
        pass
        


    def get_cube(self, albedo, sigma, sigma_t_scale, idx = None):    
        assert idx is not None, "Need to specify the index of the cube"

        # scale = np.array([ 0.008, 0.0005, 0.008 ])
        scale = np.array([ 0.008, 0.01, 0.008 ])

        cube_min = np.array([-0.004, 0.004, -0.004])

            
        # repeat sigma and albedo to have the same shape as the gridvolume 20, 20, 1, 3
        dim0, dim1, dim2 = int(scale[0] / 0.001), int(scale[1] / 0.001), int(scale[2] / 0.001)
        sigma_ = np.empty((dim0, dim1, dim2,3), dtype = np.float32)
        albedo_ = np.empty((dim0, dim1, dim2,3), dtype = np.float32)
        sigma_[:,:,:] = sigma
        albedo_[:,:,:] = albedo

        print("sigma shape", sigma_.shape)
        print("albedo shape", albedo_.shape)

        c_sigma = convert_data_to_C_indexing_style(sigma_, 3, (dim0, dim1, dim2))
        c_albedo = convert_data_to_C_indexing_style(albedo_, 3, (dim0, dim1, dim2))
        albedo_file_path = os.path.join("center_checker/meta", f"albedo_{idx}.vol")
        sigma_t_file_path = os.path.join("center_checker/meta", f"sigma_{idx}.vol")
        write_to_vol_file(albedo_file_path, c_albedo, 3, cube_min, np.array([dim0, dim1, dim2]), voxel_size=0.007)
        write_to_vol_file(sigma_t_file_path, c_sigma, 3, cube_min, np.array([dim0, dim1, dim2]), voxel_size=0.007)

        cube = {
            'type': 'obj',
            'filename': 'center_checker/meshes/Cube.obj',
            # 'bsdf': {'type': 'null'},
            # 'interior': {
            #     'type': 'homogeneous',
            #     'albedo': {
            #         'type': 'rgb',
            #         'value': list(albedo)
            #     },
            #     'sigma_t': {
            #         'type': 'rgb',
            #         'value': list(sigma)
            #     },
            #     'scale': sigma_t_scale
            # }
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'heterogeneous',
                'albedo': {
                    'type': 'gridvolume',
                    'filename': albedo_file_path,
                    'use_grid_bbox': True,
                    'filter_type': 'nearest'
                },
                'sigma_t': {
                    'type': 'gridvolume',
                    'filename': sigma_t_file_path,
                    'use_grid_bbox': True,
                    'filter_type': 'nearest'
                },
                'scale': sigma_t_scale,
                'phase': {
                    'type': 'hg',
                    'g': 0.4
                }
            }
        }
        print("sigma_t_scale", sigma_t_scale)

        return cube


    def get_checkerboard_scene_dict(self,albedo, sigma, sigma_t_scale,idx):
        cube = self.get_cube(albedo, sigma, sigma_t_scale, idx)


        base_albedo = np.array([0.9991, 0.9997, 0.999])
        base_sigma_scale = 25.0
        base_sigma = np.array([6.0, 9.0, 24.0]) / base_sigma_scale

        # Create the scene dictionary
        scene_dict = {
            'type': 'scene',
            'integrator': {'type': 'volpathmis'},
            'emitter': {'type': 'constant',
                        'radiance': {
                            'type': 'rgb',
                            'value': 1.0,
                            }
                        },
            'padding': {
                    'type': 'obj',
                    'filename': 'center_checker/meshes/padding.obj',
                    'bsdf': {'type': 'null'},
                    'interior': {
                        'type': 'homogeneous',
                        'albedo': {
                            'type': 'rgb',
                            'value': list(base_albedo)
                        },
                        'sigma_t': {
                            'type': 'rgb',
                            'value': list(base_sigma)
                        },
                        'scale': sigma_t_scale * base_sigma_scale
                    }
                },
            'base': {
                    'type': 'obj',
                    'filename': 'center_checker/meshes/base.obj',
                    'bsdf': {'type': 'null'},
                    'interior': {
                        'type': 'homogeneous',
                        'albedo': {
                            'type': 'rgb',
                            'value': list(base_albedo)
                        },
                        'sigma_t': {
                            'type': 'rgb',
                            'value': list(base_sigma)
                        },
                        'scale': sigma_t_scale * base_sigma_scale
                    }
            }
        }

        scene_dict['slab'] = cube
        return scene_dict


    def get_camera_dict(self):
        sensor_to_world = T.look_at(target=[0, 0, 0], origin=[0, 0.03, 0], up=[1, 0, 0])
        camera_dict ={
            'type': 'perspective',
            'fov': 45,
            'to_world': sensor_to_world,
            'film': {
                'type': 'hdrfilm',
                'width': 800, 
                'height': 800,
                'filter': {'type': 'box'}
            }
        }
        return camera_dict



class KM_Helper:
    def __init__(self):
        pass

    def get_mixtures(self):
        mixtures = np.array([[ 25. ,   0. ,   0. ,  50. ,  25.  ,   0. ],
            [  0. ,  25. ,  25. ,  50. ,   0.  ,   0. ],
            [  0. ,  25. ,   0. ,  50. ,  25.  ,   0. ],
            [  0. ,   0. ,  25. ,  50. ,  25.  ,   0. ],
            [ 25. ,  25. ,   0. ,   0. ,  50.  ,   0. ],
            [ 25. ,   0. ,  25. ,   0. ,  50.  ,   0. ],
            [ 25. ,   0. ,   0. ,  25. ,  50.  ,   0. ],
            [  0. ,  25. ,  25. ,   0. ,  50.  ,   0. ],
            [  0. ,  25. ,   0. ,  25. ,  50.  ,   0. ],
            [  0. ,   0. ,  25. ,  25. ,  50.  ,   0. ],
            [  0. ,  50. ,  25. ,   0. ,  25.  ,   0.],
            [  0. ,  50. ,   0. ,  25. ,  25.  ,   0.],
            [ 25. ,  25. ,  50. ,   0. ,   0.  ,   0.],
            [ 25. ,   0. ,  50. ,  25. ,   0.  ,   0.],
            [ 25. ,   0. ,  50. ,   0. ,  25.  ,   0.],
            [  0. ,  25. ,  50. ,  25. ,   0.  ,   0.],
            [  0. ,  25. ,  50. ,   0. ,  25.  ,   0.],
            [  0. ,   0. ,  50. ,  25. ,  25.  ,   0.],
            [ 25. ,  25. ,   0. ,  50. ,   0.  ,   0.],
            [ 25. ,   0. ,  25. ,  50. ,   0.  ,   0.],
            [ 50. ,  25. ,  25. ,   0. ,   0.  ,   0.],
            [ 50. ,  25. ,   0. ,  25. ,   0.  ,   0.],
            [ 50. ,  25. ,   0. ,   0. ,  25.  ,   0.],
            [ 50. ,   0. ,  25. ,  25. ,   0.  ,   0.],
            [ 50. ,   0. ,  25. ,   0. ,  25.  ,   0.],
            [ 50. ,   0. ,   0. ,  25. ,  25.  ,   0.],
            [ 25. ,  50. ,  25. ,   0. ,   0.  ,   0.],
            [ 25. ,  50. ,   0. ,  25. ,   0.  ,   0.],
            [ 25. ,  50. ,   0. ,   0. ,  25.  ,   0.],
            [  0. ,  50. ,  25. ,  25. ,   0.  ,   0.],
            [ 50. ,  50. ,   0. ,   0. ,   0.  ,   0.],
            [ 50. ,   0. ,  50. ,   0. ,   0.  ,   0.],
            [ 50. ,   0. ,   0. ,  50. ,   0.  ,   0.],
            [ 50. ,   0. ,   0. ,   0. ,  50.  ,   0.],
            [  0. ,  50. ,  50. ,   0. ,   0.  ,   0.],
            [  0. ,  50. ,   0. ,  50. ,   0.  ,   0.],
            [  0. ,  50. ,   0. ,   0. ,  50.  ,   0.],
            [  0. ,   0. ,  50. ,  50. ,   0.  ,   0.],
            [  0. ,   0. ,  50. ,   0. ,  50.  ,   0.],
            [  0. ,   0. ,   0. ,  50. ,  50.  ,   0.],
            [  0. ,   0. ,  75. ,  25. ,   0.  ,   0.],
            [  0. ,   0. ,  75. ,   0. ,  25.  ,   0.],
            [ 25. ,   0. ,   0. ,  75. ,   0.  ,   0.],
            [  0. ,  25. ,   0. ,  75. ,   0.  ,   0.],
            [  0. ,   0. ,  25. ,  75. ,   0.  ,   0.],
            [  0. ,   0. ,   0. ,  75. ,  25.  ,   0.],
            [ 25. ,   0. ,   0. ,   0. ,  75.  ,   0.],
            [  0. ,  25. ,   0. ,   0. ,  75.  ,   0.],
            [  0. ,   0. ,  25. ,   0. ,  75.  ,   0.],
            [  0. ,   0. ,   0. ,  25. ,  75.  ,   0.],
            [ 75. ,  25. ,   0. ,   0. ,   0.  ,   0.],
            [ 75. ,   0. ,  25. ,   0. ,   0.  ,   0.],
            [ 75. ,   0. ,   0. ,  25. ,   0.  ,   0.],
            [ 75. ,   0. ,   0. ,   0. ,  25.  ,   0.],
            [ 25. ,  75. ,   0. ,   0. ,   0.  ,   0.],
            [  0. ,  75. ,  25. ,   0. ,   0.  ,   0.],
            [  0. ,  75. ,   0. ,  25. ,   0.  ,   0.],
            [  0. ,  75. ,   0. ,   0. ,  25.  ,   0.],
            [ 25. ,   0. ,  75. ,   0. ,   0.  ,   0.],
            [  0. ,  25. ,  75. ,   0. ,   0.  ,   0.],
            [100. ,   0. ,   0. ,   0. ,   0.  ,   0.],
            [  0. , 100. ,   0. ,   0. ,   0.  ,   0.],
            [  0. ,   0. , 100. ,   0. ,   0.  ,   0.],
            [  0. ,   0. ,   0. , 100. ,   0.  ,   0.],
            [ 33.3,  33.3,  33.4,   0. ,   0.  ,   0.],
            [ 25. ,  25. ,  25. ,  25. ,   0.  ,   0.],
            [ 25. ,  25. ,  25. ,   0. ,  25.  ,   0.],
            [ 25. ,  25. ,   0. ,  25. ,  25.  ,   0.],
            [ 25. ,   0. ,  25. ,  25. ,  25.  ,   0.],
            [  0. ,  25. ,  25. ,  25. ,  25.  ,   0.]])
        mixtures = mixtures / 100.0

        return mixtures

    def RGB_ink_param_2_RGB(self, mix):
        '''
        input shape: (N,6)
        output shape: (N,3)
        
        '''
        INK = Ink(use_torch = False)

        assert mix.shape[-1] == 6, "Ink mixture should have 6 channels"


        scattering_RGB = INK.scattering_RGB
        absorption_RGB = INK.absorption_RGB
        print("I am using original ink absorption and scattering values")

        mix_K = mix @ absorption_RGB
        mix_S = mix @ scattering_RGB

        assert (mix_K >= 0.0).all() and (mix_S >= 0.0).all(), "albedo and scattering should be positive"
        
        return mix_K, mix_S




def render_mitsuba_scene(scene_dict, sensor_dict, filepath = None, set_spp = 256, file_name = None):
    assert filepath is not None and file_name is not None, "need to specify a filepath and file name to save the rendered images"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    sensor =  mi.load_dict(sensor_dict)
    print("successfully load sensor dict")

    
    scene_ref = mi.load_dict(scene_dict)

    print("successfully load scene dict")

    img = mi.render(scene_ref, sensor=sensor, spp=set_spp)
    name = os.path.join(filepath, file_name)
    mi.Bitmap(img).write(name+'.exr')
    mi.util.write_bitmap(name+'.png', img)



def render_checkerboard(mixtures):
    '''
    input: mixtures: (num_x * num_y, 3)
    '''

    num_x = 6
    num_y = 4

    assert mixtures.shape[0] == num_x * num_y, "Number of mixtures should be equal to the number of cubes in the checkerboard"

    # D = 0.007
    # H = W = D * 20


    D = 0.01
    H = W = 0.007 * 20

    miDict = MitsubaDict(H, W, D, num_x, num_y)
    km_helper = KM_Helper()

    absorption, scattering = km_helper.RGB_ink_param_2_RGB(mixtures)
    # t =  a + s
    sigma = scattering + absorption
    # albedo = s / t
    albedo = scattering / (sigma + 1e-8) # avoid division by zero


    


    sigma_t_scale = 50.0

    print("the used sigma_t_scale is: ", sigma_t_scale)


    scene_dict = miDict.get_checkerboard_scene_dict(albedo, sigma, sigma_t_scale)
    camera_dict = miDict.get_camera_dict()
    render_mitsuba_scene(scene_dict, camera_dict, filepath = "center_checker", set_spp = 4, file_name = "PBR_checkerboard")
    print("albedo shape:",  albedo.shape)

    # Reshape this array for a 6x4 grid
    colors_reshaped = albedo.reshape((4, 6, 3))
    colors_reshaped = np.flipud(colors_reshaped)

    # Create the figure and axis
    dpi = 100  # Display pixels per inch
    fig_size = 800 / dpi  # 800 pixels / 100 dpi = 8 inches
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.imshow(colors_reshaped, aspect='equal')

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("center_checker/albedo_checkerboard_plt.png")



def render_one_slab(mix, idx):
    km_helper = KM_Helper()
    absorption, scattering = km_helper.RGB_ink_param_2_RGB(mix)
    # t =  a + s
    sigma = scattering + absorption
    # albedo = s / t
    albedo = scattering / (sigma + 1e-8) # avoid division by zero
    sigma_t_scale = 50.0

    miDict = MitsubaDict()
    scene_dict = miDict.get_checkerboard_scene_dict(albedo, sigma, sigma_t_scale,idx)
    # import json
    # print_dict = json.dumps(scene_dict, indent=4)
    # #  print the scene dict in the json format
    # print("scene_dict", print_dict)
    # scene =  mi.load_dict(scene_dict)
    camera_dict = miDict.get_camera_dict()
    render_mitsuba_scene(scene_dict, camera_dict, filepath = "center_checker", set_spp = 512, file_name = f'slab_{idx}')

    



if __name__ == "__main__":
    km_helper = KM_Helper()
    miDict = MitsubaDict()
    # mixtures = km_helper.get_mixtures()

    mixtures = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
                         


    for idx, mix in enumerate(mixtures):
        render_one_slab(mix, idx)
        print("rendered slab", idx)

        if idx == 10:

            break
        #store the image
        # write the center 100 * 100 pixel average color to a file


    # scene = mi.load_dict({
    #     'type': 'scene',
    #     'integrator': {'type': 'volpathmis'},
    #     'sensor':  {
    #         'type': 'perspective',
    #         'to_world': T.look_at(
    #                         origin=(0, 0, 2),
    #                         target=(0, 0, 0),
    #                         up=(0, 1, 0)
    #                     ),
    #         'fov': 60,
    #         'film': {
    #             'type': 'hdrfilm',
    #             'width': 64,
    #             'height': 64,
    #             'rfilter': { 'type': 'gaussian' },
    #             'sample_border': True
    #         },
    #     },
    #     # scale: [ 0.008   0.0005  0.008 ] , min coord: [-0.004  0.004 -0.004]
    #     'slab': {
    #         'type': 'obj',
    #         'filename': 'center_checker/meshes/Cube.obj',
    #         'bsdf': {
    #             'type': 'diffuse',
    #             'reflectance': { 'type': 'rgb', 'value': (0.5, 0.5, 0.5) },
    #         }
    #     },
    #     'padding': {
    #         'type': 'obj',
    #         'filename': 'center_checker/meshes/padding.obj',
    #         'bsdf': {
    #             'type': 'diffuse',
    #             'reflectance': { 'type': 'rgb', 'value': (0.3, 0.3, 0.75) },
    #         },
    #     },
    #     'base': {
    #         'type': 'obj',
    #         'filename': 'center_checker/meshes/base.obj',
    #         'bsdf': {
    #             'type': 'diffuse',
    #             'reflectance': { 'type': 'rgb', 'value': (0.3, 0.3, 0.75) },
    #         },
    #     }
    # })
        
    # params =  mi.traverse(scene)
    # import drjit as dr
    # initial_vertex_positions = np.array(dr.unravel(mi.Point3f, params['slab.vertex_positions']))
    # g_pcd = o3d.geometry.PointCloud()
    # g_pcd.points = o3d.utility.Vector3dVector(initial_vertex_positions)
    # g_aabb = g_pcd.get_axis_aligned_bounding_box()
    # g_aabb_len = g_aabb.get_extent()
    # g_min = g_aabb.get_min_bound()
    # print("g_aabb_len", g_aabb_len)
    # print("g_min", g_min)
        