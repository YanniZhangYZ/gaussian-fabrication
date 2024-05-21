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
    def __init__(self, H, W, D, num_x, num_y):
        self.H = H
        self.W = W
        self.D = D
        self.num_x = num_x
        self.num_y = num_y

        self.cube_centers = self.get_cube_position(H, W, D, num_x, num_y)
        self.cube_scale =  np.array([H, W, D])

        


    def get_cube(self, scale, cube_center, cube_min, albedo, sigma, sigma_t_scale, idx = None, use_vol = False):      
        if not use_vol:
            cube = {
                'type': 'cube',
                'to_world': T.scale(scale).translate(cube_center/scale),
                'bsdf': {'type': 'null'},
                    'interior': {
                        'type': 'homogeneous',
                        'albedo': {
                            'type': 'rgb',
                            'value': list(albedo)
                        },
                        'sigma_t': {
                            'type': 'rgb',
                            'value': list(sigma)
                        },
                        'scale': sigma_t_scale
                    }
            }
        else:
            assert idx is not None, "Need to specify the index of the cube"
            
            # repeat sigma and albedo to have the same shape as the gridvolume 20, 20, 1, 3
            dim0, dim1, dim2 = int(self.H / 0.001), int(self.W / 0.001), int(self.D / 0.001)
            sigma_ = np.empty((dim0, dim1, dim2,3), dtype = np.float32)
            albedo_ = np.empty((dim0, dim1, dim2,3), dtype = np.float32)
            sigma_[:,:,:] = sigma
            albedo_[:,:,:] = albedo

            c_sigma = convert_data_to_C_indexing_style(sigma_, 3, (dim0, dim1, dim2))
            c_albedo = convert_data_to_C_indexing_style(albedo_, 3, (dim0, dim1, dim2))
            albedo_file_path = os.path.join("checkerboard/meta", f"albedo_{idx}.vol")
            sigma_t_file_path = os.path.join("checkerboard/meta", f"sigma_{idx}.vol")
            write_to_vol_file(albedo_file_path, c_albedo, 3, cube_min, np.array([dim0, dim1, dim2]), voxel_size=0.007)
            write_to_vol_file(sigma_t_file_path, c_sigma, 3, cube_min, np.array([dim0, dim1, dim2]), voxel_size=0.007)

            cube = {
                'type': 'cube',
                'to_world': T.scale(scale).translate(cube_center/scale),
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

    def get_km_cube(self, scale, cube_center, km_rgb):      
        cube = {
            'type': 'cube',
            'to_world': T.scale(scale).translate(cube_center/scale),
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                            'type': 'rgb',
                            'value': list (km_rgb)
                        }
                    }
        }
        return cube

    def get_spectra_cube(self, scale, cube_center, albedo_file_path, sigma_t_file_path, sigma_t_scale):
        cube = {
                'type': 'cube',
                'to_world': T.scale(scale).translate(cube_center/scale),
                'bsdf': {'type': 'null'},
                    'interior': {
                        'type': 'homogeneous',
                        'albedo': {
                            'type': 'spectrum',
                            'filename': albedo_file_path,
                        },
                        'sigma_t': {
                            'type': 'spectrum',
                            'filename': sigma_t_file_path,
                        },
                        'scale': sigma_t_scale,
                        'phase': {
                            'type': 'hg',
                            'g': 0.4
                        }
                    }
            }
        return cube

    def get_background_paper(self):
        # The w and h of the background plane is 10% larger than the total width and height of the checkerboard
        bg_w = self.num_x * self.W * 1.2
        bg_h = self.num_y * self.H * 1.2

        # Center of the background plane
        bg_center_x = 0
        bg_center_y = 0
        bg_center_z = -self.D*(1 + 0.2)  # behind the checkerboard

        scale = np.array([bg_w, bg_h, self.D])
        center = np.array([bg_center_x, bg_center_y, bg_center_z])

        plane = {
            'type': 'cube',
            'to_world': T.scale(scale).translate(center/scale),
            'bsdf': {'type': 'diffuse',
                    'reflectance': {
                            'type': 'rgb',
                            'value': [1.0, 1.0, 1.0]
                            # "type" : "checkerboard",
                            # "color0" : {
                            #     "type" : "rgb",
                            #     "value" : [1, 1, 1]
                            # },
                            # "color1" : {
                            #     "type" : "rgb",
                            #     "value" : [0.5, 0.5, 0.5]
                            # },
                            # "to_uv": T.scale(16),
                        }
                    }
        }
        return plane

    def get_cube_position(self, H, W, D, num_x, num_y):
        H = H * 2
        W = W * 2
        D = D * 2
    
        # Calculate the total width and height of the entire checkerboard
        total_width = num_x * W
        total_height = num_y * H

        # Calculate the starting positions (bottom left corner of the checkerboard)
        start_x = -total_width / 2 + W / 2
        start_y = -total_height / 2 + H / 2

        # Create arrays for x and y coordinates
        x_coords = start_x + np.arange(num_x) * W
        y_coords = start_y + np.arange(num_y) * H

        # Create a meshgrid of x and y coordinates
        X, Y = np.meshgrid(x_coords, y_coords)

        # Flatten the arrays to get a list of coordinates
        x_flat = X.flatten()
        y_flat = Y.flatten()

        # Set all z coordinates to half the depth
        z_flat = np.full_like(x_flat, D / 2)

        # Combine x, y, and z coordinates into one array
        cube_centers = np.vstack((x_flat, y_flat, z_flat)).T

        return cube_centers


    def get_checkerboard_scene_dict(self,albedo, sigma, sigma_t_scale, use_vol = False):

        # Create a list of cubes
        assert albedo.shape[0] == sigma.shape[0] and albedo.shape[0] == self.cube_centers.shape[0], "Number of mixtures should be equal to the number of cubes in the checkerboard"
        
        cubes = []
        for i in range(albedo.shape[0]):
            cube_min = self.cube_centers[i] - self.cube_scale
            cubes.append(self.get_cube(self.cube_scale, self.cube_centers[i], cube_min, albedo[i], sigma[i], sigma_t_scale, idx = i, use_vol = use_vol))
        
    

        # Create the scene dictionary
        scene_dict = {
            'type': 'scene',
            'integrator': {'type': 'volpathmis'},
            'emitter': {'type': 'constant',
                        'radiance': {
                            'type': 'rgb',
                            'value': 1.0,
                            }
                        }
        }

        scene_dict['background'] = self.get_background_paper()

        for idx, c in enumerate(cubes):
            scene_dict[f'cube_{idx}'] = c
        

        return scene_dict

    def get_KM_checkerboard_scene_dict(self, km_rgb):
        print(" get KM checkerboard scene dict")
        # Create a list of cubes
        cubes = []
        for i in range(km_rgb.shape[0]):
            cubes.append(self.get_km_cube(self.cube_scale, self.cube_centers[i], km_rgb[i]))

        # Create the scene dictionary
        scene_dict = {
            'type': 'scene',
            'integrator': {'type': 'volpathmis'},
            'emitter': {'type': 'constant',
                        'radiance': {
                            'type': 'rgb',
                            'value': 1.0,
                            }
                        }
        }

        scene_dict['background'] = self.get_background_paper()

        for idx, c in enumerate(cubes):
            scene_dict[f'cube_{idx}'] = c
        

        return scene_dict

    def get_spectra_checkerboard_scene_dict(self, albedo_file_path, sigma_t_file_path, sigma_t_scale):
        cubes = []
        for i in range(len(albedo_file_path)):
            cubes.append(self.get_spectra_cube(self.cube_scale, self.cube_centers[i], albedo_file_path[i], sigma_t_file_path[i], sigma_t_scale))

        # Create the scene dictionary
        scene_dict = {
            'type': 'scene',
            'integrator': {'type': 'volpathmis'},
            'emitter': {'type': 'constant',
                        'radiance': {
                            'type': 'rgb',
                            'value': 1.0,
                            }
                        }
        }

        scene_dict['background'] = self.get_background_paper()

        for idx, c in enumerate(cubes):
            scene_dict[f'cube_{idx}'] = c
        

        return scene_dict

    def get_camera_dict(self):
        sensor_to_world = T.look_at(target=[0, 0, 0], origin=[0, 0, 3], up=[0, 1, 0])
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

    def RGB_ink_param_2_RGB(self, mix,scale_factor = 20.0, use_vol = False):
        '''
        input shape: (N,6)
        output shape: (N,3)
        
        '''
        INK = Ink(use_torch = False)

        assert mix.shape[-1] == 6, "Ink mixture should have 6 channels"

        if not use_vol:
            albedo_RGB = INK.albedo_RGB
            sigma_RGB = INK.sigma_RGB/ scale_factor
            scattering_RGB = albedo_RGB * sigma_RGB
            absorption_RGB = sigma_RGB - scattering_RGB
        else:
            scattering_RGB = INK.scattering_RGB
            absorption_RGB = INK.absorption_RGB
            print("I am using original ink absorption and scattering values")

        mix_K = mix @ absorption_RGB
        mix_S = mix @ scattering_RGB

        assert (mix_K >= 0.0).all() and (mix_S >= 0.0).all(), "albedo and scattering should be positive"
        
        return mix_K, mix_S


    def get_wavelength_KM_spectra_4_mitsuba(self, mix):
        # linear interpolation of the absorption and scattering values
        INK = Ink(use_torch = False)
        assert mix.shape[-1] == 6, "Ink mixture should have 6 channels"
        assert (mix >= 0.0).all(), "Ink mixture should be positive"
        assert (mix <= 1.0 + 1e-1).all(), "Ink mixture should be less than 1.0. {} > 1.0".format(mix.max())

        N, C = mix.shape

        mix_K = mix @ INK.absorption_matrix
        mix_S = mix @ INK.scattering_matrix


        # manual linear interpolation, for each element in the row vector of R_mix, we compute the mean between the two values and insert it in between
        mix_K = np.insert(mix_K, np.arange(1, mix_K.shape[1]), (mix_K[:,1:] + mix_K[:,:-1]) / 2, axis=1)
        
        # check the length of the row vector, if the lenght is smaller than the length of INK.w_380_5_780, append zeros
        mix_K = np.pad(mix_K, ((0,0),(0, INK.w_num - mix_K.shape[1])))


        # manual linear interpolation, for each element in the row vector of R_mix, we compute the mean between the two values and insert it in between
        mix_S = np.insert(mix_S, np.arange(1, mix_S.shape[1]), (mix_S[:,1:] + mix_S[:,:-1]) / 2, axis=1)
        
        # check the length of the row vector, if the lenght is smaller than the length of INK.w_380_5_780, append zeros
        mix_S = np.pad(mix_S, ((0,0),(0, INK.w_num - mix_S.shape[1])))


        wavelength = INK.w_380_5_780

        assert mix_K.shape == (N, INK.w_num), "mix_K shape should be (N, INK.w_num)"
        assert mix_S.shape == (N, INK.w_num), "mix_S shape should be (N, INK.w_num)"
        assert (mix_K[:,71] == 0.0).all(), "The 71th element should be 0.0"
        assert (mix_S[:,71] == 0.0).all(), "The 71th element should be 0.0"

        # # plot mix_K[0,:] and mix_S[0,:]
        # plt.figure()
        # plt.plot(wavelength, mix_K[6,:], label = "mix_K")
        # # plt.plot(wavelength, mix_S[3,:], label = "mix_S")
        # plt.legend()
        # plt.savefig("checkerboard/mix_K_S.png")

        return mix_K[:,:71], mix_S[:,:71], wavelength[:71]





    def get_wavelength_KM_RGB(self, mix):
        INK = Ink(use_torch = False)
        assert mix.shape[-1] == 6, "Ink mixture should have 6 channels"
        assert (mix >= 0.0).all(), "Ink mixture should be positive"
        assert (mix <= 1.0 + 1e-1).all(), "Ink mixture should be less than 1.0. {} > 1.0".format(mix.max())

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
        sRGB = np.where(sRGB <= 0.0031308,
                    12.92 * sRGB,
                    1.055 * np.power(sRGB, 1 / 2.4) - 0.055)
        sRGB = np.clip(sRGB,0.0,1.0)
        assert sRGB.shape == (N, 3), "sRGB shape should be (N,3)"
        return sRGB




def render_mitsuba_scene(scene_dict, sensor_dict, filepath = None, set_spp = 256, file_name = "PBR_checkerboard"):
    assert filepath is not None, "need to specify a filepath to save the rendered images"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    sensor =  mi.load_dict(sensor_dict)
    
    scene_ref = mi.load_dict(scene_dict)

    img = mi.render(scene_ref, sensor=sensor, spp=set_spp)
    name = os.path.join(filepath, file_name)
    mi.Bitmap(img).write(name+'.exr')
    mi.util.write_bitmap(name+'.png', img)



def render_checkerboard(mixtures, use_vol = False, get_KM = False, use_KM_spectra = False):
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




    if get_KM:
        km_rgb = km_helper.get_wavelength_KM_RGB(mixtures)
        scene_dict = miDict.get_KM_checkerboard_scene_dict(km_rgb)
        camera_dict = miDict.get_camera_dict()
        render_mitsuba_scene(scene_dict, camera_dict, filepath = "checkerboard", set_spp = 16, file_name = "KM_checkerboard_rendered")
        

        # Reshape this array for a 6x4 grid
        colors_reshaped = km_rgb.reshape((4, 6, 3))
        colors_reshaped = np.flipud(colors_reshaped)

        # Create the figure and axis
        dpi = 100  # Display pixels per inch
        fig_size = 800 / dpi  # 800 pixels / 100 dpi = 8 inches
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
        ax.imshow(colors_reshaped, aspect='equal')

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig("checkerboard/KM_checkerboard_plt.png")
        
        return 0


    if use_KM_spectra:
        sigma_t_scale = 150.0
        absorption, scattering, wavelength = km_helper.get_wavelength_KM_spectra_4_mitsuba(mixtures)
        sigma = scattering + absorption
        # albedo = s / t
        albedo = scattering / (sigma + 1e-8) # avoid division by zero

        albedo_file_path = []
        sigma_t_file_path = []

        for i in range (mixtures.shape[0]):
            a_path = os.path.join(f"checkerboard/meta/albedo_spectra_{i}.spd")
            s_path = os.path.join(f"checkerboard/meta/sigma_spectra_{i}.spd")
            mi.spectrum_to_file(a_path, wavelength, albedo[i])
            mi.spectrum_to_file(s_path, wavelength, sigma[i])
            albedo_file_path.append(a_path)
            sigma_t_file_path.append(s_path)


        scene_dict = miDict.get_spectra_checkerboard_scene_dict(albedo_file_path, sigma_t_file_path, sigma_t_scale)
        camera_dict = miDict.get_camera_dict()
        render_mitsuba_scene(scene_dict, camera_dict, filepath = "checkerboard", set_spp = 512, file_name = "Spectra_checkerboard")

        
        return 0


    scale_factor =  30.0
    absorption, scattering = km_helper.RGB_ink_param_2_RGB(mixtures, scale_factor, use_vol = use_vol)
    # t =  a + s
    sigma = scattering + absorption
    # albedo = s / t
    albedo = scattering / (sigma + 1e-8) # avoid division by zero


    


    if not use_vol:
        sigma_t_scale = 50.0 * scale_factor
        assert (albedo <= 1.0).all(), "Albedo should smaller than 1, but got max albedo: {}".format(albedo.max())
        assert (sigma <= 1.0).all(), "Sigma should smaller than 1, but got max sigma: {}".format(sigma.max())
    else:
        sigma_t_scale = 50.0
        # sigma_t_scale = 10.0

    print("the used sigma_t_scale is: ", sigma_t_scale)


    scene_dict = miDict.get_checkerboard_scene_dict(albedo, sigma, sigma_t_scale, use_vol = use_vol)
    camera_dict = miDict.get_camera_dict()
    render_mitsuba_scene(scene_dict, camera_dict, filepath = "checkerboard", set_spp = 512, file_name = "PBR_checkerboard")
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
    plt.savefig("checkerboard/albedo_checkerboard_plt.png")




def test_CMY_33():
    mixtures = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]) / 3.0
    # mixtures = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

    INK = Ink(use_torch = False)

    albedo_RGB = INK.albedo_RGB
    sigma_RGB = INK.sigma_RGB
    scattering_RGB = INK.scattering_RGB
    absorption_RGB = INK.absorption_RGB
    

    assert (abs(albedo_RGB * sigma_RGB - scattering_RGB) < 1e-6).all(), "scattering should be equal to albedo * sigma"
    assert (abs(( 1.0 - albedo_RGB ) * sigma_RGB - absorption_RGB)< 1e-6) .all(), "absorption should be equal to sigma - scattering"


    # compute from sigma_rgb and albedo_rgb
    as_absorption =  mixtures @ (( 1.0 - albedo_RGB ) * sigma_RGB)
    as_scattering = mixtures @ (albedo_RGB * sigma_RGB)

    # compute from absorption and scattering
    mix_K = mixtures @ absorption_RGB
    mix_S = mixtures @ scattering_RGB

    assert (abs(as_absorption - mix_K) < 1e-6).all(), "absorption should be equal to sigma - scattering"
    assert (abs(as_scattering - mix_S) < 1e-6).all(), "scattering should be equal to albedo * sigma"


    as_albedo = as_scattering / (as_absorption + as_scattering)
    mix_albedo = mix_S / (mix_K + mix_S)

    as_sigma = as_absorption + as_scattering
    mix_sigma = mix_K + mix_S

    assert (abs(as_albedo - mix_albedo) < 1e-6).all(), "albedo should be equal to scattering / (absorption + scattering)"
    

    print(as_albedo)
    print(mix_albedo)

    print(as_sigma)
    print(mix_sigma)

   # Create a 100x100 image with this RGB color
    image_array = np.tile(as_albedo * 255, (100, 100, 1))

    # Convert the numpy array to a PIL Image
    from PIL import Image
    image = Image.fromarray(np.uint8(image_array))

    # Save the image
    image.save('checkerboard/33CMY.png')


    # check RGB color given albedo from scattering-aware 3d print equation (4)

    Cs = 0.04526
    ak = np.array([0.065773, 0.201198, 0.279264, 0.251997, 0.201767])
    bk = [1.569383, 6.802855, 28.61815, 142.0079, 1393.165]
    
    r_exp_bk = np.array([mix_albedo[0][0]**b for b in bk])
    g_exp_bk = np.array([mix_albedo[0][1]**b for b in bk])
    b_exp_bk = np.array([mix_albedo[0][2]**b for b in bk])

    r = Cs + (1.0 - Cs) * np.sum(ak * r_exp_bk)
    g = Cs + (1.0 - Cs) * np.sum(ak * g_exp_bk)
    b = Cs + (1.0 - Cs) * np.sum(ak * b_exp_bk)
    rgb = np.array([r,g,b])*255.0

    print(rgb)

    image_array = np.tile(rgb, (100, 100, 1))

    # Convert the numpy array to a PIL Image
    from PIL import Image
    image = Image.fromarray(np.uint8(image_array))

    # Save the image
    image.save('checkerboard/33CMY_eq4.png')



if __name__ == "__main__":

    # randome 24 ink mixtures, each has 6 channels and sum to 1
    mixtures = np.empty((24, 6))
    mixtures[0] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # C
    mixtures[1] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) # M
    mixtures[2] = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # Y
    mixtures[3] = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]) # CM
    mixtures[4] = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0]) # CY
    mixtures[5] = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]) # MY

    mixtures[6] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) # K
    mixtures[7] = np.array([0.0, 0.0, 0.0, 0.8, 0.2, 0.0]) # gray, 20% W
    mixtures[8] = np.array([0.0, 0.0, 0.0, 0.6, 0.4, 0.0]) # gray, 40% W
    mixtures[9] = np.array([0.0, 0.0, 0.0, 0.4, 0.6, 0.0]) # gray, 60% W
    mixtures[10] = np.array([0.0, 0.0, 0.0, 0.2, 0.8, 0.0]) # gray, 80% W
    mixtures[11] = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) # W


    mixtures[12] = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0]) # 50% C, 50% W
    mixtures[13] = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.5]) # 50% C, 50% T
    mixtures[14] = np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0]) # 50% M, 50% W
    mixtures[15] = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.5]) # 50% M, 50% T
    mixtures[16] = np.array([0.0, 0.0, 0.5, 0.0, 0.5, 0.0]) # 50% Y, 50% W
    mixtures[17] = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.5]) # 50% Y, 50% T


    mixtures[18] = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) / 3.0 # 33% CMY
    mixtures[19] = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0]) / 4.0 # 25% CMYW
    mixtures[20] = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0]) / 4.0 # 25% CMYT
    mixtures[21] = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) / 3.0 # 33% CYW
    mixtures[22] = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0]) / 3.0 # 33% MYW
    mixtures[23] = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0]) / 3.0 # 33% CMW

   
    assert (abs(mixtures.sum(axis = 1) - 1.0) < 1e-6).all(), "Ink mixtures should sum to 1, but have sum: {}".format(mixtures.sum(axis = 1))

    render_checkerboard(mixtures)

    # render_checkerboard(mixtures, use_vol = True)
    # render_checkerboard(mixtures, get_KM = True)
    # render_checkerboard(mixtures, use_KM_spectra = True)


