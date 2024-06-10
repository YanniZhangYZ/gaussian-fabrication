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

        

    def get_cube(self, scale, cube_center, cube_min, albedo, sigma, sigma_t_scale, idx = None):      
        assert idx is not None, "Need to specify the index of the cube"
            
        # repeat sigma and albedo to have the same shape as the gridvolume 20, 20, 1, 3
        dim0, dim1, dim2 = int(self.H / 0.001), int(self.W / 0.001), int(self.D / 0.0001)
        sigma_ = np.empty((dim0, dim1, dim2,3), dtype = np.float32)
        albedo_ = np.empty((dim0, dim1, dim2,3), dtype = np.float32)
        sigma_[:,:,:] = sigma
        albedo_[:,:,:] = albedo

        c_sigma = convert_data_to_C_indexing_style(sigma_, 3, (dim0, dim1, dim2))
        c_albedo = convert_data_to_C_indexing_style(albedo_, 3, (dim0, dim1, dim2))
        albedo_file_path = os.path.join("checker70/meta", f"albedo_{idx}.vol")
        sigma_t_file_path = os.path.join("checker70/meta", f"sigma_{idx}.vol")
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

    def get_background_paper(self):
        # The w and h of the background plane is 10% larger than the total width and height of the checkerboard
        bg_w = self.num_x * self.W * 1.2
        bg_h = self.num_y * self.H * 1.2

        # Center of the background plane
        bg_center_x = 0
        bg_center_y = 0
        # bg_center_z = -self.D*(1 + 0.2)  # behind the checkerboard
        bg_center_z = -self.D*(1 + 0.5)  # behind the checkerboard


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


    def get_checkerboard_scene_dict(self,albedo, sigma, sigma_t_scale):

        # Create a list of cubes
        assert albedo.shape[0] == sigma.shape[0] and albedo.shape[0] == self.cube_centers.shape[0], "Number of mixtures should be equal to the number of cubes in the checkerboard"
        
        cubes = []
        for i in range(albedo.shape[0]):
            cube_min = self.cube_centers[i] - self.cube_scale
            cubes.append(self.get_cube(self.cube_scale, self.cube_centers[i], cube_min, albedo[i], sigma[i], sigma_t_scale, idx = i))
        
    

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
        sensor_to_world = T.look_at(target=[0, 0, 0], origin=[0, 0, 4], up=[0, 1, 0])
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
    return img



def render_checkerboard(mixtures):
    '''
    input: mixtures: (num_x * num_y, 3)
    '''

    num_x = 10
    num_y = 7

    assert mixtures.shape[0] == num_x * num_y, "Number of mixtures should be equal to the number of cubes in the checkerboard"

    # D = 0.007
    D = 0.0005
    H = W = 0.007 * 20


    # D = 0.01
    # H = W = 0.007 * 20

    miDict = MitsubaDict(H, W, D, num_x, num_y)
    km_helper = KM_Helper()
    absorption, scattering = km_helper.RGB_ink_param_2_RGB(mixtures)
    # t =  a + s
    sigma = scattering + absorption
    # albedo = s / t
    albedo = scattering / (sigma + 1e-8) # avoid division by zero
    # sigma_t_scale = 50.0
    sigma_t_scale = 1000.0


    print("the used sigma_t_scale is: ", sigma_t_scale)


    scene_dict = miDict.get_checkerboard_scene_dict(albedo, sigma, sigma_t_scale)
    camera_dict = miDict.get_camera_dict()
    img = render_mitsuba_scene(scene_dict, camera_dict, filepath = "checker70", set_spp = 512, file_name = "checkerboard")
    print("albedo shape:",  albedo.shape)

    # Reshape this array for a 6x4 grid
    colors_reshaped = albedo.reshape((7, 10, 3))
    colors_reshaped = np.flipud(colors_reshaped)

    # Create the figure and axis
    dpi = 100  # Display pixels per inch
    fig_size = 800 / dpi  # 800 pixels / 100 dpi = 8 inches
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.imshow(colors_reshaped, aspect='equal')

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("checker70/albedo_checkerboard_plt.png")

    return img



def get_mixtures( nomarlize = True):
    mixtures = np.array([
        # line 1
        [ 25. ,   0. ,   0. ,  50. ,  25.  ,   0. ],
        [  0. ,  25. ,  25. ,  50. ,   0.  ,   0. ],
        [  0. ,  25. ,   0. ,  50. ,  25.  ,   0. ],
        [  0. ,   0. ,  25. ,  50. ,  25.  ,   0. ],
        [ 25. ,  25. ,   0. ,   0. ,  50.  ,   0. ],
        [ 25. ,   0. ,  25. ,   0. ,  50.  ,   0. ],
        [ 25. ,   0. ,   0. ,  25. ,  50.  ,   0. ],
        [  0. ,  25. ,  25. ,   0. ,  50.  ,   0. ],
        [  0. ,  25. ,   0. ,  25. ,  50.  ,   0. ],
        [  0. ,   0. ,  25. ,  25. ,  50.  ,   0. ],
        # line 2
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
        # line 3
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
        # line 4
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
        # line 5
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
        # line 6
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
        # line 7
        [100. ,   0. ,   0. ,   0. ,   0.  ,   0.],
        [  0. , 100. ,   0. ,   0. ,   0.  ,   0.],
        [  0. ,   0. , 100. ,   0. ,   0.  ,   0.],
        [  0. ,   0. ,   0. , 100. ,   0.  ,   0.],
        [  0. ,   0. ,   0. , 0. ,   100.  ,   0.],
        [ 25. ,  25. ,  25. ,  25. ,   0.  ,   0.],
        [ 25. ,  25. ,  25. ,   0. ,  25.  ,   0.],
        [ 25. ,  25. ,   0. ,  25. ,  25.  ,   0.],
        [ 25. ,   0. ,  25. ,  25. ,  25.  ,   0.],
        [  0. ,  25. ,  25. ,  25. ,  25.  ,   0.]])
    if nomarlize:   
        mixtures = mixtures / 100.0
        return mixtures
    else:
        return mixtures


def get_avg_rgb_centers():
    # in the current rendering, each slab is about 70*70 pixels
    # we compute the average RGB value of each slab in its center 40*40 pixels

   # Constants
    slab_size = 70
    matrix_dims = (7, 10)  # 7 rows, 10 columns
    center_position = (400, 400)  # Center of the entire matrix

    # Calculate the top-left corner of the matrix
    top_left_x = center_position[0] - (matrix_dims[1] * slab_size // 2)
    top_left_y = center_position[1] - (matrix_dims[0] * slab_size // 2)

    # Generate the centers of each square
    x_centers = top_left_x + np.arange(matrix_dims[1]) * slab_size + slab_size // 2
    y_centers = top_left_y + np.arange(matrix_dims[0]) * slab_size + slab_size // 2

    # Create a meshgrid of x and y coordinates
    X, Y = np.meshgrid(x_centers, y_centers)
    return X, Y


def compute_avg_rgb_from_mitsuba(img):
    from PIL import Image
    

    X, Y = get_avg_rgb_centers()
    img = mi.Bitmap(img)
    img = np.array(img)
    avg_rgbs = np.empty((7,10,3))
    for i in range(7):
        for j in range(10):
            x = X[i, j]
            y = Y[i, j]
            avg_rgb = img[y - 15: y + 15, x - 15: x + 15].mean(axis=(0, 1))
            # store the average RGB value as an image
            image = mi.Bitmap(img[y - 15: y + 15, x - 15: x + 15])
            mi.util.write_bitmap(f'checker70/avg_rgb/{i}_{j}.png', image)
            avg_rgbs[i, j] = avg_rgb
    
    #save the average RGB values to npy file
    np.save("checker70/avg_rgbs.npy", avg_rgbs)
    return 0


def compute_avg_rgb():
    from PIL import Image

    # Open an image file
    img = Image.open('checker70/checkerboard.png')

    # Convert image to a numpy array
    img = np.array(img)/255.0

    assert img.shape == (800, 800, 3), "Image should have shape (800, 800, 3), but has shape: {}".format(img.shape)
    

    X, Y = get_avg_rgb_centers()
    # img = mi.Bitmap(img)
    # img = np.array(img)
    avg_rgbs = np.empty((7,10,3))
    for i in range(7):
        for j in range(10):
            x = X[i, j]
            y = Y[i, j]
            avg_rgb = img[y - 15: y + 15, x - 15: x + 15].mean(axis=(0, 1))
            # store the average RGB value as an image
            image = mi.Bitmap(img[y - 15: y + 15, x - 15: x + 15])
            mi.util.write_bitmap(f'checker70/avg_rgb/{i}_{j}.png', image)
            avg_rgbs[i, j] = avg_rgb
    
    #save the average RGB values to npy file
    np.save("checker70/avg_rgbs.npy", avg_rgbs)
    return 0


def plot_avg_rgb():
    avg_rgbs = np.load("checker70/avg_rgbs.npy")
    # Create the figure and axis
    dpi = 100  # Display pixels per inch
    fig_size = 800 / dpi  # 800 pixels / 100 dpi = 8 inches
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.imshow(avg_rgbs, aspect='equal')

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig("checker70/avg_rgbs.png")


    # repeat each rgb color 70*70 times to form small squres
    avg_rgbs = np.repeat(avg_rgbs, 70, axis = 0)
    avg_rgbs = np.repeat(avg_rgbs, 70, axis = 1)
    image = mi.Bitmap(avg_rgbs)
    mi.util.write_bitmap('checker70/avg_rgb_bitmap.png', image)

    from PIL import Image
    image = Image.fromarray(np.uint8(avg_rgbs*255))

    # Save the image
    image.save('checker70/avg_rgb_PIL.png')


def rgb2lab2csv():
    import colorspacious
    import pandas as pd


    avg_rgbs = np.load("checker70/avg_rgbs.npy")

    lab_array = colorspacious.cspace_convert(avg_rgbs.reshape(70,3), "sRGB1", "CIELab")
    df = pd.DataFrame(lab_array, columns=["LAB_L", "LAB_A", "LAB_B"])
    df.to_csv("checker70/avg_labs.csv", index = False)

    df = pd.DataFrame(avg_rgbs.reshape(70,3), columns=["R", "G", "B"])
    df.to_csv("checker70/avg_rgbs.csv", index = False)


def mixture2csv():
    import pandas as pd
    mixtures = get_mixtures(nomarlize = False)
    assert mixtures.shape == (70,6), "Ink mixtures should have shape (70,6), but have shape: {}".format(mixtures.shape)
    mixtures = mixtures[:,:5]
    assert mixtures.shape[1] == 5, "Ink mixture should have 5 channels"

    # The original order is CMYKW, the new order is KCMYW
    mixtures = mixtures[:, [4, 0, 1, 2, 3]]

    # the value saved in the csv file should have the format, e.g. k25, c0, m0, y0, w75
    mixtures = mixtures.astype(int)
    mixtures = mixtures.astype(str)
    mixtures = np.core.defchararray.add(["k", "c", "m", "y", "w"],mixtures)
    df = pd.DataFrame(mixtures, columns=["K", "C", "M", "Y", "W"])
    df.to_csv("checker70/ratios.csv", index = False)


def convert_tattoo_lab_2_rgb():
    import colorspacious
    import pandas as pd

    df = pd.read_csv("checker70/tattoo_validation_meta/color_measurements.csv")
    lab_array = df.values[:,:3]
    rgb_array = colorspacious.cspace_convert(lab_array, "CIELab", "sRGB1")
    df = pd.DataFrame(rgb_array, columns=["R", "G", "B"])
    df.to_csv("checker70/tattoo_validation_meta/avg_rgbs.csv", index = False)
    df = pd.DataFrame(lab_array, columns=["LAB_L", "LAB_A", "LAB_B"])
    df.to_csv("checker70/tattoo_validation_meta/avg_labs.csv", index = False)


def plot_ink_spectra_validation():
    import pandas as pd

    df = pd.read_csv("checker70/tattoo_validation_meta/color_measurements_rgb2spec_256.csv")
    sepcs = df.values[:,3:]
    # Set the dimensions of the overall figure
    plt.figure(figsize=(20, 14))  # Width and height in inches

    # Iterate over the rows of the data array
    for i in range(70):
        ax = plt.subplot(7, 10, i + 1)  # Create a subplot in a 7x10 grid
        ax.plot(sepcs[i])  # Plot the ith row of data
        ax.set_title(f"Plot {i+1}", fontsize=8)  # Set a title for each subplot
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig('checker70/tattoo_validation_meta/specs_plots_rgb2spec_256.png', dpi=300)  # Save as PNG with high resolution
    

if __name__ == "__main__":

    mixtures = get_mixtures()

    mixtures = mixtures.reshape((7,10,6))

    mixtures = mixtures[::-1, :, :]

    mixtures = mixtures.reshape((70,6))

   
    assert (abs(mixtures.sum(axis = 1) - 1.0) < 1e-6).all(), "Ink mixtures should sum to 1, but have sum: {}".format(mixtures.sum(axis = 1))


    img = render_checkerboard(mixtures)
    # compute_avg_rgb_from_mitsuba(img)

    compute_avg_rgb()
    
    plot_avg_rgb() 

    rgb2lab2csv()
    mixture2csv()

    # convert_tattoo_lab_2_rgb()
    # plot_ink_spectra_validation()
