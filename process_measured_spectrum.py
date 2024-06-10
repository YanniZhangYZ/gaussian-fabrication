import numpy as np
import matplotlib.pyplot as plt
from ink_intrinsics import Ink




def get_r_mix_2_rgb(R_mix):
    INK = Ink(use_torch = False)
    N = R_mix.shape[0]
    # the new R_mix contains 3 parts: the original R_mix, the mean of the original R_mix, and zeros
    R_mix = np.concatenate([R_mix, (R_mix[:,1:] + R_mix[:,:-1]) / 2], axis=1)
    R_mix = np.concatenate([R_mix, np.zeros((R_mix.shape[0], INK.w_num - R_mix.shape[1]))], axis=1)
    assert (R_mix[:,71] == 0.0).all(), "The 71th element should be 0.0"
    
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

    print("XYZ: ",XYZ.shape)


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

def get_albedo_validation(mix):
    INK = Ink(use_torch = False)

    assert mix.shape[-1] == 6, "Ink mixture should have 6 channels"

    scattering_RGB = INK.scattering_RGB
    absorption_RGB = INK.absorption_RGB
    print("I am using original ink absorption and scattering values")

    mix_K = mix @ absorption_RGB
    mix_S = mix @ scattering_RGB

    assert (mix_K >= 0.0).all() and (mix_S >= 0.0).all(), "albedo and scattering should be positive"
    # t =  a + s
    sigma =  mix_K + mix_S
    # albedo = s / t
    albedo = mix_S / (sigma + 1e-8) # avoid division by zero
    return albedo



def xyz2rgb(XYZ):
    N = XYZ.shape[0]
    XYZ = XYZ.T

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




def plot_color_grid(colors, save_path=None):

    # repeat each rgb color 70*70 times to form small squres
    colors = np.repeat(colors, 20, axis = 0)
    colors = np.repeat(colors, 20, axis = 1)
    from PIL import Image
    image = Image.fromarray(np.uint8(colors*255))
    # Save the image
    image.save(save_path)

def plot_ink_spectra(sepcs,save_path=None):
    assert sepcs.shape[0] == 70, "The shape of the data array should be (70, 351)"
    # Set the dimensions of the overall figure
    plt.figure(figsize=(70, 4))  # Width and height in inches

    # Iterate over the rows of the data array
    for i in range(70):
        ax = plt.subplot(2, 35, i + 1)  # Create a subplot in a 7x10 grid
        ax.plot(sepcs[i])  # Plot the ith row of data
        ax.set_title(f"Plot {i+1}", fontsize=8)  # Set a title for each subplot
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig(save_path, dpi=300)  # Save as PNG with high resolution
    

def mixture2csv():
    import pandas as pd
    df = pd.read_csv("measured_spectrum/ratio_2_35.csv", sep=',')
    sepcs = df.values
    print(type(sepcs[0,0]))
    mixtures = np.empty((2,35,5))
    for i in range(2):
        for j in range(35):
            
            mixtures[i,j] = np.fromstring(sepcs[i,j], dtype=int, sep=',')

    mixtures = mixtures.reshape(-1,5)
    # append 0 to the end of each row
    temp_mixtures = np.concatenate([mixtures, np.zeros((70,1))], axis=1)
    albedo_validation = get_albedo_validation(temp_mixtures)
    plot_color_grid(albedo_validation.reshape((2,35,3)), save_path='measured_spectrum/albedo_validation.png')
    # The original order is CMYKW, the new order is KCMYW
    mixtures = mixtures[:, [3, 0, 1, 2, 4]]


    # the value saved in the csv file should have the format, e.g. k25, c0, m0, y0, w75
    mixtures = mixtures.astype(int)
    mixtures = mixtures.astype(str)
    mixtures = np.core.defchararray.add(["k", "c", "m", "y", "w"],mixtures)
    df = pd.DataFrame(mixtures, columns=["K", "C", "M", "Y", "W"])
    df.to_csv("measured_spectrum/ratios.csv", index = False)

    print("saved the ratios to measured_spectrum/ratios.csv")




if __name__ == "__main__":

    #  save the ratio files
    mixture2csv()

    # Load the file
    file_path = 'measured_spectrum/raw_data/chart_v2.3_M0.txt'

    # Initialize lists to hold the LAB values and spectral data
    lab_colors = []
    xyz_colors = []
    spectral_data = []


    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the header to get the indices of LAB and spectral data
    header_line = lines.index('BEGIN_DATA_FORMAT\n') + 1
    data_start_line = lines.index('BEGIN_DATA\n') + 1

    # Extract the header names to determine indices
    header = lines[header_line].strip().split('\t')
    lab_indices = [header.index('LAB_L'), header.index('LAB_A'), header.index('LAB_B')]
    xyz_indices = [header.index('XYZ_X'), header.index('XYZ_Y'), header.index('XYZ_Z')]
    spectral_indices = [i for i, col in enumerate(header) if col.startswith('SPECTRAL_NM')]

    # Process each data line
    for line in lines[data_start_line:data_start_line + 72]:
        columns = line.strip().split('\t')
        # Extract LAB values
        lab = [float(columns[i]) for i in lab_indices]
        lab_colors.append(lab)
        # Extract XYZ values
        xyz = [float(columns[i]) for i in xyz_indices]
        xyz_colors.append(xyz)
        # Extract spectral data
        spectral = [float(columns[i]) for i in spectral_indices]
        spectral_data.append(spectral)

    # Convert lists to numpy arrays
    lab_colors = np.array(lab_colors)
    xyz_colors = np.array(xyz_colors)
    spectral_data = np.array(spectral_data)



    # visualize the orginal colors
    rgb = get_r_mix_2_rgb(spectral_data)
    colors_reshaped = rgb.reshape((36, 2, 3))
    colors_reshaped = np.transpose(colors_reshaped, (1, 0, 2))
    colors_reshaped = colors_reshaped.reshape((-1,3))
    colors_reshaped = colors_reshaped.reshape((2,36,3))

    plot_color_grid(colors_reshaped, save_path='measured_spectrum/2_36.png')

    # substitute the 9th spectrum with the mean of last two spectra (white)
    spectral_data[9] = (spectral_data[70] + spectral_data[71]) / 2

    # remove the 35th and 71th spectra and store in new_spec
    new_spec = np.delete(spectral_data, [70,71], axis=0)
    new_spec = new_spec.reshape(35, 2, 36)
    new_spec = np.transpose(new_spec, (1, 0, 2))
    new_spec = new_spec.reshape((-1,36))
    print(new_spec.shape)
    # visualize the orginal colors
    rgb = get_r_mix_2_rgb(new_spec)
    colors_reshaped = rgb.reshape((2, 35, 3))
    plot_color_grid(colors_reshaped, save_path='measured_spectrum/2_35.png')
    plot_ink_spectra(new_spec,save_path='measured_spectrum/spectra.png')


    cmykw_spec = new_spec[35:40]
    cmykw_rgb = get_r_mix_2_rgb(cmykw_spec)
    plot_color_grid(cmykw_rgb.reshape((1,5,3)), save_path='measured_spectrum/cmykw.png')


    import colorspacious as cs
    lab_colors[9] = (lab_colors[70] + lab_colors[71]) / 2
    new_lab = np.delete(lab_colors, [70,71], axis=0)
    new_lab = new_lab.reshape(35, 2, 3)
    new_lab = np.transpose(new_lab, (1, 0, 2))
    new_lab = new_lab.reshape((-1,3))
    print(new_lab.shape)
    rgbs = cs.cspace_convert(new_lab, "CIELab", "sRGB1")
    rgbs = np.clip(rgbs, 0, 1)
    plot_color_grid(rgbs.reshape((2, 35, 3)), save_path='measured_spectrum/2_35_lab2rgb.png')


    import csv
    # Open a CSV file to write
    with open('measured_spectrum/color_measurements.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Create the header
        header = ['LAB_L', 'LAB_A', 'LAB_B'] + [f'SPECTRAL_NM{w}' for w in range(380, 731, 10)]
        writer.writerow(header)
        
        # Iterate over each row of lab and specs
        for lab_row, spec_row in zip(new_lab, new_spec):
            # Combine lab row and spec row
            full_row = np.concatenate((lab_row, spec_row))
            # Write the combined row to the CSV
            writer.writerow(full_row)





