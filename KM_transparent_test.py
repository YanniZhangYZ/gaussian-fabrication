import numpy as np
import os
import matplotlib.pyplot as plt
from ink_intrinsics import Ink




def get_wavelength_KM_RGB(mix):
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
    return sRGB, R_mix



def get_mixtures():
    '''
        This function proves that KM( mix ) - KM( mix with transparent ink) is around error e-9
    '''
    # mix cmykw accordingly with transparent ink in 25% 50% 75% 100%
    mixtures = np.array([[25, 0.0, 0.0, 0.0, 0.0, 75], # 75% transparent of Cyan
                         [50, 0.0, 0.0, 0.0, 0.0, 50],
                         [75, 0.0, 0.0, 0.0, 0.0, 25],
                         [75, 0.0, 0.0, 0.0, 0.0, 0.0], # alpha 75
                         [100, 0.0, 0.0, 0.0, 0.0, 0.0],

                         [0.0, 25, 0.0, 0.0, 0.0, 75], # 75% transparent of Magenta
                         [0.0, 50, 0.0, 0.0, 0.0, 50],
                         [0.0, 75, 0.0, 0.0, 0.0, 25],
                         [0.0, 75, 0.0, 0.0, 0.0, 0.0], # alpha 75
                         [0.0, 100, 0.0, 0.0, 0.0, 0.0],

                         [0.0, 0.0, 25, 0.0, 0.0, 75], # 75% transparent of Yellow
                         [0.0, 0.0, 50, 0.0, 0.0, 50],
                         [0.0, 0.0, 75, 0.0, 0.0, 25],
                         [0.0, 0.0, 75, 0.0, 0.0, 0.0], # alpha 75
                         [0.0, 0.0, 100, 0.0, 0.0, 0.0],
                         
                         [0.0, 0.0, 0.0, 25, 0.0, 75], # 75% transparent of Key
                         [0.0, 0.0, 0.0, 50, 0.0, 50],
                         [0.0, 0.0, 0.0, 75, 0.0, 25],
                         [0.0, 0.0, 0.0, 75, 0.0, 0.0], # alpha 75
                         [0.0, 0.0, 0.0, 100, 0.0, 0.0],
                         
                         [0.0, 0.0, 0.0, 0.0, 25, 75], # 75% transparent of White
                         [0.0, 0.0, 0.0, 0.0, 50, 50],
                         [0.0, 0.0, 0.0, 0.0, 75, 25],
                         [0.0, 0.0, 0.0, 0.0, 75, 0.0], # alpha 75
                         [0.0, 0.0, 0.0, 0.0, 100, 0.0]])
    xticklabels = ['0.25 ink', '0.5 ink', '0.75 ink', '0.75 alpha','1.0 ink']

    return mixtures/100.0, 5, 5, xticklabels


def get_mixtures2():
    '''
    This function proves that KM( alpha * mixture) - KM(mixture) is around error e-17
    '''
      # alpha 25% 50% 75% 100%
    # generate 5 random mixtures of CMYKW that sum up to 100
    random_mixtures = np.random.rand(5, 5)
    random_mixtures = random_mixtures / random_mixtures.sum(axis=1)[:, np.newaxis] * 100
    # append 0 to the end of each row
    mixtures = np.concatenate([random_mixtures, np.zeros((5, 1))], axis=1)
    # repeat each mixture 4 times and reshape the array to (5,4,6)
    mixtures = np.repeat(mixtures, 4, axis=0).reshape(5, 4, 6)

    assert (mixtures.sum(axis=2)== 100).all, "The sum of each mixture should be 100"

    # times 25% 50% 75% 100% to each row
    alpha = np.array([25, 50, 75, 100])/100.0
    mixtures = mixtures * alpha[np.newaxis, :, np.newaxis]
    mixtures = mixtures.reshape(20, 6)

    xticklabels = ['0.25 alpha', '0.5 alpha', '0.75 alpha','1.0 alpha']

    return mixtures/100.0, 5, 4, xticklabels

def get_mixtures3():
    '''
    This prove KM( a * mix1 + b * mix2) - KM(mix1 + mix2) is around error e-6
    '''    

    # generate 2 random mixtures of CMYKW that sum up to 100
    random_mixtures = np.random.rand(2, 5)
    random_mixtures = random_mixtures / random_mixtures.sum(axis=1)[:, np.newaxis] * 100
    # append 0 to the end of each row
    mixtures = np.concatenate([random_mixtures, np.zeros((2, 1))], axis=1)
    mix1 = mixtures[0]
    mix2 = mixtures[1]
    # # generate 20 combinations of size (20,2), each sum up to 1
    # alpha = np.random.rand(20, 2)
    # alpha = alpha / alpha.sum(axis=1)[:, np.newaxis]
    # alpha = alpha.reshape(5,4,2)
    # alpha[:,-1] = np.array([0.5, 0.5])
    # alpha = alpha.reshape(20,2)


    # generate 15 combinations of size (15,2), each sum up to 1, in order (divide 1 by 15 and get 15 combinations)
    alpha = np.array([[i/15, 1-i/15] for i in range(1, 16)])
    alpha = alpha.reshape(5,3,2)
    # add 0.5, 0.5 to the column so that the final shape is (5,4,2)
    temp = np.empty((5,4,2))
    temp[:5,:3] = alpha
    temp[:,-1] = np.array([0.5, 0.5])
    alpha = temp.reshape(20,2)

    # compute the linear combination of mix1 and mix2 given the alpha
    mixtures = alpha[:, 0][:, np.newaxis] * mix1 + alpha[:, 1][:, np.newaxis] * mix2

    print(mixtures.shape)
    assert (mixtures.sum(axis=1)== 100).all, "The sum of each mixture should be 100"  

    xticklabels = [' ', ' ', ' ','0.5 m1 + 0.5 m2']

    return mixtures/100.0, 5,4, xticklabels
    
    
def get_mixtures4():
    '''
     Here we approximate final_mix = sum(mix_i * alpha_i * multiply(1-alpha_j)) where j is between (0, i-1)

     The experiment shows that the difference between three mix is visually distinguishable
    '''

    # generate 5 random mixtures of CMYKW that sum up to 100
    random_mixtures = np.random.rand(5, 5)
    random_mixtures = random_mixtures / random_mixtures.sum(axis=1)[:, np.newaxis] * 100
    # append 0 to the end of each row
    mixtures = np.concatenate([random_mixtures, np.zeros((5, 1))], axis=1)
    
    # generate 5 random alpha, each is between 0 to 1
    alpha = np.random.rand(5)
    
    # The first mix follows final_mix = sum(mix_i * alpha_i * multiply(1-alpha_j)) where j is between (0, i-1) 
    mix_1 = np.zeros((6,))
    for i in range(5):
        mix_1 += mixtures[i] * alpha[i] * np.prod(1-alpha[:i])
    

    # The second mix follows final_mix = sum(mix_i * alpha_i * multiply(1-alpha_j)) where j is between (0, i-1)
    # but the first alpha has been changed to 1
    mix_2 = np.zeros((6,))
    alpha2 = alpha.copy()
    alpha2[0] = 1
    for i in range(5):
        mix_2 += mixtures[i] * alpha2[i] * np.prod(1-alpha2[:i])
    

    # The third mix is the mean of all the mixtures
    mix_3 = mixtures.mean(axis=0)

    # put mix_1, mix_2, mix_3 together as a  (1,3,6) array
    mixtures = np.stack([mix_1, mix_2, mix_3], axis=0)

    print(mixtures)

    xticklabels = ['rasterize mix', 'alpha 1 rasterize mix', 'equal mix']

    return mixtures/100.0, 1, 3, xticklabels
    



      
if __name__ == "__main__":
    mixtures, rows, cols, xticklabels =  get_mixtures4()
    km_rgb, r_mix = get_wavelength_KM_RGB(mixtures)


    colors_reshaped = km_rgb.reshape((rows, cols, 3))
    r_mix = r_mix.reshape((rows, cols, -1))

    # Create the figure and axis
    dpi = 100  # Display pixels per inch
    fig_size = 800 / dpi  # 800 pixels / 100 dpi = 8 inches
    #  add black line between each color
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)

    # Display the image
    ax.imshow(colors_reshaped, aspect='equal')

    # Add black grid lines to separate each color
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    # Mark percentages on the x-axis
    num_cols = colors_reshaped.shape[1]
    percentages = [i *1/num_cols for i in range(1, num_cols+1)] 
    xticks = [int(p * num_cols - 1) for p in percentages]
    # xticklabels = [f'alpha {int(p * 100)}%' for p in percentages]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # Remove default minor axis ticks for clarity
    ax.set_yticks([])
    ax.tick_params(axis='x', which='minor', bottom=False)

    # compute the mean squared difference between the colors
    color_100 = colors_reshaped[:,-1,:][:,np.newaxis,:]
    diff =  np.mean(np.square(colors_reshaped - color_100), axis=2)

    spec_100 = r_mix[:,-1,:][:,np.newaxis,:]
    spec_diff = np.mean(np.square(r_mix - spec_100), axis=2)



    for i in range(rows):
        for j in range(cols):
            rgb_diff = diff[i, j]
            spec_diff_ = spec_diff[i, j]
            ax.text(j, i, "MSE: "+"{:.2e}".format(rgb_diff)+ ",\n"+ "{:.2e}".format(spec_diff_), ha='center', va='center', color='black', fontsize=8, fontweight='bold')


    
    # Remove axes
    # plt.savefig("KM_linear_comb_alpha_test.png")
    # plt.savefig("KM_alpha_test.png")
    # plt.savefig("KM_transparent_test.png")
    plt.savefig("KM_rasterize_test.png")

