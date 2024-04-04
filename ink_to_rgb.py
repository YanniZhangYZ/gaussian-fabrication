import numpy as np
import json
import colour
import torchvision
import matplotlib.image
import torch

ink_intrinsic = json.load(open('ink_intrinsic.json'))
# mix = np.load('Two_blobs/renders/final_ink_mix.npy')
mix = np.zeros((6, 800, 800))
recipe = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])/3.0
mix += recipe[:,None,None]

back_ground_mask =np.all(mix == 0.0, axis=0).astype(int)

blob_mask = 1 - back_ground_mask


transparent_mask = 1.0 - mix.sum(axis=0)
transparent_blob_mask = blob_mask - mix.sum(axis=0)

# mix[5,:,:] = transparent_blob_mask # fake transparent ink concentration
# mix[4,:,:] = transparent_mask # fake white ink concentration

C, H, W = mix.shape
print(C, H, W)



mix = mix.transpose(1,2,0).reshape(-1,C)

wavelength = np.array(ink_intrinsic["wavelength"])
C_absorption = np.array(ink_intrinsic["C_absorption"])
M_absorption = np.array(ink_intrinsic["M_absorption"])
Y_absorption = np.array(ink_intrinsic["Y_absorption"])
K_absorption = np.array(ink_intrinsic["K_absorption"])
# W_absorption = np.array(ink_intrinsic["W_absorption"])
W_absorption = np.zeros_like(C_absorption) # Fake perfect white ink data
T_absorption = np.zeros_like(C_absorption) # Fake perfect transparent ink data

absorption_matrix = np.array([C_absorption, M_absorption, Y_absorption, K_absorption, W_absorption, T_absorption])

C_scattering = np.array(ink_intrinsic["C_scattering"])
M_scattering = np.array(ink_intrinsic["M_scattering"])
Y_scattering = np.array(ink_intrinsic["Y_scattering"])
K_scattering = np.array(ink_intrinsic["K_scattering"])
W_scattering = np.array(ink_intrinsic["W_scattering"])
T_scattering = np.zeros_like(C_scattering) + 1e-4 # Fake perfect transparent ink data

scattering_matrix = np.array([C_scattering, M_scattering, Y_scattering, K_scattering, W_scattering, T_scattering])



# Fetch the D65 illuminant spectral power distribution
illuminant_D65_spd = colour.SDS_ILLUMINANTS['D65']
sampled_illuminant_D65 = np.array([illuminant_D65_spd[w] for w in wavelength])



# Fetch the CIE 1931 2-degree Standard Observer Color Matching Functions
cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
# Sample the color matching functions at these wavelengths
x_observer = np.array([cmfs[w][0] for w in wavelength])
y_observer = np.array([cmfs[w][1] for w in wavelength])
z_observer = np.array([cmfs[w][2] for w in wavelength])


def ink_to_rgb():

    # print(sampled_illuminant_D65.sum())
    # mix: (H*W,6) array of ink mixtures
    # K
    mix_K = mix @ absorption_matrix + 1e-10
    # S
    mix_S = mix @ scattering_matrix + 1e-10
    #equation 2
    R_mix = 1 + mix_K / mix_S - np.sqrt( (mix_K / mix_S)**2 + 2 * mix_K / mix_S)
    # Saunderson’s correction reflectance coefficient, commonly used values
    k1 = 0.04
    k2 = 0.6
    # equation 6
    R_mix = (1-k1) * (1 - k2) * R_mix / (1 - k2 * R_mix)

    # equation 3 - 5
    x_D56 = x_observer * sampled_illuminant_D65
    # x_D56 /= np.sum(x_D56)
    y_D56 = y_observer * sampled_illuminant_D65
    # y_D56 /= np.sum(y_D56)
    z_D56 = z_observer * sampled_illuminant_D65
    # z_D56 /= np.sum(z_D56)
    X = R_mix @ x_D56
    Y = R_mix @ y_D56
    Z = R_mix @ z_D56

    XYZ = np.stack([X,Y,Z],axis=1).T
    Y_D56 = np.sum(sampled_illuminant_D65 * y_observer)
    
    # Convert XYZ to sRGB, Equation 7
    sRGB_matrix = np.array([[3.2406, -1.5372, -0.4986],
                            [-0.9689, 1.8758, 0.0415],
                            [0.0557, -0.2040, 1.0570]])
    sRGB = ((sRGB_matrix @ XYZ) / Y_D56).T
    sRGB = np.clip(sRGB,0,1).reshape(H, W, 3).transpose(2,0,1)
    # sRGB[:, back_ground_mask == 1] = 1
    # print(sRGB.shape)
    # sRGB = sRGB + transparent_mask.reshape(1, H, W)

    print(sRGB[:, int(H/2), int(W/2)])
    # print(transparent_mask.shape)
    torchvision.utils.save_image(torch.tensor(sRGB), "ink_RGB.png")



import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab
from torchmetrics.image import StructuralSimilarityIndexMeasure


def torch_ink_to_rgb(w):
    mix = w.repeat(1,16,16).permute(1, 2, 0).view(-1, 6)

    # print(sampled_illuminant_D65.sum())
    # mix: (H*W,6) array of ink mixtures
    # K

    mix_K = mix @ torch.tensor(absorption_matrix).float() + 1e-8
    # S
    mix_S = mix @ torch.tensor(scattering_matrix).float() + 1e-8
    #equation 2
    R_mix = 1 + mix_K / mix_S - torch.sqrt( (mix_K / mix_S)**2 + 2 * mix_K / mix_S)
    # Saunderson’s correction reflectance coefficient, commonly used values
    k1 = 0.04
    k2 = 0.6
    # equation 6
    R_mix = (1-k1) * (1 - k2) * R_mix / (1 - k2 * R_mix)

    # equation 3 - 5
    x_D56 = x_observer * sampled_illuminant_D65
    # x_D56 /= np.sum(x_D56)
    y_D56 = y_observer * sampled_illuminant_D65
    # y_D56 /= np.sum(y_D56)
    z_D56 = z_observer * sampled_illuminant_D65
    # z_D56 /= np.sum(z_D56)
    X = R_mix @ torch.tensor(x_D56).float()
    Y = R_mix @ torch.tensor(y_D56).float()
    Z = R_mix @ torch.tensor(z_D56).float()

    XYZ = torch.stack([X,Y,Z],axis=1).T
    Y_D56 = torch.sum(torch.tensor(y_D56).float())
    
    # Convert XYZ to sRGB, Equation 7
    sRGB_matrix = torch.tensor([[3.2406, -1.5372, -0.4986],
                            [-0.9689, 1.8758, 0.0415],
                            [0.0557, -0.2040, 1.0570]])
    sRGB = ((sRGB_matrix @ XYZ) / Y_D56).T
    sRGB = torch.clip(sRGB,0,1).view(16, 16, 3).permute(2,0,1)
    return sRGB



def loss_delta_76_func(w, GT_rgb):
    w = F.relu(w)
    w = w / torch.sum(w)

    # here the w is only a vector (6,)
    current_render = torch_ink_to_rgb(w)

    # MSE between the reference and the current render
    lab_GT = rgb_to_lab(GT_rgb.reshape(1, 3, 16, 16)).reshape(16,16,3)
    lab_image = rgb_to_lab(current_render.reshape(1, 3, 16,16)).reshape(16,16,3)
    delta_e76 = torch.sqrt(torch.sum((lab_image - lab_GT)**2, dim=(0, 1))/(16*16))

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    # return 0.7* ssim(current_render.reshape(1,3, 16, 16), GT_rgb.reshape(1,3, 16, 16)) + 0.3 * F.mse_loss(current_render, GT_rgb)


    # return 0.1 * torch.mean(delta_e76) + 0.9 * F.mse_loss(current_render, GT_rgb)
    return torch.mean(delta_e76)
    # return F.mse_loss(current_render, GT_rgb)


def optimization_loop():
    
    # weight = torch.nn.Parameter(torch.tensor([0.0000, 0.5107, 0.4893, 0.0000, 0.0000, 0.0000]))
    # weight = torch.nn.Parameter(torch.tensor([0.3, 0.0, 0.3, 0.0000, 0.4, 0.0000]))
    weight = torch.nn.Parameter(torch.tensor([0.3, 0.0, 0.3, 0.0000, 0.4, 0.0]))



    lr = 0.01
    optimizer = torch.optim.Adam([weight], lr=lr)
    GT_rgb = torch.zeros(3,16,16)
    GT_rgb[1,:,:] = 1 # red
    torchvision.utils.save_image( GT_rgb, "GT_rgb.png")


    train_losses = []
    iteration_count = 100
    torch.autograd.set_detect_anomaly(True)
    for i in range(iteration_count):
        optimizer.zero_grad()
        loss = loss_delta_76_func(weight,GT_rgb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss)
        print(f'Training iteration {i+1}/{iteration_count}, loss: {train_losses[-1]}', end='\r')
    return weight
    

ink_to_rgb()  
    
# w = optimization_loop()
# w = F.relu(w)
# w = w / torch.sum(w)

# print(w.detach())
# final_render = torch_ink_to_rgb(w.detach())
# torchvision.utils.save_image( final_render, "ink_optimization.png")


# print(sampled_illuminant_D65)
# print(sampled_illuminant_D65.sum())


