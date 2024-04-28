import numpy as np
import json
import colour
import torch
from torch import nn
import torchvision



import scipy.io as sio

# mat = sio.loadmat('ink_meta/D65.mat')

# # print(mat.keys())
# # print(mat)
# print(type(mat['D65']))
# wavelength = mat['D65'][:,0].astype(np.int32)
# D65 = mat['D65'][:,1]
# D65 = D65 / D65.max()

# print(wavelength)


# illuminant_D65_spd = colour.SDS_ILLUMINANTS['D65']
# D56_ = np.array([illuminant_D65_spd[w] for w in wavelength])
# D56_ = D56_ / D56_.max()


class Ink(nn.Module):
    def __init__(self, use_torch = True, device = "cuda"):

        super(Ink, self).__init__()
        self.device = device


        absorption_file = "ink_meta/absorbtion_K.csv"
        scattering_file = "ink_meta/scattering_S.csv"

        # 1: K, 2:C 3:M, 4:Y, 5: W
        absorption_matrix = np.genfromtxt(absorption_file, delimiter=',') # they are not in the order of CMYKW
        scattering_matrix = np.genfromtxt(scattering_file, delimiter=',')

        C_absorption = absorption_matrix[1]
        M_absorption = absorption_matrix[2]
        Y_absorption = absorption_matrix[3]
        K_absorption = absorption_matrix[0]
        # W_absorption = np.array(ink_intrinsic["W_absorption"])
        W_absorption = np.zeros_like(C_absorption) # Fake perfect white ink data
        T_absorption = np.zeros_like(C_absorption) # Fake perfect transparent ink data
        
        self.absorption_matrix = np.array([C_absorption, M_absorption, Y_absorption, K_absorption, W_absorption, T_absorption])

        C_scattering = scattering_matrix[1]
        M_scattering = scattering_matrix[2]
        Y_scattering = scattering_matrix[3]
        K_scattering = scattering_matrix[0]
        W_scattering = scattering_matrix[4]
        T_scattering = np.zeros_like(C_scattering) + 1e-4 # Fake perfect transparent ink data

        self.scattering_matrix = np.array([C_scattering, M_scattering, Y_scattering, K_scattering, W_scattering, T_scattering])


       
        self.w_380_10_730 = np.arange(380, 740, 10)
        self.w_380_5_780 = np.arange(380, 785, 5)
        self.w_delta = int(self.w_380_5_780[1] - self.w_380_5_780[0])
        self.w_num = self.w_380_5_780.shape[0]

        # Sample the color matching functions at these wavelengths
        # The helper_wavelength consists of two parts:
        # first part : element in self.w_380_5_780
        # [380 390 400 410 420 430 440 450 460 470 480 490 500 510 520 530 540 550 560 570 580 590 600 610 620 630 640 650 660 670 680 690 700 710 720 730]
        # second part: different elements between self.w_380_5_780 and self.w_380_10_730
        # [385 395 405 415 425 435 445 455 465 475 485 495 505 515 525 535 545 555 565 575 585 595 605 615 625 635 645 655 665 675 685 695 705 715 725 735 740 745 750 755 760 765 770 775 780]
        helper_wavelength = np.concatenate([self.w_380_10_730, np.setdiff1d(self.w_380_5_780, self.w_380_10_730)])
        assert helper_wavelength[36] == 385, "The 36th element should be 385"
        assert helper_wavelength[71] == 735, "The 37th element should be 395"


        # Fetch the D65 illuminant spectral power distribution
        illuminant_D65_spd = colour.SDS_ILLUMINANTS['D65']
        sampled_illuminant_D65 = np.array([illuminant_D65_spd[w] for w in helper_wavelength])
        self.sampled_illuminant_D65 =  sampled_illuminant_D65 / sampled_illuminant_D65.max()



        # Fetch the CIE 1931 2-degree Standard Observer Color Matching Functions
        # cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']

       
        self.x_observer = np.array([cmfs[w][0] for w in helper_wavelength])
        self.y_observer = np.array([cmfs[w][1] for w in helper_wavelength])
        self.z_observer = np.array([cmfs[w][2] for w in helper_wavelength])
       


        # RGB absorption and scattering data from  the scoped paper and https://research-explorer.ista.ac.at/download/486/7189/ElekSumin2017SGA_reduced_file_size.pdf
        albedo_RGB = np.array([
                    [0.05, 0.7, 0.98],  # Cyan
                    [0.98, 0.1, 0.9],  # Magenta
                    [0.997, 0.995, 0.15],  # Yellow
                    [0.35, 0.35, 0.35],  # KEY: Black
                    [0.9991, 0.9997, 0.999],   # White
                    [1.0, 1.0, 1.0] #Transparent
                    ])
        sigma_RGB = np.array([
                    [9.0, 4.5, 7.5],  # Cyan
                    [2.5, 3.0, 10.0],  # Magenta
                    [2.25, 3.75, 19.0],  # Yellow
                    [5.0, 5.5, 6.5],  # KEY: Black
                    [6.0, 9.0, 24.0],   # White
                    [1e-4, 1e-4, 1e-4]] #Transparent
                    )
        # NOTE: in mitusba, the input vol file only reads values between 0 and 1, so we divide by 20 to make sure the values are in that range
        # NOTE: but in rendering we should put the 20 back
        
        self.scattering_RGB = albedo_RGB * sigma_RGB
        self.absorption_RGB = sigma_RGB - self.scattering_RGB
        
        if use_torch:
            with torch.no_grad():
                self.absorption_matrix = torch.tensor(self.absorption_matrix, dtype=torch.float32, device=self.device)
                self.scattering_matrix = torch.tensor(self.scattering_matrix, dtype=torch.float32, device=self.device)
                self.sampled_illuminant_D65 = torch.tensor(self.sampled_illuminant_D65, dtype=torch.float32, device=self.device)
                self.x_observer = torch.tensor(self.x_observer, dtype=torch.float32, device=self.device)
                self.y_observer = torch.tensor(self.y_observer, dtype=torch.float32, device=self.device)
                self.z_observer = torch.tensor(self.z_observer, dtype=torch.float32, device=self.device)


INK = Ink(use_torch = False)

def mix_2_RGB_wavelength(mix, keep_dim = False):
        assert mix.shape[-1] == 6, "Ink mixture should have 6 channels"
        assert (mix >= 0.0).all(), "Ink mixture should be positive"
        assert (mix <= 1.0 + 1e-1).all(), "Ink mixture should be less than 1.0. {} > 1.0".format(mix.max())

        H,W,C = mix.shape
        mix = mix.reshape(-1,6)
        N, C = mix.shape
        # mix = mix[0].reshape(1,6)

        # K
        mix_K = mix @ INK.absorption_matrix
        # S
        mix_S = mix @ INK.scattering_matrix + 1e-8

        #equation 2
        R_mix = 1 + mix_K / mix_S - np.sqrt( (mix_K / mix_S)**2 + 2 * mix_K / mix_S)

        print("R_mix", R_mix.shape)


        # # manual linear interpolation, for each element in the row vector of R_mix, we compute the mean between the two values and insert it in between
        # R_mix = np.insert(R_mix, np.arange(1, R_mix.shape[1]), (R_mix[:,1:] + R_mix[:,:-1]) / 2, axis=1)
        
        # # check the length of the row vector, if the lenght is smaller than the length of INK.w_380_5_780, append zeros
        # R_mix = np.pad(R_mix, ((0,0),(0, INK.w_num - R_mix.shape[1])))


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

        print("x_D56", x_D56.shape)
        print("R_mix", R_mix.shape)

        X = R_mix @ x_D56
        Y = R_mix @ y_D56
        Z = R_mix @ z_D56

        X = X / INK.w_num
        Y = Y / INK.w_num
        Z = Z / INK.w_num

        XYZ = np.stack([X,Y,Z],axis=1).T
        print("xyz", XYZ[:,0])
        print("lab ",colour.XYZ_to_Lab(XYZ[:,0]))
        print("converted sRGB", colour.XYZ_to_sRGB(XYZ[:,0]))
        

        # Convert XYZ to sRGB, Equation 7
        sRGB_matrix = np.array([[3.2406, -1.5372, -0.4986],
                                [-0.9689, 1.8758, 0.0415],
                                [0.0557, -0.2040, 1.0570]])
        sRGB = (sRGB_matrix @ XYZ).T

        # Apply gamma correction to convert linear RGB to sRGB
        sRGB = np.where(sRGB <= 0.0031308,
                      12.92 * sRGB,
                      1.055 * np.power(sRGB, 1 / 2.4) - 0.055)
        
        assert sRGB.shape == (N, 3), "sRGB shape should be (N,3)"
        sRGB = np.clip(sRGB,0.0,1.0)
        if keep_dim:

            return sRGB.reshape(H,W,3).transpose(2,0,1)
        return sRGB
        # sRGB = torch.clip(sRGB,0,1).view(H, W, 3).permute(2,0,1)



debug_mix = np.zeros((100,100,6))
debug_mix[:,:,0] = 1.0 # 100% Cyan
# debug_mix[:,:,1] = 1.0 # 100% Megenta
# debug_mix[:,:,2] = 1.0 # 100% Yellow
# debug_mix[:,:,3] = 1.0 # 100% Black
# debug_mix/=3.0


rgb = mix_2_RGB_wavelength(debug_mix, keep_dim=True)
print("compute srgb: ", rgb[:,0,0])
print("compute srgb: ", rgb[:,0,0]*255)




# save rgb to image using torchvision
rgb = torch.tensor(rgb, dtype=torch.float32)
torchvision.utils.save_image(rgb, "test.png")




# GT
# C RGB(0.161143603338491	0.386079697591527	0.539380368502028), XYZ(0.0988064738174822	0.111138600848107	0.254908369218620)
# M RGB(0.593880494758789	0.149149970340178	0.369410053323513), XYZ(0.155687650067820	0.0882343164324080	0.115177390969472)
# Y RGB(0.793278213431997	0.509936000327339	0.102114178582802), XYZ(0.326113187459846	0.286486866493540	0.0479167129566719)
# K RGB(0.271209453689649	0.275162687332554	0.280489460047089), XYZ(0.0582012747518211	0.0613386870063383	0.0692602108511878)
# CMYK RGB(0.390048612293088	0.291172415634963	0.305326381991949), XYZ(0.0903183070080099	0.0815806340399647	0.0827855223554335)