import numpy as np
import json
import colour
import torch
from torch import nn

class Ink(nn.Module):
    def __init__(self, use_torch = True, device = "cuda"):

        super(Ink, self).__init__()
        self.device = device


        absorption_file = "ink_meta/absorbtion_K.csv"
        scattering_file = "ink_meta/scattering_S.csv"
        wavelength_file = "ink_meta/spectral_samples.csv"

        # 1: K, 2:C 3:M, 4:Y, 5: W
        absorption_matrix = np.genfromtxt(absorption_file, delimiter=',') # they are not in the order of CMYKW
        scattering_matrix = np.genfromtxt(scattering_file, delimiter=',')
        wavelength = np.genfromtxt(wavelength_file, delimiter=',')

        # num_inks = absorption_matrix.shape[0]
        
        # self.wavelength = wavelength
        # W_absorption = np.zeros_like(absorption_matrix[0]) # Fake perfect white ink data
        # T_absorption = np.zeros_like(absorption_matrix[0]) # Fake perfect transparent ink data

        # print(W_absorption.shape)
        # print(absorption_matrix.shape)

        
        # self.absorption_matrix = np.vstack([absorption_matrix[:num_inks-1,:], W_absorption, T_absorption])
        # print(self.absorption_matrix.shape)

        # return None

        ink_intrinsic = json.load(open('ink_meta/ink_intrinsic.json'))
        self.wavelength = wavelength

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


        # Fetch the D65 illuminant spectral power distribution
        illuminant_D65_spd = colour.SDS_ILLUMINANTS['D65']
        self.sampled_illuminant_D65 = np.array([illuminant_D65_spd[w] for w in self.wavelength])



        # Fetch the CIE 1931 2-degree Standard Observer Color Matching Functions
        # cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']

        # Sample the color matching functions at these wavelengths
        self.x_observer = np.array([cmfs[w][0] for w in self.wavelength])
        self.y_observer = np.array([cmfs[w][1] for w in self.wavelength])
        self.z_observer = np.array([cmfs[w][2] for w in self.wavelength])


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



def RGB2INK():
    #TODO: Implement RGB2INK
    return 0

Ink()