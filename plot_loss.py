import numpy as np
import matplotlib.pyplot as plt

loss_record1 = np.load('/home/yanni/Thesis/gaussian-fabrication/3dgs_lego_train/mitsuba/render/z_z_high_spp/spec_xyz5e-3sqrt_0.01relu1d0.01_0.0025_surface_AS_AS_factor_xyz_10k_new/loss.npy')
loss_record2 = np.load('/home/yanni/Thesis/gaussian-fabrication/3dgs_lego_train/mitsuba/render/z_z_high_spp/spec_rand_mix_xyz5e-3sqrt_0.01relu1d0.01_0.0025_surface_AS_AS_factor_xyz_10k_new/loss.npy')

print(np.abs(loss_record1 - loss_record2).sum())


# plot loss
plt.figure()
idx = np.arange(0, loss_record1.shape[0], step=1000)
plt.plot(loss_record1[idx], label='white')
plt.plot(loss_record2[idx], label='random')
# plt.xticks(np.arange(0, loss_record1.shape[0], step=1000))
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("white_random_loss.png")
