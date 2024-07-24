import numpy as np
import matplotlib.pyplot as plt

loss_record = np.load('/home/yanni/Thesis/gaussian-fabrication/3dgs_lego_train/result_imgs/loss.npy')

# plot loss
plt.figure()
plt.plot(loss_record)
plt.xticks(np.arange(0, loss_record.shape[0], step=1000))
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("/home/yanni/Thesis/gaussian-fabrication/3dgs_lego_train/result_imgs/loss.png")
