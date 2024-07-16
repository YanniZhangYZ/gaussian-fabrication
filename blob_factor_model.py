import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from ink_intrinsics import Ink

INK = Ink()


def load_mixtures_factors():
    import os
    #  get all the subfolder names in blob_factor/data
    folders = [f for f in os.listdir('blob_factor/data') if os.path.isdir(os.path.join('blob_factor/data', f))]

    cyan_mixture = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    magenta_mixture = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    yellow_mixture = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    black_mixture = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]



    # cyan_factor = np.load('blob_factor/data/cyan/factors.npy').reshape(-1).tolist()
    # magenta_factor = np.load('blob_factor/data/magenta/factors.npy').reshape(-1).tolist()
    # yellow_factor = np.load('blob_factor/data/yellow/factors.npy').reshape(-1).tolist()
    # black_factor = np.load('blob_factor/data/black/factors.npy').reshape(-1).tolist()


    cyan_factor = np.load('blob_factor/data/cyan/factors.npy').squeeze().tolist()
    magenta_factor = np.load('blob_factor/data/magenta/factors.npy').squeeze().tolist()
    yellow_factor = np.load('blob_factor/data/yellow/factors.npy').squeeze().tolist()
    black_factor = np.load('blob_factor/data/black/factors.npy').squeeze().tolist()


    mixtures = [cyan_mixture, magenta_mixture, yellow_mixture, black_mixture]
    factors = [cyan_factor, magenta_factor, yellow_factor, black_factor]
    for f in folders:
        if f in ['cyan', 'magenta', 'yellow', 'black']:
            continue
        # get all the files in the subfolder
        path = os.path.join('blob_factor/data', f)
        # factor = np.load(path + "/factors.npy").reshape(-1).tolist()
        factor = np.load(path + "/factors.npy").squeeze().tolist()
        mix = np.load(path + "/mixtures.npy").reshape(-1).tolist()
        mixtures.append(mix)
        factors.append(factor)


    mixtures = torch.tensor(mixtures, dtype=torch.float32, device='cuda')
    factors = torch.tensor(factors, dtype=torch.float32, device='cuda')

    return mixtures, factors

       


def mixture_to_extinction_rgb(mixture):
    N, C = mixture.shape
    assert C == 6, "Mixture should have 6 columns"
    # preprocess given ink mixtures given the transmittance
    # K
    mix_absorption_RGB = mixture[:,:5] @ INK.absorption_RGB[:5]
    # S
    mix_scattering_RGB = mixture[:,:5] @ INK.scattering_RGB[:5]
    mix_extinction_RGB = mix_absorption_RGB + mix_scattering_RGB

    assert mix_extinction_RGB.shape == (N, 3), "Extinction should be RGB"

    return torch.tensor(mix_extinction_RGB / 255.0, dtype=torch.float32, device='cuda')



class ExtinctionModel(nn.Module):
    def __init__(self, input_features, hidden_units, output_features):
        super(ExtinctionModel, self).__init__()
        # Define the first layer, which maps from input features to hidden units
        self.hidden_layer = nn.Linear(input_features, hidden_units)
        # Activation function for the hidden layer
        self.activation = nn.ReLU()
        # Define the second hidden layer
        self.hidden_layer_2 = nn.Linear(hidden_units, hidden_units)
        # Activation function for the hidden layer
        self.activation_2 = nn.ReLU()
        # Define the output layer, which maps from hidden units to output features
        self.output_layer = nn.Linear(hidden_units, output_features)





    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation_2(x)
        # x = torch.sigmoid(self.output_layer(x))
        x = torch.exp(self.output_layer(x))


        return x
    



if __name__ == "__main__":
    # Parameters
    input_features = 4  # cov1d,  and three for the RGB vector
    hidden_units = 64   # Number of hidden units, adjust based on complexity
    output_features = 2 # Output is a single scalar

    # Create the model
    model = ExtinctionModel(input_features, hidden_units, output_features).to('cuda')

    print()
    print(model)
    print()

    # Load the data
    mixtures, factors = load_mixtures_factors()


    dataset_size = mixtures.shape[0]
    print(f"Dataset size: {dataset_size}")

    # plot the factors, it has 105 group of data, each group has 50 data points
    plt.figure()
    x_axis = np.linspace(0.001, 0.05, 50)
    for i in range(factors.shape[0]):
        plt.plot(x_axis, factors[i,:, 1].cpu().numpy())
    plt.xlabel('Thickness')
    plt.ylabel('Factors')
    plt.savefig('blob_factor/factors_z_gt.png')

    plt.figure()
    x_axis = np.linspace(0.001, 0.05, 50)
    for i in range(factors.shape[0]):
        plt.plot(x_axis, factors[i,:, 0].cpu().numpy())
    plt.xlabel('Thickness')
    plt.ylabel('Factors')
    plt.savefig('blob_factor/factors_xy_gt.png')


    thickness =  torch.tensor(np.linspace(0.001, 0.05, 50), dtype=torch.float32, device='cuda').unsqueeze(1)/ 0.05
    ext_coeff = mixture_to_extinction_rgb(mixtures)


    thickness_ = thickness.repeat(ext_coeff.shape[0], 1)
    ext_coeff_ = ext_coeff.repeat_interleave(thickness.shape[0], dim=0)
    inputs = torch.cat((thickness_.flatten().unsqueeze(1), ext_coeff_), dim=1)
    targets = factors.reshape(-1,2) / torch.tensor([20.0, 200.0], dtype=torch.float32, device='cuda')
    # targets = factors.reshape(-1,2) / torch.tensor([20.0, 60.0], dtype=torch.float32, device='cuda')



    # # Expand z_scales to match the first dimension of ext_coeff
    # thickness_ = thickness.repeat(1, ext_coeff.shape[0]).view(-1, 1)
    # # Expand ext_coeff to match the new dimension
    # ext_coeff_ = ext_coeff.repeat(thickness.shape[0], 1)
    # # Concatenate along the second dimension to form the new input vectors
    # inputs = torch.cat(( thickness_,ext_coeff_), dim=1)
    # targets = factors.transpose(0, 1).reshape(-1,1)
    
    # split the data into training and testing
    train_size = int(0.8 * inputs.shape[0])
    test_size = inputs.shape[0] - train_size
    train_inputs_idx = np.random.choice(inputs.shape[0], train_size, replace=False)
    test_inputs_idx = np.setdiff1d(np.arange(inputs.shape[0]), train_inputs_idx)

    train_inputs = inputs[train_inputs_idx]
    train_targets = targets[train_inputs_idx]

    test_inputs = inputs[test_inputs_idx]
    test_targets = targets[test_inputs_idx]


    # import pdb; pdb.set_trace()

    # Create an instance of the network

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer (using Adam here for better performance compared to SGD)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs
    epochs = int(5e4)

    best_val_loss = np.inf
    val_not_improved = 0

    losses = []
    val_losses = []
    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()
        outputs = model(train_inputs)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()

        # Print statistics
        if epoch % 100 == 0:
            model.eval()  # Set the model to evaluation mode for validation
            with torch.no_grad():
                outputs = model(test_inputs)
                val_loss = criterion(outputs, test_targets).item()
                print(f'Epoch {epoch}, Validation Loss: {val_loss}')
                val_losses.append(val_loss)
                losses.append(loss.item())

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    val_not_improved = 0
                    torch.save(model.state_dict(), 'blob_factor/best_model.pth')
                else:
                    val_not_improved += 1
                # if val_not_improved > 20:
                #     print('Early stopping')
                #     break
    
    # torch.save(model.state_dict(), 'blob_factor/factor_model.pth')

    plt.figure()
    plt.plot(losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # print the final loss and validation loss in the figure
    plt.text(0.5, 0.5, f'Final Loss: {losses[-1]:.4f}\nValidation Loss: {val_losses[-1]:.4f}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
    plt.savefig('blob_factor/model_loss.png')

    
    model = ExtinctionModel(input_features, hidden_units, output_features).to('cuda')
    model.load_state_dict(torch.load('blob_factor/best_model.pth'))
    model.eval() 

    with torch.no_grad():
        gt = []
        predict = []
        plt_thickness = torch.tensor(np.linspace(0.001, 0.05, 50), dtype=torch.float32, device='cuda').unsqueeze(1)/ 0.05
        for i in range(dataset_size):
            plt_ext_coeff = mixture_to_extinction_rgb(mixtures)[i].unsqueeze(0)
            plt_thickness_ = plt_thickness.repeat(1, plt_ext_coeff.shape[0]).view(-1, 1)
            plt_ext_coeff_ = plt_ext_coeff.repeat(plt_thickness.shape[0], 1)
            plt_inputs = torch.cat((plt_thickness_, plt_ext_coeff_), dim=1)
            plt_predict = model(plt_inputs).cpu().numpy()

            gt.append(factors[i].cpu().numpy())
            predict.append(plt_predict)


        # plot 7*15 subplots in one figure
        plt.figure(figsize=(15, 7))
        for i in range(dataset_size):
            plt.subplot(7, 15, i+1)
            plt.plot(plt_thickness.cpu().numpy() * 0.05, predict[i][:,1] * 200.0, color='red', label='Predicted')
            plt.plot(plt_thickness.cpu().numpy() * 0.05, gt[i][:,1], color = 'blue',label='True')
            #  don't show the x and y tick
            plt.xticks([])
            plt.yticks([])
            # plt.xlabel('Thickness')
            # plt.ylabel('Factors')

        # plt.figure()
        # plt.plot(test_thickness.cpu().numpy(), predicted, label='Predicted')
        # plt.plot(test_thickness.cpu().numpy(), test_targets.cpu().numpy(), label='True')
        # plt.xlabel('Thickness')
        # plt.ylabel('Factors')
        # plt.legend()
        plt.savefig('blob_factor/extinction_model_z.png')

                # plot 7*15 subplots in one figure
        plt.figure(figsize=(15, 7))
        for i in range(dataset_size):
            plt.subplot(7, 15, i+1)
            plt.plot(plt_thickness.cpu().numpy() * 0.05, predict[i][:,0]*20.0, color='red', label='Predicted')
            plt.plot(plt_thickness.cpu().numpy() * 0.05, gt[i][:,0], color = 'blue',label='True')
            #  don't show the x and y tick
            plt.xticks([])
            plt.yticks([])
            # plt.xlabel('Thickness')
            # plt.ylabel('Factors')

        # plt.figure()
        # plt.plot(test_thickness.cpu().numpy(), predicted, label='Predicted')
        # plt.plot(test_thickness.cpu().numpy(), test_targets.cpu().numpy(), label='True')
        # plt.xlabel('Thickness')
        # plt.ylabel('Factors')
        # plt.legend()
        plt.savefig('blob_factor/extinction_model_xy.png')



        


