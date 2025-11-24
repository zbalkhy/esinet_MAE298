from convnet import ConvDipNet
from timeDistributed import TimeDistributed
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import torch
import sys; sys.path.insert(0, '../')
from CNN_LSTM.util import *
from dipoleDataset import DipoleDataset
import os

## define loss function
def weighted_MSE_loss(outputs, targets):
    weights = torch.softmax(targets, dim=-1) # sum along the dipole dimension
    error = (targets-outputs)**2
    return torch.mean(weights*error)

## define paths
model_save_path = "/mnt/data/convdip/model/"
loss_save_path = "/mnt/data/convdip/model/convdip_loss.npy"
data_path = "/mnt/data/convdip/training_data/"
eeg_data_path = os.path.join(data_path, "eeg_data")
interp_data_path = os.path.join(data_path, "interp_data")
source_data_path = os.path.join(data_path, "source_data")
info_path = os.path.join(data_path, "info.fif")

if __name__ == "__main__":
    ##define model
    # define hyperparameters
    in_channels = 1
    im_shape = (9,9)
    n_filters = 8
    kernel_size = (3,3)

    # create single input ConvDipNet 
    convnet: nn.Module  = ConvDipNet(in_channels, im_shape, n_filters, kernel_size)

    # print model summary
    summary(convnet, input_size=(32, 1, im_shape[0], im_shape[1])) # (batch_size, n_timesteps, in_channels, height, width)

    
    dataset = DipoleDataset(eeg_data_path, interp_data_path, source_data_path, info_path, im_shape=im_shape)
    test_size = 0.15
    val_size = 0.15

    test_amount, val_amount = int(dataset.__len__() * test_size), int(dataset.__len__() * val_size)

    # this function will automatically randomly split your dataset but you could also implement the split yourself
    gen = torch.Generator()
    gen.manual_seed(0) # this is the seed we use to split the data the same way each time
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [
                (dataset.__len__() - (test_amount + val_amount)), 
                test_amount, 
                val_amount
    ], generator=gen)

    B = 512  # batch size
    train_dataloader = torch.utils.data.DataLoader(
                train_set,
                batch_size=B,
                shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=B,
                shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
                test_set,
                batch_size=B,
                shuffle=True,
    )

    lr = 0.001
    betas=(0.9, 0.999)
    eps = 1e-8
    optimizer = optim.Adam(convnet.parameters(), lr=lr, 
                        betas=betas, eps=eps)

    loss_values = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    convnet.to(device)

    for epoch in range(500):  # loop over the dataset multiple times
        running_loss = 0.0
        for j, data in enumerate(tqdm(train_dataloader)):
            sample, target = data
            sample, target = sample.to(device), target.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = convnet(sample)
            loss = weighted_MSE_loss(outputs, target)#criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f'[{epoch + 1}] loss: {running_loss:.8e}')
        loss_values.append(running_loss)
        if epoch % 50 == 0:
            np.save(loss_save_path, np.array(loss_values))
            torch.save(convnet.state_dict(), os.path.join(model_save_path,"convdip_{}.pt".format(epoch)))

    print('Finished Training')

    np.save(loss_save_path, np.array(loss_values))
    torch.save(convnet.state_dict(), os.path.join(model_save_path,"convdip_{}.pt".format(epoch)))
