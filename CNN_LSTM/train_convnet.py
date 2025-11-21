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

def weighted_MSE_loss(outputs, targets):
    weights = torch.softmax(targets, dim=-1) # sum along the dipole dimension
    error = (targets-outputs)**2
    return torch.mean(weights*error)

if __name__=="__main__":
    # define hyperparameters
    in_channels = 1
    im_shape = (9,9)
    n_filters = 8
    kernel_size = (3,3)

    # create single input ConvDipNet 
    convnet: nn.Module  = ConvDipNet(in_channels, im_shape, n_filters, kernel_size)

    # create TimeDistributed ConvDipNet to process all samples of timeseries at once
    time_distributed_convnet: nn.Module = TimeDistributed(convnet, batch_first=True) # change batch_first to False for now for evaluation, will change back later

    # print model summary
    summary(time_distributed_convnet, input_size=(32, 100, 1, im_shape[0], im_shape[1])) # (batch_size, n_timesteps, in_channels, height, width)

    data_dir = "/mnt/data/convdip/training_data/"
    eeg_data_dir = os.path.join(data_dir, "interp_data")
    source_data_dir = os.path.join(data_dir, "source_data")
    info_path = os.path.join(data_dir, "info.fif")
    dataset = DipoleDataset(eeg_data_dir, source_data_dir, info_path, im_shape=im_shape)
    test_size = 0.15
    val_size = 0.15

    test_amount, val_amount = int(dataset.__len__() * test_size), int(dataset.__len__() * val_size)

    # this function will automatically randomly split your dataset but you could also implement the split yourself
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [
                (dataset.__len__() - (test_amount + val_amount)), 
                test_amount, 
                val_amount
    ])

    B = 256  # batch size
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
    optimizer = optim.Adam(time_distributed_convnet.parameters(), lr=lr, 
                        betas=betas, eps=eps)

    loss_values = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    time_distributed_convnet.to(device)
    for epoch in range(500):  # loop over the dataset multiple times
        running_loss = 0.0
        for j, data in enumerate(train_dataloader):
            sample, target = data
            sample, target = sample.to(device), target.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = time_distributed_convnet(sample)
            loss = weighted_MSE_loss(outputs, target)#criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f'[{epoch + 1}] loss: {running_loss:.8e}')
        loss_values.append(running_loss)

    print('Finished Training')
    model_save_path = "/mnt/data/convdip/model/convdip.pt"
    loss_save_path = "/mnt/data/convdip/model/convdip_loss.npy"
    np.save(loss_save_path, np.array(loss_values))
    torch.save(time_distributed_convnet.state_dict(), model_save_path)