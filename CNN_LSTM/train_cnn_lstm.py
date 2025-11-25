from convnet import ConvDipNet
from timeDistributed import TimeDistributed, TimeDistributedLinear
from torchinfo import summary
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import sys; sys.path.insert(0, '../')
from esinet.forward import create_forward_model, get_info
from esinet import Simulation
from copy import deepcopy
from CNN_LSTM.util import *
from dipoleDataset import DipoleDataset
import os
import mne
from cnn_lstm import CNNLSTM
from datetime import datetime

def weighted_MSE_loss(outputs, targets):
    weights = torch.softmax(targets, dim=-1) # sum along the dipole dimension
    error = (targets-outputs)**2
    return torch.mean(weights*error)

if __name__ == "__main__":
    # define hyperparameters
    in_channels = 1
    im_shape = (9,9)
    n_filters = 8
    kernel_size = (3,3)

    # create single input ConvDipNet
    cnnlstm = CNNLSTM(in_channels, im_shape, n_filters, kernel_size)

    # print model summary
    summary(cnnlstm, input_size=(32, 100, 1, im_shape[0], im_shape[1])) # (batch_size, n_timesteps, in_channels, height, width)

    data_dir = "/mnt/data/convdip/training_data/"
    eeg_data_dir = os.path.join(data_dir, "eeg_data")
    interp_data_dir = os.path.join(data_dir, "interpolated_eeg_data_for_lstm")
    source_data_dir = os.path.join(data_dir, "source_data")
    info_path = os.path.join(data_dir, "info.fif")
    dataset = DipoleDataset(eeg_data_dir, interp_data_dir, source_data_dir, info_path, im_shape=im_shape, get_whole_trial=True)
    test_size = 0.15
    val_size = 0.15

    test_amount, val_amount = int(dataset.__len__() * test_size), int(dataset.__len__() * val_size)

    # this function will automatically randomly split your dataset but you could also implement the split yourself
    gen = torch.Generator()
    gen.manual_seed(0)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [
                (dataset.__len__() - (test_amount + val_amount)), 
                test_amount, 
                val_amount], 
                generator=gen)

    B = 1024  # batch size
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
    optimizer = optim.Adam(cnnlstm.parameters(), lr=lr, 
                        betas=betas, eps=eps)

    model_save_path = "/mnt/data/convdip/model/"
    loss_save_path = "/mnt/data/convdip/model/cnnlstm_loss.npy"

    loss_values = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cnnlstm.to(device)

    for epoch in range(500):  # loop over the dataset multiple times
        running_loss = 0.0
        for j, data in enumerate(train_dataloader):
            sample, target = data
            sample, target = sample.to(device), target.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnnlstm(sample)
            loss = weighted_MSE_loss(outputs, target)#criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f'[{epoch + 1}] loss: {running_loss:.8e}, time: {datetime.now()}')
        loss_values.append(running_loss)
        if epoch % 50 == 0:
            np.save(loss_save_path, np.array(loss_values))
            torch.save(cnnlstm.state_dict(), os.path.join(model_save_path,"cnnlstm_{}.pt".format(epoch)))

    print('Finished Training')

    np.save(loss_save_path, np.array(loss_values))
    torch.save(cnnlstm.state_dict(), os.path.join(model_save_path,"cnnlstm_{}.pt".format(epoch)))