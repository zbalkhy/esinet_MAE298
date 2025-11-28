from convnet import ConvDipNet
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import sys; sys.path.insert(0, '../')
from CNN_LSTM.util import *
from dipoleDataset import DipoleDataset
import os
from datetime import datetime
from WHD import WHD
import pandas as pd
from tqdm import tqdm


## define loss function
def weighted_MSE_loss(outputs, targets):
    weights = torch.softmax(targets, dim=-1) # sum along the dipole dimension
    error = (targets-outputs)**2
    return torch.mean(weights*error)

def contrastive_loss_fn(output, target):
    temp = 0.07

    output_norm = F.normalize(output, p=2, dim=1)  # Shape: (512, 5124)
    target_norm = F.normalize(target, p=2, dim=1)  # Shape: (512, 5214)

    cosine_sim = output_norm @ target_norm.T
    pos_sim = torch.diag(cosine_sim)
    neg_sim = cosine_sim

    loss = -(1/temp)*pos_sim + torch.log(torch.sum(torch.sum((1-neg_sim)*torch.exp((1/temp)*neg_sim), dim=1)))
    return loss


## define paths
model_save_path = "/mnt/data/convdip/model/convdip_run4"

whd_loss_save_path = "/mnt/data/convdip/model/convdip_run4/convdip_whd_loss.npy"
whd_val_loss_save_path = "/mnt/data/convdip/model/convdip_run4/convdip_whd_val_loss.npy"

contrastive_loss_save_path = "/mnt/data/convdip/model/convdip_run4/convdip_contrastive_loss.npy"
contrastive_val_loss_save_path = "/mnt/data/convdip/model/convdip_run4/convdip_contrastive_val_loss.npy"

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
    convnet_whd: nn.Module  = ConvDipNet(in_channels, im_shape, n_filters, kernel_size)

    convnet_contrastive: nn.Module  = ConvDipNet(in_channels, im_shape, n_filters, kernel_size)
    
    # print model summary
    summary(convnet_whd, input_size=(32, 1, im_shape[0], im_shape[1])) # (batch_size, n_timesteps, in_channels, height, width)

    
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
    optimizer_whd = optim.Adam(convnet_whd.parameters(), lr=lr, 
                        betas=betas, eps=eps)
    
    optimizer_contrastive = optim.Adam(convnet_contrastive.parameters(), lr=lr, 
                    betas=betas, eps=eps)

    whd_loss_values = []
    whd_val_loss_values = []

    contrastive_loss_values = []
    contrastive_val_loss_values = []
    

    sim_info = pd.read_pickle(os.path.join(data_path, 'simulation_info.pkl'))
    pos = np.load(os.path.join(data_path, "dipole_pos.npy"))
    positions_per_trial = sim_info['positions']
    pos = torch.tensor(pos)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    convnet_whd.to(device)
    convnet_contrastive.to(device)

    whd = WHD(pos, positions_per_trial, device)

    for epoch in range(500):  # loop over the dataset multiple times
        convnet_whd.train()
        convnet_contrastive.train()
        running_loss_whd = 0.0
        running_loss_contrastive = 0.0

        # train for epoch
        for j, data in enumerate(tqdm(train_dataloader)):
            idx, sample, target = data
            sample, target = sample.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            
            # zero the parameter gradients
            optimizer_whd.zero_grad()
            optimizer_contrastive.zero_grad()

            # forward + backward + optimize
            outputs_whd = convnet_whd(sample)
            outputs_contrastive = convnet_contrastive(sample)

            loss_whd = torch.zeros(idx.shape[0])
            loss_contrastive = torch.zeros(idx.shape[0])
            for b in range(idx.shape[0]):
                loss_whd[b] = whd.WHD_loss(torch.relu(outputs_whd[b,:]), idx[b].item())#criterion(outputs, target)
            loss_contrastive = contrastive_loss_fn(outputs_contrastive, target)
            loss_whd = torch.mean(loss_whd)
            loss_contrastive = torch.mean(loss_contrastive)

            loss_whd.backward()
            loss_contrastive.backward()

            optimizer_whd.step()
            optimizer_contrastive.step()

            # print statistics
            running_loss_whd += loss_whd.item()
            running_loss_contrastive += loss_contrastive.item()

        # get validation loss
        if epoch % 5 == 0:
            with torch.no_grad():
                convnet_whd.eval()
                convnet_contrastive.eval()

                running_val_loss_whd = 0
                running_val_loss_contrastive = 0
                for j, data in enumerate(val_dataloader):
                    idx, sample, target = data
                    sample, target = sample.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
                    
                    outputs_whd = convnet_whd(sample)
                    outputs_contrastive = convnet_contrastive(sample)

                    loss_whd = torch.zeros(idx.shape[0])
                    loss_contrastive = torch.zeros(idx.shape[0])
                    for b in range(idx.shape[0]):
                        loss_whd[b] = whd.WHD_loss(torch.relu(outputs_whd[b,:]), idx[b].item())#criterion(outputs, target)
                    
                    loss_contrastive = contrastive_loss_fn(outputs_contrastive, target)
                    loss_whd = torch.mean(loss_whd)
                    loss_contrastive = torch.mean(loss_contrastive)
                    
                    running_val_loss_whd += loss_whd.item()
                    running_val_loss_contrastive += loss_contrastive.item()
                whd_val_loss_values.append(running_val_loss_whd)
                contrastive_val_loss_values.append(running_val_loss_contrastive)

        print(f'[{epoch + 1}] train_loss: {running_loss_whd:.8e} val_loss: {running_val_loss_whd:.8e} time: {datetime.now()}')
        
        # append losses
        whd_loss_values.append(running_loss_whd)
        contrastive_loss_values.append(running_loss_contrastive)

        # save every 50 epochs
        if epoch % 20 == 0:
            np.save(whd_loss_save_path, np.array(whd_loss_values))
            np.save(contrastive_loss_save_path, np.array(contrastive_loss_values))
            np.save(whd_val_loss_save_path, np.array(whd_val_loss_values))
            np.save(contrastive_val_loss_save_path, np.array(contrastive_val_loss_values))
            
            torch.save(convnet_whd.state_dict(), os.path.join(model_save_path,"convdip__whd{}.pt".format(epoch)))
            torch.save(convnet_contrastive.state_dict(), os.path.join(model_save_path,"convdip__contrastive{}.pt".format(epoch)))

    print('Finished Training')

    np.save(whd_loss_save_path, np.array(whd_loss_values))
    np.save(contrastive_loss_save_path, np.array(contrastive_loss_values))
    np.save(whd_val_loss_save_path, np.array(whd_val_loss_values))
    np.save(contrastive_val_loss_save_path, np.array(contrastive_val_loss_values))
    
    torch.save(convnet_whd.state_dict(), os.path.join(model_save_path,"convdip__whd{}.pt".format(epoch)))
    torch.save(convnet_contrastive.state_dict(), os.path.join(model_save_path,"convdip__contrastive{}.pt".format(epoch)))
