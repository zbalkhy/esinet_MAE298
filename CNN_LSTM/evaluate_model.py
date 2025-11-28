from convnet import ConvDipNet
from torchinfo import summary
import torch.nn as nn
import numpy as np
import torch
import sys; sys.path.insert(0, '../')
from esinet.forward import create_forward_model, get_info
from CNN_LSTM.util import *
from dipoleDataset import DipoleDataset
import os
from esinet.evaluate import eval_auc, eval_nmse, eval_mse, eval_mean_localization_error
import json
from util import solve_p

model_dir = "/mnt/data/convdip/model/convdip_run2"
model_weight_path = os.path.join(model_dir, "convdip_499.pt")

model_save_path = "/mnt/data/convdip/model/"
loss_save_path = "/mnt/data/convdip/model/convdip_loss.npy"
data_path = "/mnt/data/convdip/training_data/"
eeg_data_path = os.path.join(data_path, "eeg_data")
interp_data_path = os.path.join(data_path, "interp_data")
source_data_path = os.path.join(data_path, "source_data")
info_path = os.path.join(data_path, "info.fif")

if __name__=="__main__":
    ### create model
    # define hyperparameters
    in_channels = 1
    im_shape = (9,9)
    n_filters = 8
    kernel_size = (3,3)

    # create single input ConvDipNet 
    convnet: nn.Module  = ConvDipNet(in_channels, im_shape, n_filters, kernel_size)


    # print model summary
    summary(convnet, input_size=(32, 1, im_shape[0], im_shape[1])) # (batch_size, n_timesteps, in_channels, height, width)

    ### load weights into model
    convnet.load_state_dict(torch.load(model_weight_path, weights_only=True))
    convnet.eval()

    ### load in dataset
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
                shuffle=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=B,
                shuffle=False,
    )
    test_dataloader = torch.utils.data.DataLoader(
                test_set,
                batch_size=B,
                shuffle=False,
    )

    ### create forward model, load in dipole positions
    fs = 100
    info = get_info(sfreq=fs)
    fwd = create_forward_model(sampling='ico4', info=info)
    leadfield = fwd['sol']['data']

    dipole_pos = np.load(os.path.join(data_path, "dipole_pos.npy"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    metric_save_path = os.path.join(model_dir, "evaluation_metrics.json")
    convnet.to(device)
    with torch.no_grad():
        metrics_per_sample = {}
        i=0
        for idxs, batch, target in test_dataloader:
            i+=1
            print(i)
            batch = batch.to(device, dtype=torch.float)
            output = convnet(batch)
            output = output.cpu()

            for idx in range(output.shape[0]):
                data_idx = int(idxs[idx])
                target_sample = np.array(target[idx])
                output_sample = np.array(output[idx])
                
                eeg = np.load(os.path.join(data_path, f"eeg_data/sample_{data_idx}.npy"))
                max_idx = np.unravel_index(np.argmax(eeg), eeg.shape)[1] # this is the timestep with the maximum eeg value, this will be used to train
                output_sample = solve_p(output_sample, eeg[:,max_idx], leadfield)

                
                auc_close, auc_far = eval_auc(target_sample, output_sample, dipole_pos)
                sample_auc = (auc_close + auc_far)/2
                
                mle = eval_mean_localization_error(target_sample, output_sample, dipole_pos)
                mse = eval_mse(target_sample, output_sample)
                nmse = eval_nmse(target_sample, output_sample)
                metrics_per_sample[data_idx] = [sample_auc, mle, mse, nmse]
            
            with open(metric_save_path, "w") as json_file:
                json.dump(metrics_per_sample, json_file)