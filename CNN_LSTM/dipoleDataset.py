import torch
import os
import numpy as np
from mne.io import read_info

class DipoleDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_data_path, interp_data_path, source_path, info_path, im_shape=(9,9), get_whole_trial=False):
        self.eeg_data_path = eeg_data_path
        self.interp_data = interp_data_path
        self.source_path = source_path
        self.im_shape = im_shape
        self.get_whole_trial = get_whole_trial
        self.eeg_info = read_info(info_path)
        self.eeg_str = "sample_{}.npy"
        self.src_str = "source_{}.npy"

    # get sample
    def __getitem__(self, idx):

        eeg_sample = np.load(os.path.join(self.eeg_data_path, self.eeg_str.format(idx))) # shape: (n_channels, n_timesteps)
        interp_sample = np.load(os.path.join(self.interp_data, self.eeg_str.format(idx))) # shape: (n_timesteps, height, width)

        if not self.get_whole_trial:
            max_idx = np.unravel_index(np.argmax(eeg_sample), eeg_sample.shape)[1] # this is the timestep with the maximum eeg value, this will be used to train
            interp_sample = interp_sample[max_idx]

        # convert to tensor
        interp_sample = torch.Tensor(interp_sample)
       
        target = np.load(os.path.join(self.source_path, self.src_str.format(idx))) # shape: (n_dipoles, n_timesteps)
        target = np.swapaxes(target,0,1)
        
        if not self.get_whole_trial:
            target = target[max_idx] # take sources when eeg is maxed, then scale
        
        target = self.scale_source(target)
        
        # convert to tensor
        target = torch.Tensor(target)

        return interp_sample, target
    
    def scale_source(self, sources):
        ''' Scales the sources prior to training the neural network.

        Parameters
        ----------
        source : numpy.ndarray
            A 3D matrix of the source data (samples, dipoles, time_points)
        
        Return
        ------
        source : numpy.ndarray
            Scaled sources
        '''
        sources /= np.max(np.abs(sources))
        return sources

    def __len__(self):
        return sum([len(files) for r, d, files in os.walk(self.eeg_data_path)])