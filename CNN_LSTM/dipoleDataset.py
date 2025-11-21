import torch
import os
import numpy as np
from mne.io import read_info
from mne.channels.layout import _find_topomap_coords
from mne.viz.topomap import (_setup_interp, _make_head_outlines, _check_sphere, 
    _check_extrapolate)

class DipoleDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_data_path, source_path, info_path, im_shape=(7,11)):
        self.eeg_data_path = eeg_data_path
        self.source_path = source_path
        self.im_shape = im_shape
        self.eeg_info = read_info(info_path)
        self.eeg_str = "sample_{}.npy"
        self.src_str = "source_{}.npy"

    # get sample
    def __getitem__(self, idx):

        eeg_sample = np.load(os.path.join(self.eeg_data_path, self.eeg_str.format(idx))) # shape: (n_timesteps, height, width)
        #eeg_sample = self.transfrom_eeg_data(eeg_sample)
        # convert to tensor
        eeg_sample = torch.Tensor(eeg_sample)
       

        target = np.load(os.path.join(self.source_path, self.src_str.format(idx))) # shape: (n_dipoles, n_timesteps)
        target = self.scale_source(target)
        target = np.swapaxes(target,0,1)
        # convert to tensor
        target = torch.Tensor(target)

        return eeg_sample, target
    
    def transfrom_eeg_data(self, eeg_sample):
        eeg_sample = self.scale_eeg(eeg_sample)
        eeg_sample = np.swapaxes(eeg_sample,0,1)
        eeg_interpolated = self.interpolate_eeg(eeg_sample, self.im_shape, self.eeg_info)
        return eeg_interpolated


    def scale_eeg(self, eeg: np.ndarray, scale_individually: bool=True) -> np.ndarray:
        ''' Scales the EEG prior to training/ predicting with the neural 
        network.

        Parameters
        ----------
        eeg : torch.Tensor
            A 3D matrix of the EEG data (samples, channels, time_points)
        
        Return
        ------
        eeg : torch.Tensor
            Scaled EEG
        '''        
        if scale_individually:
            # Common average ref:
            for time in range(eeg.shape[-1]):
                eeg[:, time] -= np.mean(eeg[:, time])
                eeg[:, time] /= eeg[:, time].std()
                    
        else:
            eeg = self.robust_minmax_scaler(eeg)
            # Common average ref:
            for time in range(eeg.shape[-1]):
                eeg[:, time] -= torch.mean(eeg[:, time])
        return eeg
    
    def robust_minmax_scaler(eeg: np.ndarray) -> np.ndarray:
        lower, upper = [torch.quantile(eeg, 25), torch.quantile(eeg, 75)]
        return (eeg-lower) / (upper-lower)
    
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
    
    def make_interpolator(self, elec_pos, res=9, ch_type='eeg', image_interp="linear"):
        extrapolate = _check_extrapolate('auto', ch_type)
        sphere = sphere = _check_sphere(None)
        outlines = 'head'
        outlines = _make_head_outlines(sphere, elec_pos, outlines, (0., 0.))
        border = 'mean'
        extent, Xi, Yi, interpolator = _setup_interp(
            elec_pos, res, image_interp, extrapolate, outlines, border)
        interpolator.set_locations(Xi, Yi)
        return interpolator

    def interpolate_eeg(self, x_scaled, im_shape, info):
        elec_pos = _find_topomap_coords(info, info.ch_names)
        interpolator = self.make_interpolator(elec_pos, res=im_shape[0])
        list_of_time_slices = []
        for time_slice in x_scaled:
            time_slice_interp = interpolator.set_values(time_slice)()[::-1]
            time_slice_interp = time_slice_interp[np.newaxis, :, :]# (1, height, width)
            list_of_time_slices.append(time_slice_interp)
        x_scaled = np.stack(list_of_time_slices, axis=0)
        x_scaled[np.isnan(x_scaled)] = 0
        return x_scaled

    def __len__(self):
        return sum([len(files) for r, d, files in os.walk(self.eeg_data_path)])