from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
from mne.channels.layout import _find_topomap_coords
from mne.viz.topomap import (_setup_interp, _make_head_outlines, _check_sphere, 
    _check_extrapolate)

def robust_minmax_scaler(eeg: np.ndarray) -> np.ndarray:
    lower, upper = [torch.quantile(eeg, 25), torch.quantile(eeg, 75)]
    return (eeg-lower) / (upper-lower)

def scale_eeg(eeg: np.ndarray, scale_individually: bool=True) -> np.ndarray:
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
    eeg_out = deepcopy(eeg)
    
    if scale_individually:
        for sample, eeg_sample in enumerate(tqdm(eeg, desc="scaler")):
            # Common average ref:
            for time in range(eeg_sample.shape[-1]):
                eeg_out[sample][:, time] -= np.mean(eeg_sample[:, time])
                eeg_out[sample][:, time] /= eeg_out[sample][:, time].std()
                
    else:
        for sample, eeg_sample in enumerate(eeg):
            eeg_out[sample] = robust_minmax_scaler(eeg_sample)
            # Common average ref:
            for time in range(eeg_sample.shape[-1]):
                eeg_out[sample][:, time] -= torch.mean(eeg_sample[:, time])
    return eeg_out

def scale_source(sources):
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
    source_out = deepcopy(sources)
    # for sample in range(source.shape[0]):
    #     for time in range(source.shape[2]):
    #         # source_out[sample, :, time] /= source_out[sample, :, time].std()
    #         source_out[sample, :, time] /= np.max(np.abs(source_out[sample, :, time]))
    for sample, _ in enumerate(sources):
        # source_out[sample, :, time] /= source_out[sample, :, time].std()
        source_out[sample] /= np.max(np.abs(source_out[sample]))

    return source_out

def make_interpolator(elec_pos, res=9, ch_type='eeg', image_interp="linear"):
    extrapolate = _check_extrapolate('auto', ch_type)
    sphere = sphere = _check_sphere(None)
    outlines = 'head'
    outlines = _make_head_outlines(sphere, elec_pos, outlines, (0., 0.))
    border = 'mean'
    extent, Xi, Yi, interpolator = _setup_interp(
        elec_pos, res, image_interp, extrapolate, outlines, border)
    interpolator.set_locations(Xi, Yi)
    return interpolator

def interpolate_eeg(x_scaled, im_shape, info):
    elec_pos = _find_topomap_coords(info, info.ch_names)
    interpolator = make_interpolator(elec_pos, res=im_shape[0])
    x_scaled_interp = deepcopy(x_scaled)
    for i, sample in enumerate(tqdm(x_scaled, desc="interpolator")):
        list_of_time_slices = []
        for time_slice in sample:
            time_slice_interp = interpolator.set_values(time_slice)()[::-1]
            time_slice_interp = time_slice_interp[np.newaxis, :, :]# (1, height, width)
            list_of_time_slices.append(time_slice_interp)
        x_scaled_interp[i] = np.stack(list_of_time_slices, axis=0)
        x_scaled_interp[i][np.isnan(x_scaled_interp[i])] = 0
    x_scaled = x_scaled_interp
    del x_scaled_interp
    return x_scaled


def interpolate_single_eeg(x_scaled, interpolator):
    list_of_time_slices = []
    for time_slice in x_scaled:
        time_slice_interp = interpolator.set_values(time_slice)()[::-1]
        time_slice_interp = time_slice_interp[np.newaxis, :, :]# (1, height, width)
        list_of_time_slices.append(time_slice_interp)
    x_scaled = np.stack(list_of_time_slices, axis=0)
    x_scaled[np.isnan(x_scaled)] = 0
    return x_scaled