from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
from mne.channels.layout import _find_topomap_coords
from mne.viz.topomap import (_setup_interp, _make_head_outlines, _check_sphere, 
    _check_extrapolate)
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr


def robust_minmax_scaler(eeg: np.ndarray) -> np.ndarray:
    lower, upper = [np.percentile(eeg, 25), np.percentile(eeg, 75)]
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
                eeg_out[sample][:, time] -= np.mean(eeg_sample[:, time])
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







########## functions for brents method

# this function operates on a single time sample
def solve_p(y_est, x_true, leadfield):
    '''
    Parameters
    ---------
    y_est : numpy.ndarray
        The estimated source vector.
    x_true : numpy.ndarray
        The original input EEG vector.
    
    Return
    ------
    y_scaled : numpy.ndarray
        The scaled estimated source vector.
    
    '''
    # Check if y_est is just zeros:
    if np.max(y_est) == 0:
        return y_est
    y_est = np.squeeze(np.array(y_est))
    x_true = np.squeeze(np.array(x_true))
    # Get EEG from predicted source using leadfield
    x_est = np.matmul(leadfield, y_est)

    # optimize forward solution
    tol = 1e-9
    options = dict(maxiter=1000, disp=False)

    # base scaling
    rms_est = np.mean(np.abs(x_est))
    rms_true = np.mean(np.abs(x_true))
    base_scaler = rms_true / rms_est

    
    opt = minimize_scalar(correlation_criterion, args=(leadfield, y_est* base_scaler, x_true), bracket=(0,1), method='Brent', options=options, tol=tol)
    
    # opt = minimize_scalar(self.correlation_criterion, args=(self.leadfield, y_est* base_scaler, x_true), \
    #     bounds=(0, 1), method='L-BFGS-B', options=options, tol=tol)

    scaler = opt.x
    y_scaled = y_est * scaler * base_scaler
    return y_scaled

def correlation_criterion(scaler, leadfield, y_est, x_true):
    ''' Perform forward projections of a source using the leadfield.
    This is the objective function which is minimized in Net::solve_p().
    
    Parameters
    ----------
    scaler : float
        scales the source y_est
    leadfield : numpy.ndarray
        The leadfield (or sometimes called gain matrix).
    y_est : numpy.ndarray
        Estimated/predicted source.
    x_true : numpy.ndarray
        True, unscaled EEG.
    '''

    x_est = np.matmul(leadfield, y_est*scaler) 
    error = np.abs(pearsonr(x_true-x_est, x_true)[0])
    return error