from util import *
import mne
from tqdm import tqdm
import os

im_shape = (9,9)
eeg_data_path = "/mnt/data/convdip/training_data/eeg_data"
info = mne.io.read_info("/mnt/data/convdip/training_data/info.fif")
eeg_interpolated_path = "/mnt/data/convdip/training_data/interpolated_eeg_data_for_lstm"

x_scaled = []
for r, d, files in os.walk(eeg_data_path):
    for file in tqdm(files, desc="loader"):
        x_scaled.append(np.load(os.path.join(eeg_data_path, file)))

x_scaled = scale_eeg(x_scaled, scale_individually=False)
x_scaled = [np.swapaxes(x,0,1) for x in x_scaled]
x_scaled = interpolate_eeg(x_scaled, im_shape, info)

for i, x in enumerate(tqdm(x_scaled, desc="saver")):
    np.save(os.path.join(eeg_interpolated_path, "sample_{}.npy".format(i)), x_scaled[i])
