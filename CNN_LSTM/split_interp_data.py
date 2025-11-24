import numpy as np
import os
from tqdm import tqdm
interp_data_file = "/mnt/data/convdip/training_data/sample_1.npy"
save_path = "/mnt/data/convdip/training_data/interp_data/"
data = np.load(interp_data_file)
for i in tqdm(range(data.shape[0])):
    np.save(os.path.join(save_path, "sample_{}.npy".format(i)), data[i])