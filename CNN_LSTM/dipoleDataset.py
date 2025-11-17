import torch

class DipoleDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_data, source_labels, im_shape=(7,11)):
        self.eeg_data = eeg_data
        self.source_labels = source_labels
        self.im_shape = im_shape

    # get sample
    def __getitem__(self, idx):
        eeg_sample = self.eeg_data[idx]  # shape: (n_timesteps, height, width)

        # convert to tensor
        eeg_sample = torch.Tensor(eeg_sample)
        
        # get the label, in this case the label was noted in the name of the image file, ie: 1_image_28457.png where 1 is the label and the number at the end is just the id or something
        target = self.source_labels[idx]  # shape: (n_dipoles, )
        # convert to tensor
        target = torch.Tensor(target)

        return eeg_sample, target

    def __len__(self):
        return len(self.eeg_data)