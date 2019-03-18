import random
import os

from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
import h5py
import numpy as np

class VCTKDataset(Dataset):
    def __init__(self, datah5, debug=False):

        self.X, self.Y = load_h5(datah5)
        if debug:
            self.X = self.X[:10,:,:]
            self.Y = self.X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def load_h5(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        X = np.array(hf.get('data'))
        Y = np.array(hf.get('label'))
    return X, Y

def fetch_dataloader(types, data_paths, params, debug=False):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending
            on which data is required
        data_paths: (dict) dict of files of the dataset
        params: (Params) hyperparameters
        debug: (boolean) if true, single batch size with train and val
            of same size

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in types:
        if split in types:
            # path = os.path.join(data_dir, "{}_signs".format(split))
            path = data_paths[split]

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(VCTKDataset(path, debug=debug),
                                batch_size=params.batch_size,
                                shuffle=True,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)
            else:
                dl = DataLoader(VCTKDataset(path, debug=debug),
                                batch_size=params.batch_size,
                                shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
