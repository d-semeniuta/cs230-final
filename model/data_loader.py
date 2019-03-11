import random
import os

from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
import h5py
import numpy as np


class SIGNSDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]

        self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]

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
