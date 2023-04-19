import os

import numpy as np
from torch.utils import data


class CrowdDataset(data.Dataset):
    def __init__(self, mode='train', dataset='labelme'):
        data_path = f'data/{dataset}/{mode}/'
        X = np.load(os.path.join(data_path, 'data.npy')).astype(np.float32)
        self.y = np.load(os.path.join(data_path, 'labels.npy')).astype(np.int64)
        self.X = X.reshape(self.y.size, -1)
        self.mode = mode
        if self.mode == 'train':
            self.anno = np.load(os.path.join(data_path, 'annotations.npy')).astype(np.int64)
            classes = np.unique(self.anno)
            self.num_classes = len(classes) - 1 if -1 in classes else len(classes)
            self.input_size = self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.mode == 'train':
            return idx, self.X[idx], self.anno[idx], self.y[idx]
        else:
            return idx, self.X[idx], self.y[idx]
