import json
import os

import h5py
import numpy as np
import torch
from imageio.core import image_as_uint
from torch.utils.data import Dataset
from .utils import get_dataset_path

class BrainDataset(Dataset):
    def __init__(self, brains):
        self.brains = list(brains)
        path = get_dataset_path()
        self.data = h5py.File(os.path.join(path,"cell_data.h5"), 'r')
        self.stats = json.load(open(os.path.join(path, 'stats.json')))
        self.length = sum([len(self.data[brain]) for brain in self.brains])

    def __getitem__(self, index):
        brain, image, row, column, tile_size = index
        image = self.__convert_index__(image)
        assert brain in self.brains, "Attempting to access brain that's not in the dataset"
        brain_image = torch.tensor(np.array(self.data[brain][image]), dtype=torch.float32)
        brain_image = brain_image[row:row+tile_size, column:column+tile_size]
        brain_image = (brain_image - self.stats["mean"]) / self.stats["std"]
        return brain_image.unsqueeze(0)

    def __len__(self):
        return self.length

    def get_brains(self):
        return self.brains

    def get_length_of_brain(self, brain):
        return len(self.data[brain])

    def get_pixel_count(self, brain, image):
        image = self.__convert_index__(image)
        return np.prod(self.data[brain][image].shape)

    def get_shape(self, brain, image):
        image = self.__convert_index__(image)
        return self.data[brain][image].shape

    def __convert_index__(self, i):
        return f"{i}".zfill(4)
