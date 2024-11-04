import torch
from torch.utils.data.sampler import Sampler

from data_utils.BrainDataset import BrainDataset


class BrainSampler(Sampler):
    def __init__(self, data, tile_size, weighting="equal"):
        assert weighting in ["equal", "by_pixel"]
        self.data = data
        self.brains = self.data.get_brains()
        self.tile_size = tile_size

        if weighting == "equal":
            self.brain_prob = torch.tensor([1 for brain in self.brains], dtype=torch.float32)
            self.image_prob = {brain:torch.tensor([1 for i in range(self.data.get_length_of_brain(brain))], dtype=torch.float32) for brain in self.brains}
        else:
            pixels_per_brain = {}
            for brain in self.brains:
                pixels_per_brain[brain] = []
                for i in range(self.data.get_length_of_brain(brain)):
                    pixels_per_brain[brain].append(self.data.get_pixel_count(brain, i))
            total_pixels_per_brain = {brain:sum(pixels_per_brain[brain]) for brain in self.brains}
            total_pixels = sum([total_pixels_per_brain[brain] for brain in self.brains])
            self.brain_prob = torch.tensor([total_pixels_per_brain[brain]/total_pixels for brain in self.brains])
            self.image_prob = {brain:torch.tensor(pixels_per_brain[brain])/torch.tensor(total_pixels_per_brain[brain]) for brain in self.brains}


    def __iter__(self):
        while True:
            brain = self.brains[torch.multinomial(self.brain_prob, 1)]
            image = torch.multinomial(self.image_prob[brain], 1).item()
            row = torch.randint(self.data.get_shape(brain, image)[0]-self.tile_size, (1,)).item()
            column = torch.randint(self.data.get_shape(brain, image)[1]-self.tile_size, (1,)).item()
            yield (brain, image, row, column, self.tile_size)

    def __len__(self):
        return float('inf')