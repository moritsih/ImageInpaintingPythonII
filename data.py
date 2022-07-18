"""
Author: Moritz Haderer
Matr.Nr.: K11774793
Exercise 5
"""

from utils import plot_img
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import glob
from ex4 import ex4
from utils import set_random_spacing_offset

class DataExtraction(Dataset):

    def __init__(self, root_dir):
        self.img_folder = glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True)

    def __len__(self):
        return len(self.img_folder)

    def __getitem__(self, idx):

        path = self.img_folder[idx]
        return path, idx


class InpaintingData(Dataset):

    def __init__(self, dataset: Dataset, transform_chain: transforms.Compose = None):

        self.transform = transform_chain
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, idx = self.dataset[idx]

        im = Image.open(im)

        if self.transform is not None:
            im = self.transform(im)

        # full_image shape: (3, 100, 100)
        full_image = torch.clone(im)

        offset, spacing = set_random_spacing_offset(idx)

        image_array, known_array, target_array = ex4(np.moveaxis(im.numpy(), 0, 2), offset, spacing)

        known_array = known_array[0:1,:,:]

        image_array_stacked = np.array(np.concatenate((image_array, known_array), 0), dtype=np.float32)

        return full_image, image_array_stacked, idx

'''
# >>> debugging <<<
### mean and std calculated using ex3: ImageStandardizer - mean&std over entire training set
ch_mean = torch.tensor([124.771666, 120.852102, 113.64321009])/255
ch_std = torch.tensor([51.91158522, 51.73857714, 55.01331596])/255

im_shape = (100, 100)
tfm = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize(mean=ch_mean, std=ch_std),
                          transforms.Resize(size=im_shape)])

                          
DIR = "training"
data_paths_by_idx = DataExtraction(root_dir=DIR)

trainingset = torch.utils.data.Subset(
    data_paths_by_idx,
    indices=np.arange(int(len(data_paths_by_idx) * (3 / 5))))

trainingset = InpaintingData(dataset=trainingset, transform_chain=tfm)

trainloader = torch.utils.data.DataLoader(trainingset, batch_size=2, shuffle=False, num_workers=0)

for data in trainloader:
    print(data[0][0][1])
    print(data[1][0][1])
    break

'''


