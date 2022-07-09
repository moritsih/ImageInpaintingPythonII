"""
Author: Moritz Haderer
Matr.Nr.: K11774793
Exercise 2
"""

import os
import numpy as np
from PIL import Image


class ImageStandardizer:


    def __init__(self, input_dir: str):

        self.mean = None
        self.std = None
        self.files = []
        jpg_exist = False

        for root, dir, images in os.walk(input_dir, topdown=True):

            for name in images:
                if name.endswith(".jpg"):
                    jpg_exist = True
                    self.files.append(os.path.join(root, name))

        if not jpg_exist:
            raise ValueError("No .jpg found!")

        self.files = sorted(self.files, reverse=False)



    def analyze_images(self):

        self.temp_mean = []
        self.temp_std = []

        for file in self.files:
            im = Image.open(file)
            im_array = np.asarray(im, dtype="float64")
            channel_mean = [np.mean(im_array[:,:,i], dtype="float64") for i in range(im_array.shape[-1])]
            channel_std = [np.std(im_array[:,:,i], dtype="float64") for i in range(im_array.shape[-1])]
            self.temp_mean.append(channel_mean)
            self.temp_std.append(channel_std)

        self.mean = np.mean(self.temp_mean, axis=0, dtype="float64")
        self.std = np.mean(self.temp_std, axis=0, dtype="float64")

        return (self.mean, self.std)



    def get_standardized_images(self):

        if np.any(self.mean == None) or np.any(self.std == None):
            raise ValueError("Mean or Standard deviation are not available!")

        for file in self.files:
            im = Image.open(file)
            im_array = np.asarray(im, dtype="float32")
            im_array[:,:] = im_array[:,:] - self.mean
            im_array[:,:] = im_array[:,:] / self.std

            yield im_array

