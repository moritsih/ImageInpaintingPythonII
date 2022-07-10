"""
Author: Moritz Haderer
Matr.Nr.: K11774793
Exercise 5
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch

def set_random_spacing_offset(SEED):
    np.random.seed(SEED)

    #offsets = [0,1,2,3,4,5,6,7,8]
    #spacings = [2,3,4,5,6]

    return (np.random.randint(0,8), np.random.randint(0,8)), \
           (np.random.randint(2,6), np.random.randint(2,6))


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear( )
            ax.set_title(title)
            ax.imshow(data[i, 0], interpolation="none")
            ax.set_axis_off( )
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)

    plt.close(fig)


def plot_img(img):
    if type(img) == "torch.Tensor":
        img = torch.permute(img, (1, 2, 0))

    #img = TF.to_pil_image(img)
    plt.imshow(img)
    plt.show()