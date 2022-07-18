'''
NEW INFERENCE
'''

import os
import sys
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

colab_path = os.path.join("gdrive", "MyDrive", "Colab")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
sys.path.append(colab_path)

ch_mean = torch.tensor([124.771666, 120.852102, 113.64321009])/255
ch_std = torch.tensor([51.91158522, 51.73857714, 55.01331596])/255



tfm = transforms.Compose([transforms.ToPILImage(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=ch_mean, std=ch_std)])

tfm_inverse = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/(ch_std[0]), 1/ch_std[1], 1/ch_std[2]]),
                                transforms.Normalize(mean = [ -ch_mean[0], -ch_mean[1], -ch_mean[2]],
                                                     std = [ 1., 1., 1. ])])


def stack_known_to_input_normalization(input_array, known_array):
    return torch.concat((input_array, known_array), 0)


class TestData(Dataset):
    def __init__(self, dataset: Dataset,
                 transform_chain: transforms.Compose = None,
                 transform_chain_inv: transforms.Compose = None):
        with open(dataset, 'rb') as f:
            data_dict = pickle.load(f)

        self.transform = transform_chain

        self.input_arrays = data_dict["input_arrays"] # 1-255; grid cut out
        self.known_arrays = data_dict["known_arrays"]
        self.offsets = data_dict["offsets"]
        self.spacings = data_dict["spacings"]
        self.ids = data_dict["sample_ids"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        known_array = self.known_arrays[idx]
        input_array = self.input_arrays[idx]
        #print(type(input_array), type(known_array)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
        #print(input_array.shape, known_array.shape) #(3, 100, 100) (1, 100, 100)

        if self.transform is not None:
            input_array = self.transform(np.moveaxis(input_array, 0, 2))

        known_array = torch.from_numpy(known_array)

        # input_array is normalized by mean and std and is tensor
        stacked_array = stack_known_to_input_normalization(input_array, known_array[0:1, :, :])


        #stacked_array = stack_known_to_input(self.input_arrays, self.known_arrays, idx)
        #print(stacked_array.shape)
        #print(stacked_array[1,50:70, 50:70])

        return stacked_array, known_array, idx

test_data_file = os.path.join(colab_path, "test", "inputs.pkl")
test_data = TestData(test_data_file, transform_chain=tfm)

testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)


def get_test_predictions(model, test_loader, device, transform):
    model = model.double()
    model.eval( )
    targets = []

    with torch.no_grad():
        for data in tqdm(test_loader, position=0):

            input_stacked, known_array, idx = data
            known_array = known_array.to(device).double()
            inputs = input_stacked.to(device).double()
            #print(type(inputs), inputs.shape)

            output = model(inputs) # get prediction: (batch, 4, 100, 100) to (batch, 3, 100, 100)

            output_denorm = transform(output) # apply inverse mean&std transform to get image

            known_array = known_array.detach().cpu().numpy()
            target_array = output_denorm.detach().cpu().numpy()[known_array == 0] * 255

            #print(target_array.shape)

            target_array = target_array.astype(np.uint8)

            targets.append(target_array)

    model.train()
    return targets

saved_model_file = os.path.join(colab_path, "results", "best_model.pt")
net = torch.load(saved_model_file, map_location=dev)

preds = get_test_predictions(net, testloader, device=dev, transform = tfm_inverse)

with open(os.path.join(colab_path, "results", 'test_preds.pickle'), 'wb') as f:
    pickle.dump(preds, f, protocol=pickle.HIGHEST_PROTOCOL)

