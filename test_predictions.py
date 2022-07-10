"""
Author: Moritz Haderer
Matr.Nr.: K11774793
Exercise 5
"""

import os
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
import pickle

def stack_known_to_input(input_arrays, known_arrays, idx):
    known_array = TF.to_tensor(known_arrays[idx][0, :, :])
    input_array = torch.transpose(TF.to_tensor(input_arrays[idx]), 1, 0)

    return torch.concat((input_array, known_array), 0)


class TestData(Dataset):
    def __init__(self, test_data):
        with open(test_data, 'rb') as f:
            data_dict = pickle.load(f)

        self.input_arrays = data_dict["input_arrays"]
        self.known_arrays = data_dict["known_arrays"]
        self.offsets = data_dict["offsets"]
        self.spacings = data_dict["spacings"]
        self.ids = data_dict["sample_ids"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        known_array = self.known_arrays[idx]
        stacked_array = stack_known_to_input(self.input_arrays, self.known_arrays, idx)

        return stacked_array, known_array, idx

test_data_file = os.path.join("test", "inputs.pkl")
test_data = TestData(test_data_file)

testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)


def get_test_predictions(net, test_loader):

    for data in test_loader:

        input_stacked, known_array, idx = data
        targets = []

        output = net(input_stacked)
        target_array = output[known_array == 0]
        targets.append(target_array)

    return targets


saved_model_file = os.path.join("results", "best_model.pt")
net = torch.load(saved_model_file)

preds = get_test_predictions(net, testloader)

with open('test_preds.pickle', 'wb') as f:
    pickle.dump(preds, f, protocol=pickle.HIGHEST_PROTOCOL)