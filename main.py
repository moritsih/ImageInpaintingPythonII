"""
Author: Moritz Haderer
Matr.Nr.: K11774793
Exercise 5
"""

import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import glob
from ex4 import ex4
from ex3 import ImageStandardizer
from data import DataExtraction, InpaintingData
from model import SimpleCNN
from tqdm import tqdm


'''
Calculating training set mean & std using ex3:
normalizer = ImageStandardizer(DIR)
ch_mean, ch_std = normalizer.analyze_images()
'''
ch_mean = [124.771666, 120.852102, 113.64321009]
ch_std = [51.91158522, 51.73857714, 55.01331596]

im_shape = (100, 100)
DIR = "training"
#transforms.Normalize(mean=ch_mean,std=ch_std)
tfm = transforms.Compose([transforms.Resize(size=im_shape),
                          transforms.ToTensor()])


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`,
    using the specified `loss_fn` loss function"""
    model.eval( )
    # We will accumulate the mean loss in variable `loss`
    loss = 0
    with torch.no_grad( ):  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            targets, inputs, idx = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get outputs of the specified model
            outputs = model(inputs)

            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance

            # Add the current loss, which is the mean loss over all minibatch samples
            # (unless explicitly otherwise specified when creating the loss function!)
            loss += loss_fn(outputs, targets).item( )
    # Get final mean loss by dividing by the number of minibatch iterations (which
    # we summed up in the above loop)
    loss /= len(dataloader)
    model.train( )
    return loss

def main(results_path, network_config: dict, learningrate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = 50_000, device: torch.device = torch.device("cuda:0")):

    data_paths_by_idx = DataExtraction(root_dir=DIR)

    is_it_working = torch.utils.data.Subset(
        data_paths_by_idx,
        indices=np.arange(30))

    trainingset = torch.utils.data.Subset(
        data_paths_by_idx,
        indices=np.arange(30, int(len(data_paths_by_idx) * (3 / 5))))

    validationset = torch.utils.data.Subset(
        data_paths_by_idx,
        indices=np.arange(int(len(data_paths_by_idx) * (3 / 5)),
                          int(len(data_paths_by_idx) * (4 / 5))))

    testset = torch.utils.data.Subset(
        data_paths_by_idx,
        indices=np.arange(int(len(data_paths_by_idx) * (4 / 5)),
                          len(data_paths_by_idx)))

    is_it_working = InpaintingData(dataset=is_it_working, transform_chain=tfm)
    trainingset = InpaintingData(dataset=trainingset, transform_chain=tfm)
    validationset = InpaintingData(dataset=validationset, transform_chain=tfm)
    testset = InpaintingData(dataset=testset, transform_chain=tfm)

    is_it_working = torch.utils.data.DataLoader(is_it_working, batch_size=2, shuffle=False, num_workers=0)#, collate_fn=custom_collate)
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=2, shuffle=False, num_workers=0)#, collate_fn=custom_collate)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=2, shuffle=False, num_workers=0)#, collate_fn=custom_collate)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=0)#, collate_fn=custom_collate)

    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))

    net = SimpleCNN(**network_config)
    net.to(device)

    # Get mse loss function
    mse = torch.nn.MSELoss( )

    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters( ), lr=learningrate, weight_decay=weight_decay)

    print_stats_at = 100  # print status to tensorboard every x updates
    plot_at = 10_000  # plot every x updates
    validate_at = 5000  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)

    # Train until n_updates updates have been reached
    while update < n_updates:
        for data in is_it_working:

            '''
            full_image = original image without grid cut out
            image_array = image with specified grid set to 0
            known_array = array of ones (same shape as image_array) where the same specified grid is set to 0
            '''

            full_image, image_array_stacked, idx = data

            full_image = full_image.to(device)
            image_array_stacked = image_array_stacked.to(device)

            # Reset gradients
            optimizer.zero_grad( )

            # Get outputs of our network
            outputs = net(image_array_stacked)

            # Calculate loss, do backward pass and update weights
            loss = mse(outputs, full_image)
            loss.backward( )
            optimizer.step( )

            # Print current status and score
            if (update + 1) % print_stats_at == 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu( ), global_step=update)

            # Plot output
            #if (update + 1) % plot_at == 0:
             #   plot(inputs.detach( ).cpu( ).numpy( ), targets.detach( ).cpu( ).numpy( ),
              #       outputs.detach( ).cpu( ).numpy( ),
               #      plotpath, update)

            # Evaluate model on validation set
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(net, dataloader=valloader, loss_fn=mse, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
                # Add weights and gradients as arrays to tensorboard
                for i, (name, param) in enumerate(net.named_parameters( )):
                    writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param.cpu( ), global_step=update)
                    writer.add_histogram(tag=f"validation/gradients_{i} ({name})", values=param.grad.cpu( ),
                                         global_step=update)
                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(net, saved_model_file)

            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update( )

            # Increment update counter, exit if maximum number of updates is reached
            # Here, we could apply some early stopping heuristic and also exit if its
            # stopping criterion is met
            update += 1
            if update >= n_updates:
                break

    update_progress_bar.close( )
    writer.close( )
    print("Finished Training!")

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)
    train_loss = evaluate_model(net, dataloader=trainloader, loss_fn=mse, device=device)
    val_loss = evaluate_model(net, dataloader=valloader, loss_fn=mse, device=device)
    test_loss = evaluate_model(net, dataloader=testloader, loss_fn=mse, device=device)

    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"      test loss: {test_loss}")

    # Write result to file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"      test loss: {test_loss}", file=rf)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser( )
    parser.add_argument("config_file", type=str, help="Path to JSON config file")
    args = parser.parse_args( )

    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)

