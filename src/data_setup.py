"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""

import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from art.utils import load_mnist, load_cifar10
from torch.utils.data import TensorDataset, DataLoader

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def load_data(
    dataset: str, batch_size: int = None, train_data: str = None, test_data: str = None
):
    # Load the dataset
    min_pixel_value, max_pixel_value = None, None
    if dataset.lower() == "mnist":
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = (
            load_mnist()
        )
        # Swap axes to PyTorch's NCHW format
        x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

        return (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value

    elif dataset.lower() == "cifar10":
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = (
            load_cifar10()
        )
        # Swap axes to PyTorch's NCHW format
        x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

        return (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value

    elif dataset.lower() == "classification":
        # Define the transformations to apply to the images
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load the train and validation datasets
        train_dataset = ImageFolder(train_data, transform=transform)
        test_dataset = ImageFolder(test_data, transform=transform)

        # Create data loaders for the train and test datasets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, 0, 255
    else:
        raise NotImplementedError


def create_data_loader(x, y, batch_size, shuffle=True):
    if type(x[0]) == np.ndarray and type(y[0]) == np.ndarray:
        x = torch.tensor(np.stack(x))
        y = torch.tensor(np.stack(y))

    # Create a TensorDataset
    dataset = TensorDataset(x, y)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
