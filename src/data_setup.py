"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from art.utils import load_mnist

NUM_WORKERS = os.cpu_count()

def load_data(dataset: str):
  # Load the dataset
  min_pixel_value, max_pixel_value = None, None
  if dataset.lower() == "mnist":
      (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
      
      # Swap axes to PyTorch's NCHW format
      x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
      x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
  else:
      raise NotImplementedError
    
  return (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value
