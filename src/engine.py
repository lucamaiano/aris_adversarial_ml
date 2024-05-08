"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from art.estimators.classification import PyTorchClassifier
from art.utils import random_targets

from typing import Dict, List, Tuple
from tqdm.auto import tqdm

from attack.targeted import TargetedAttacks
from attack.untargeted import UntargetedAttacks


def train(model: torch.nn.Module, 
        x_train: torch.utils.data.DataLoader, 
        y_train: torch.utils.data.DataLoader, 
        nb_classes,
        clip_values,
        input_shape,
        batch_size,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: torch.device) -> Dict[str, List[float]]:
    """
        Trains and tests a PyTorch model.

        Passes a target PyTorch models through train_step() and test_step()
        functions for a number of epochs, training and testing the model
        in the same epoch loop.

        Calculates, prints and stores evaluation metrics throughout.

        Args:
        model: A PyTorch model to be trained and tested.
        x_train: A DataLoader instance for the model to be trained on.
        y_train: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
    """
    # Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=nb_classes,
        device_type=device
    )
    
    # Train the ART classifier
    classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs, verbose=True)
    
    return classifier

def test(model: torch.nn.Module, 
        model_path: str,
        x_train: torch.utils.data.DataLoader, 
        y_train: torch.utils.data.DataLoader, 
        nb_classes,
        clip_values,
        input_shape,
        batch_size,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: torch.device) -> Dict[str, List[float]]:
    # Load pretrained model
    model.load_state_dict(torch.load(model_path))
    
    # Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=nb_classes,
        device_type="gpu" if device == "cuda" else "cpu"
    )
    
    return classifier

def attack(attack_name, x, y, nb_classes, classifier, logger, batch_size=32, targeted=False):
    logger.info(f"Attacking the model with the {attack_name} attack")
    
    if targeted:
        y_target = random_targets(y, nb_classes)
        attack = TargetedAttacks(attack_name=attack_name, classifier=classifier, x=x, y_target=y_target, batch_size=batch_size)
    else:
        attack = UntargetedAttacks(attack_name=attack_name, classifier=classifier, x=x, batch_size=batch_size)
        
    return attack