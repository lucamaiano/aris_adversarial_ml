"""
Contains functions for training and testing a PyTorch model.
"""

import torch
import numpy as np

from pathlib import Path

from art.estimators.classification import PyTorchClassifier
from art.utils import random_targets

from typing import Dict, List, Tuple
from tqdm.auto import tqdm

from attack.targeted import TargetedAttacks
from attack.untargeted import UntargetedAttacks

from data_setup import create_data_loader


def train(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.DataLoader,
    nb_classes,
    clip_values,
    input_shape,
    batch_size,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    use_art: bool,
) -> Dict[str, List[float]]:
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
    use_art: A flag that enables the art classifier.
    """

    x_train, y_train = train_dataset
    
    if use_art:
        # Create the ART classifier
        classifier = PyTorchClassifier(
            model=model,
            clip_values=clip_values,
            loss=loss_fn,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=nb_classes,
            device_type=device,
        )

        # Train the ART classifier
        classifier.fit(
            x_train, y_train, batch_size=batch_size, nb_epochs=epochs, verbose=True
        )
    else:
        classifier = model

        # Create a Data Loader
        dataloader = create_data_loader(
            x=x_train, y=y_train, batch_size=batch_size, shuffle=True
        )

        # Freeze the classifier
        # for param in classifier.parameters():
        #     param.requires_grad = False

        # Substitute last layer with current number of classes
        num_ftrs = classifier.classifier[-1].in_features
        classifier.classifier[-1] = torch.nn.Linear(num_ftrs, nb_classes)

        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(
                model=classifier,
                train_dataloader=dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )

            # Print out what's happening
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f}"
            )

    return classifier


def test(
    model_path: str, device: torch.device, use_art: bool
) -> Dict[str, List[float]]:
    if use_art:
        import art
        art.config.ART_DATA_PATH = Path(model_path).parent if isinstance(model_path, str) else model_path.parent
        
    # Load pretrained model
    classifier = torch.load(model_path, map_location=torch.device(device))

    return classifier


def attack(
    attack_name, x, y, nb_classes, classifier, logger, batch_size=32, targeted=False
):
    logger.info(f"Attacking the model with the {attack_name} attack")

    if targeted:
        y_target = random_targets(y, nb_classes)
        attack = TargetedAttacks(
            attack_name=attack_name,
            classifier=classifier,
            x=x,
            y_target=y_target,
            batch_size=batch_size,
        )
    else:
        attack = UntargetedAttacks(
            attack_name=attack_name, classifier=classifier, x=x, batch_size=batch_size
        )

    return attack


def train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(train_dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        y_true = torch.argmax(y, dim=1)
        train_acc += (y_pred_class == y_true).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)
    return train_loss, train_acc
