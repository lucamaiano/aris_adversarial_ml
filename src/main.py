# Adding Hydra support
import logging

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from hydra.utils import to_absolute_path
from pathlib import Path

import torch
import copy
import data_setup, engine, utils
from model.baseline import BaseCNN


# Logger
log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(conf: DictConfig):
    log.info(OmegaConf.to_yaml(conf))

    # Train the model
    train(conf)


def train(conf: DictConfig) -> None:
    # Setup hyperparameters
    NUM_EPOCHS = conf.model.num_epochs
    BATCH_SIZE = conf.model.batch_size
    LEARNING_RATE = conf.model.learning_rate
    INPUT_SHAPE = tuple(conf.dataset.input_shape)
    NB_CLASSES = conf.dataset.nb_classes
    MODEL_PATH = to_absolute_path(
        Path("models", conf.model.name, conf.model.model_path)
    )

    # Load data
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = (
        data_setup.load_data(conf.dataset.name)
    )
    
    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    
    # Create the model
    if conf.model.name.lower() == "base_cnn":
        model = BaseCNN()
    elif conf.model.name.lower() == "pretrained":
        model = None
    else:
        raise NotImplementedError

    # Start training with the engine
    if not conf.model.load_pretrained:
        # Move to target device
        model.to(device)

        # Set loss and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        classifier = engine.train(
            model=model,
            x_train=x_train,
            y_train=y_train,
            nb_classes=NB_CLASSES,
            clip_values=(min_pixel_value, max_pixel_value),
            input_shape=INPUT_SHAPE,
            batch_size=BATCH_SIZE,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=NUM_EPOCHS,
            device=device,
        )
    else:
        classifier = engine.test(
            model_path=MODEL_PATH,
            device=device
        )

    # Evaluate the original model
    predictions, accuracy = utils.evaluate_model(classifier, x_test, y_test, log)

    # Save the model with help from utils.py
    utils.save_model(
        model=classifier,
        target_dir=to_absolute_path(Path("models", conf.model.name)),
        model_name=conf.model.model_path,
    )

    # Attack the model
    classifier_adv = copy.copy(classifier)
    attack = engine.attack(
        attack_name=conf.attack.name,
        x=x_test,
        y=y_test,
        nb_classes=NB_CLASSES,
        classifier=classifier_adv,
        logger=log,
        batch_size=BATCH_SIZE,
        targeted=conf.attack.targeted,
    )

    # Evaluate the attacked model metrics
    predictions_adv, accuracy_adv = utils.evaluate_model(
        classifier=classifier_adv, x=attack.x_adv, y=y_test, logger=log, attacked=True
    )

    # Evaluate the attacked model on robustness metrics
    utils.evaluate_robustness(
        classifier=classifier,
        predictions=predictions_adv,
        x_test=x_test,
        x_test_adv=attack.x_adv,
        y_test=y_test,
        y_target=attack.y_target if conf.attack.targeted else None,
        targeted=conf.attack.targeted,
        dataset=conf.dataset.name,
        save_dir=to_absolute_path(Path("output/compressed_imgs")),
        logger=log,
    )


if __name__ == "__main__":
    main()
