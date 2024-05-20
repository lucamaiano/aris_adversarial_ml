"""
Contains various utility functions for PyTorch model training and saving.
"""
from pathlib import Path

import torch
import numpy as np

from metric.metrics import RobustnessMetrics
from sklearn import metrics
from scipy.special import softmax

def evaluate_robustness(classifier, predictions, x_test, x_test_adv, y_test, y_target, targeted, dataset, save_dir, logger):
  metrics = RobustnessMetrics(
    classifier=classifier,
    predictions=predictions, 
    x_test=x_test,
    x_test_adv=x_test_adv,
    y_test=y_test, 
    y_target=y_target, 
    targeted=targeted,
    save_dir=save_dir,
    dataset=dataset)

  # Missclassification
  mr = metrics.misclassification_rate()
  logger.info(f"MR: {mr*100:.3f}%")

  acac = metrics.avg_confidence_adv_class()
  logger.info(f"ACAC: {acac:.3f}")

  actc = metrics.avg_confidence_true_class()
  logger.info(f"ACAC: {actc:.3f}")
  
  # Imperceptibility
  alp = metrics.avg_lp_distortion()
  logger.info(f"ALP L0: {alp[0]:.3f}, ALP L1: {alp[1]:.3f} ALP Linf: {alp[2]:.3f}")

  ass = metrics.avg_SSIM()
  logger.info(f"ASS: {ass:.3f}")
  
  # Robustness
  nte = metrics.avg_noise_tolerance_estimation()
  logger.info(f"NTE: {nte:.3f}")

  rgb = metrics.robust_gaussian_blur()
  logger.info(f"RGB: {rgb:.3f}")

  ric = metrics.robust_image_compression()
  logger.info(f"RIC: {ric:.3f}")

def evaluate_model(classifier, x, y, logger, attacked=False):
  predictions = classifier.predict(x)
  
  accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / len(y)
  auc = metrics.roc_auc_score(y, softmax(predictions), multi_class='ovo')
  if attacked:
    logger.info("Accuracy on attacked test examples: {:.3f}%".format(accuracy * 100))
    logger.info("AUC on attacked test examples: {:.3f}%".format(auc * 100))
  else:
    logger.info("Accuracy on benign test examples: {:.3f}%".format(accuracy * 100))
    logger.info("AUC on benign test examples: {:.3f}%".format(auc * 100))

  return predictions, accuracy
    

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
  model: A target PyTorch model to save.
  target_dir: A directory for saving the model to.
  model_name: A filename for the saved model. Should include
    either ".pth" or ".pt" as the file extension.

  Example usage:
  save_model(model=model_0,
              target_dir="models",
              model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                      exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {target_dir}")
  torch.save(model, Path(target_dir, model_name))
