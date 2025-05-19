# svmtrainer_project/README.md

# SVM Trainer (`svmtrainer`)

A Python package for easily training Support Vector Machine (SVM) models for classification tasks.
It supports linear, RBF, and polynomial kernels, and includes features for 5-fold
cross-validation and hyperparameter tuning using scikit-learn.

## Features

* Train SVM models with 'linear', 'rbf', or 'poly' kernels.
* Automatic 5-fold cross-validation for hyperparameter tuning.
* Hyperparameter search for C, gamma, degree, and coef0 where applicable.
* Optional feature scaling using StandardScaler.
* Simple API for training, prediction, and model evaluation.
* Save and load trained models.

## Installation

```bash
pip install .
