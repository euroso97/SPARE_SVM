# svmtrainer_project/svmtrainer/__init__.py
from .model import SVMTrainer
from .tuner import tune_svm_classifier_hyperparameters
from .utils import load_tabular_data # If you add this function

__version__ = "0.1.0"
__all__ = ['SVMTrainer', 
           'tune_svm_regressor_hyperparameters', 
           'tune_svm_classifier_hyperparameters', 
           'load_tabular_data']