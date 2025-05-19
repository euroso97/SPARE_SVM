# svmtrainer_project/svmtrainer/__init__.py
from .model import SVMModel
from .tuner import tune_svm_hyperparameters
from .utils import load_tabular_data # If you add this function

__version__ = "0.1.0"
__all__ = ['SVMModel', 'tune_svm_hyperparameters', 'load_tabular_data']