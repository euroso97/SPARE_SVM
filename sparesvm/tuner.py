import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, RepeatedKFold
from sklearn.svm import LinearSVR, SVR, SVC
from .utils import expspace

def tune_svm_regressor_hyperparameters(X, 
                                       y,
                                       kernel='rbf', 
                                       random_state=None, 
                                       cv_method='KFold', # options: KFold, StratifiedKFold, RepeatedKFold
                                       cv_folds=5, 
                                       return_fold=False):
    """
    Performs 5-fold cross-validation and hyperparameter tuning for an SVM model.

    Args:
        X (pd.DataFrame or np.ndarray): Training features.
        y (pd.Series or np.ndarray): Training labels.
        kernel (str): Type of SVM kernel ('linear', 'rbf', 'poly').
        cv_folds (int): Number of cross-validation folds.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (best_model, best_params, best_score)
               - best_model: The trained SVC model with the best hyperparameters.
               - best_params: Dictionary of the best hyperparameters found.
               - best_score: The mean cross-validated score of the best_estimator.
    """
    param_grid = {}
    if kernel == 'linear':
        param_grid = {
            "C": expspace([-9, 3])
        }
        svr = LinearSVR(loss='epsilon_insensitive') # probability=True for predict_proba if needed

    elif kernel == 'rbf':
        param_grid = {
            "C": expspace([-9, 3]),
            'gamma': ['scale', 'auto']
        }
        svr = SVR(kernel=kernel, shrinking=True) # probability=True for predict_proba if needed

    elif kernel == 'poly':
        param_grid = {
            "C": expspace([-9, 3]),
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'],
            'coef0': [0.0, 0.1, 0.5, 1.0] # Independent term in kernel function
        }
        svr = SVR(kernel=kernel, shrinking=True) # probability=True for predict_proba if needed

    else:
        raise ValueError(f"Unsupported kernel: {kernel}. Choose from 'linear', 'rbf', 'poly'.")

    
    # Define cross-validation strategy
    if cv_method=='KFold':
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    elif cv_method=='StratifiedKFold':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    elif cv_method=='RepeatedKFold':
        cv = RepeatedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Setup GridSearchCV
    # Scoring can be 'accuracy', 'f1', 'roc_auc', etc. depending on the problem
    
    grid_search = GridSearchCV(estimator=svr, 
                            param_grid=param_grid, 
                            cv=cv,
                            scoring='r2', # explaining variance is key
                            verbose=3, 
                            n_jobs=-1) # n_jobs=-1 uses all available cores

    # Fit GridSearchCV
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters for {kernel} kernel: {best_params}")
    print(f"Best cross-validation accuracy: {best_score:.4f}")

    return best_model, best_params, best_score


def tune_svm_classifier_hyperparameters(X, 
                                        y, 
                                        kernel='rbf', 
                                        random_state=None, 
                                        class_balancing=True,
                                        cv_method='KFold', # options: KFold, StratifiedKFold, RepeatedKFold
                                        cv_folds=5, 
                                        return_fold=False):
    """
    Performs 5-fold cross-validation and hyperparameter tuning for an SVM model.

    Args:
        X (pd.DataFrame or np.ndarray): Training features.
        y (pd.Series or np.ndarray): Training labels.
        kernel (str): Type of SVM kernel ('linear', 'rbf', 'poly').
        cv_folds (int): Number of cross-validation folds.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (best_model, best_params, best_score)
               - best_model: The trained SVC model with the best hyperparameters.
               - best_params: Dictionary of the best hyperparameters found.
               - best_score: The mean cross-validated score of the best_estimator.
    """
    param_grid = {}
    if kernel == 'linear':
        param_grid = {"C": expspace([-9, 3])}
    elif kernel == 'rbf':
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
        }
    elif kernel == 'poly':
        param_grid = {
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4],
            'gamma': [0.001, 0.01, 0.1, 'scale', 'auto'],
            'coef0': [0.0, 0.1, 0.5, 1.0] # Independent term in kernel function
        }
    else:
        raise ValueError(f"Unsupported kernel: {kernel}. Choose from 'linear', 'rbf', 'poly'.")

    svc = SVC(kernel=kernel, 
              random_state=random_state, 
              probability=True, 
              class_weight='balanced' if class_balancing else None) # probability=True for predict_proba if needed

    # Define cross-validation strategy
    if cv_method=='KFold':
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    elif cv_method=='StratifiedKFold':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    elif cv_method=='RepeatedKFold':
        cv = RepeatedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Setup GridSearchCV
    # Scoring can be 'accuracy', 'f1', 'roc_auc', etc. depending on the problem

    
    grid_search = GridSearchCV(estimator=svc, 
                            param_grid=param_grid, 
                            cv=cv,
                            scoring='roc_auc' if np.unique(y).shape[0] <= 2 else 'f1_weighted', # switch for binary | multiclass classification
                            verbose=3, 
                            n_jobs=-1) # n_jobs=-1 uses all available cores

    # Fit GridSearchCV
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters for {kernel} kernel: {best_params}")
    print(f"Best cross-validation accuracy: {best_score:.4f}")

    return best_model, best_params, best_score