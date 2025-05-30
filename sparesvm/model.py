import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR, SVR, SVC
from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
from .tuner import tune_svm_regressor_hyperparameters, tune_svm_classifier_hyperparameters

class SVMTrainer:
    """
    A class to train Support Vector Machine (SVM) models with various kernels,
    including hyperparameter tuning and cross-validation.
    """
    def __init__(self, 
                 task='classification',
                 kernel='linear', 
                 C=1.0, 
                 gamma='scale', 
                 degree=3, 
                 coef0=0.0,
                 tune_hyperparameters=False, 
                 cv_folds=5, 
                 random_state=None, 
                 scale_features=True):
        """
        Initializes the SVM_Classifier.

        Args:
            task (str): Type of task to be performed ('classification', 'regression')
            kernel (str): Type of SVM kernel ('linear', 'rbf', 'poly').
            C (float): Regularization parameter.
            gamma (str or float): Kernel coefficient for 'rbf' and 'poly'.
            degree (int): Degree for 'poly' kernel.
            coef0 (float): Independent term in kernel function for 'poly'.
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning.
            cv_folds (int): Number of cross-validation folds for tuning.
            random_state (int, optional): Random seed for reproducibility.
            scale_features (bool): Whether to apply StandardScaler to features.
        """
        if task not in ['classification', 'regression']:
            raise ValueError("Task must be one of 'classification' or 'regression'")

        if kernel not in ['linear', 'rbf', 'poly']:
            raise ValueError("kernel must be one of 'linear', 'rbf', or 'poly'")

        self.task = task
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scale_features = scale_features

        self.model = None
        self.best_params_ = None
        self.best_cv_score_ = None
        self._scaler = None # To store the scaler if features are scaled

    def _preprocess_data(self, features_df, labels_series=None):
        """
        Validates and preprocesses input data.
        Converts pandas DataFrame/Series to NumPy arrays.
        """
        if not isinstance(features_df, (pd.DataFrame, np.ndarray)):
            raise TypeError("Features must be a pandas DataFrame or NumPy array.")
        if labels_series is not None and not isinstance(labels_series, (pd.Series, np.ndarray)):
            raise TypeError("Labels must be a pandas Series or NumPy array.")

        X = features_df.values if isinstance(features_df, pd.DataFrame) else features_df
        y = None
        if labels_series is not None:
            y = labels_series.values if isinstance(labels_series, pd.Series) else labels_series
            if X.shape[0] != y.shape[0]:
                raise ValueError("Number of samples in features and labels must match.")

        return X, y

    def train(self, features, labels):
        """
        Trains the SVM model.

        Args:
            features (pd.DataFrame or np.ndarray): Training features.
                                                   Assumes rows are samples, columns are features.
            labels (pd.Series or np.ndarray): Training labels.
        """
        X, y = self._preprocess_data(features, labels)

        if y is None:
            raise ValueError("Labels are required for training.")

        if self.scale_features:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = X
        
        print(f"Training a {self.task} model...")
        
        if self.task == 'regression':
            if self.tune_hyperparameters:
                print(f"Starting hyperparameter tuning for {self.kernel} kernel...")
                best_model, best_params, best_score = tune_svm_regressor_hyperparameters(
                    X_scaled, 
                    y, 
                    kernel = self.kernel, 
                    cv_folds=self.cv_folds, 
                    random_state=self.random_state
                )
                self.model = best_model
                self.best_params_ = best_params
                self.best_cv_score_ = best_score
                print("Hyperparameter tuning complete. Training final model with best parameters.")
            
            else:
                print(f"Training SVR {self.task} model with kernel='{self.kernel}', C={self.C}, gamma='{self.gamma}', degree={self.degree}...")
                svm_params = {}
                if self.kernel == 'linear':
                    svm_params.update({'C': self.C})
                    self.model = LinearSVR(**svm_params)
                    self.model.fit(X_scaled, y)
                else:
                    if self.kernel == 'rbf':
                        svm_params.update({'kernel': 'rbf', 'C': self.C, 'gamma': self.gamma, 'shrinking': False})
                    elif self.kernel == 'poly':
                        svm_params.update({'kernel': 'poly', 'C': self.C, 'gamma': self.gamma,
                                            'degree': self.degree, 'coef0': self.coef0, 'shrinking': False})
                        self.model = SVR(**svm_params)
                        self.model.fit(X_scaled, y)
                        
                print("Model training complete.")
        
        elif self.task == 'classification':

            if self.tune_hyperparameters:
                print(f"Starting hyperparameter tuning for {self.kernel} kernel...")
                best_model, best_params, best_score = tune_svm_classifier_hyperparameters(
                    X_scaled, 
                    y, kernel = self.kernel, 
                    cv_folds=self.cv_folds, 
                    random_state=self.random_state
                )
                self.model = best_model
                self.best_params_ = best_params
                self.best_cv_score_ = best_score
                print("Hyperparameter tuning complete. Model trained with best parameters.")
            
            else:
                print(f"Training SVC {self.task} model with kernel='{self.kernel}', C={self.C}, gamma='{self.gamma}', degree={self.degree}...")
                svm_params = {'random_state': self.random_state, 'probability': True, 'class_weight' : 'balanced'}
                if self.kernel == 'linear':
                    svm_params.update({'kernel': 'linear', 'C': self.C})
                elif self.kernel == 'rbf':
                    svm_params.update({'kernel': 'rbf', 'C': self.C, 'gamma': self.gamma})
                elif self.kernel == 'poly':
                    svm_params.update({'kernel': 'poly', 'C': self.C, 'gamma': self.gamma,
                                    'degree': self.degree, 'coef0': self.coef0})

                self.model = SVC(**svm_params)
                self.model.fit(X_scaled, y)
                print("Model training complete.")

    def predict(self, features):
        """
        Makes predictions on new data.

        Args:
            features (pd.DataFrame or np.ndarray): New data for prediction.

        Returns:
            np.ndarray: Predicted labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        X_pred, _ = self._preprocess_data(features)

        if self.scale_features and self._scaler is not None:
            X_pred_scaled = self._scaler.transform(X_pred)
        else:
            X_pred_scaled = X_pred

        return self.model.predict(X_pred_scaled)

    def predict_proba(self, features):
        """
        Makes probability predictions on new data.
        Only works if probability=True was set during SVC initialization (default in this class).

        Args:
            features (pd.DataFrame or np.ndarray): New data for prediction.

        Returns:
            np.ndarray: Predicted probabilities for each class.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError("The underlying SVM model was not configured for probability estimates.")


        X_pred, _ = self._preprocess_data(features)

        if self.scale_features and self._scaler is not None:
            X_pred_scaled = self._scaler.transform(X_pred)
        else:
            X_pred_scaled = X_pred

        return self.model.predict_proba(X_pred_scaled)

    def get_model_details(self):
        """
        Returns details about the trained model.
        """
        if self.model is None:
            return "Model not trained yet."
        
        details = {
                "kernel": self.kernel,
                "tuned_hyperparameters": self.tune_hyperparameters
        }
        
        if self.task == 'classification':
            if self.tune_hyperparameters:
                details["best_parameters"] = self.best_params_
                details["best_cv_score (accuracy)"] = self.best_cv_score_
            else:
                details["parameters"] = self.model.get_params()

            if self.scale_features and self._scaler:
                details["feature_scaling_mean"] = self._scaler.mean_
                details["feature_scaling_scale"] = self._scaler.scale_
        
        elif self.task == 'regression':
            if self.tune_hyperparameters:
                details["best_parameters"] = self.best_params_
                details["best_cv_score (MAE)"] = self.best_cv_score_
            else:
                details["parameters"] = self.model.get_params()

            if self.scale_features and self._scaler:
                details["feature_scaling_mean"] = self._scaler.mean_
                details["feature_scaling_scale"] = self._scaler.scale_

        return details

    def save_model(self, filepath):
        """
        Saves the trained model (including scaler if used) to a file.
        Uses joblib for efficient saving of scikit-learn models.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        try:
            import joblib
            joblib.dump({'model': self.model, 'scaler': self._scaler if self.scale_features else None,
                         'best_params': self.best_params_, 'best_cv_score': self.best_cv_score_,
                         'kernel': self.kernel, 'scale_features': self.scale_features}, filepath)
            print(f"Model saved to {filepath}")
        except ImportError:
            print("joblib is not installed. Please install it to save the model: pip install joblib")
        except Exception as e:
            print(f"Error saving model: {e}")


    @classmethod
    def load_model(cls, filepath):
        """
        Loads a trained model (and scaler) from a file.
        """
        try:
            import joblib
            saved_data = joblib.load(filepath)
            if not all(k in saved_data for k in ['model', 'scaler', 'kernel', 'scale_features']):
                raise ValueError("Loaded model file is missing required components.")

            # Create an instance of SVM_Classifier. Some parameters are placeholders as they are overwritten.
            instance = cls(kernel=saved_data['kernel'],
                           tune_hyperparameters=bool(saved_data.get('best_params_')), # Infer if tuning was done
                           scale_features=saved_data['scale_features'])
            instance.model = saved_data['model']
            instance._scaler = saved_data['scaler']
            instance.best_params_ = saved_data.get('best_params_')
            instance.best_cv_score_ = saved_data.get('best_cv_score_')

            print(f"Model loaded from {filepath}")
            return instance
        except ImportError:
            print("joblib is not installed. Please install it to load the model: pip install joblib")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None